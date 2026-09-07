"""Offline provider contracts and a real MCP SDK round trip on loopback."""
from __future__ import annotations

import asyncio
import json
import socket
import sys
from types import SimpleNamespace as Obj

import pytest

import agento
from agento.core.llm.base import LLMRequest
from agento.core.tools.base import AuthRequiredOutcome
from agento.core.tools.remote_mcp import RemoteMCP, _Connection, _to_outcome, _to_schema


def connection(**kwargs):
    return _Connection(name='local', url='http://127.0.0.1:1/mcp', headers={},
        transport='streamable-http', session_id=None, request_timeout=0.025, connect_timeout=1, **kwargs)


async def test_expired_mcp_queue_never_starts_a_write():
    release = asyncio.Event()
    effects = []

    class Session:
        async def call_tool(self, name, arguments):
            if name == 'slow':
                await release.wait()
            effects.append(name)
            return 'done'

    conn = connection()
    conn._task = asyncio.create_task(conn._serve(Session()))
    results = await asyncio.gather(conn.request('call_tool', name='slow', arguments={}),
        conn.request('call_tool', name='write', arguments={}), return_exceptions=True)
    assert all(isinstance(result, asyncio.TimeoutError) for result in results)
    assert 'outcome unknown' in str(results[0])
    assert 'never started' in str(results[1])
    release.set()
    await conn.close()
    assert effects == ['slow']
    await conn.close()
    with pytest.raises(agento.errors.McpConnectionError):
        await conn.request('list_tools')


async def test_mcp_pagination_stops_on_repeated_cursor():
    calls = []

    class Session:
        async def list_tools(self, cursor=None):
            calls.append(cursor)
            return Obj(tools=[{'name': f'page{len(calls)}'}], nextCursor='repeat')

    tools = await connection()._list_tools(Session())
    assert len(tools) == 2
    assert calls == [None, 'repeat']


@pytest.mark.parametrize('wrap', [lambda x: x, lambda x: Obj(**x)])
def test_mcp_descriptors_and_content(wrap):
    schema = _to_schema(wrap({'name': 'lookup', 'description': 'Read', 'inputSchema': {'type': 'object'},
                             'annotations': {'readOnlyHint': True}}))
    assert schema.annotations.read_only
    result = _to_outcome(wrap({'content': [wrap({'type': 'text', 'text': 'hello'}),
        wrap({'type': 'resource', 'resource': wrap({'uri': 'test:x', 'text': 'resource'})}),
        wrap({'type': 'image', 'data': 'omitted'})], 'isError': True}))
    assert result.is_error
    assert result.content == 'hello\nresource\n[image content omitted]'
    result = _to_outcome(wrap({'structuredContent': {'value': 7}}))
    assert result.is_structured and json.loads(result.content) == {'value': 7}


async def test_mcp_http_round_trip_and_credentials_refresh():
    pytest.importorskip('mcp')
    uvicorn = pytest.importorskip('uvicorn')
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer

    server_mcp = MCPServer('local-contract')

    @server_mcp.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(server_mcp.streamable_http_app(), log_level='error'))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    current_headers = {'X-Contract': 'first'}
    auth_required = False

    async def headers():
        if auth_required:
            return AuthRequiredOutcome()
        return dict(current_headers)

    remote = RemoteMCP('local', f'http://127.0.0.1:{port}/mcp', headers=headers,
                       transport='streamable-http', connect_timeout=2, request_timeout=2)
    try:
        async def wait_started():
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError('MCP server stopped during startup')
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_started(), 5)
        listing = await remote.list_tools()
        assert [tool.name for tool in listing.tools] == ['add']
        assert listing.initialized is not None
        assert (await remote.list_tools()).initialized is None
        result = await remote.call_tool('add', {'a': 2, 'b': 3})
        assert '5' in result.content
        previous_connection = remote._connection
        current_headers['X-Contract'] = 'second'
        await remote.list_tools()
        assert remote._connection is not previous_connection
        assert remote._connection.headers == current_headers
        info = await remote.tool_info('add')
        assert info.original_tool_name == 'add'
        auth_required = True
        assert isinstance(await remote.list_tools(), AuthRequiredOutcome)
    finally:
        await remote.aclose()
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


async def test_openai_sdk_parses_stream_and_usage_with_offline_http_transport():
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    from agento.core.llm.openai_client import OpenAIClient, OpenAIProvider

    requests = []
    chunks = [
        {'choices': [{'index': 0, 'delta': {'content': 'hello'}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call1', 'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]},
        {'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10,
            'prompt_tokens_details': {'cached_tokens': 2}, 'completion_tokens_details': {'reasoning_tokens': 1}}},
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        payload = ''.join('data: ' + json.dumps({'id': 'test', 'object': 'chat.completion.chunk',
            'created': 1, 'model': 'local', **chunk}) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=payload.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        sdk = openai.AsyncOpenAI(api_key='offline-test', http_client=http)
        provider = OpenAIProvider(client=sdk)
        client = provider('local')
        assert provider('local') is client
        request = LLMRequest(messages=[{'role': 'assistant', 'content': 'old', 'thinking_blocks': []}],
                             params={'temperature': 0}, response_format={'type': 'json_object'})
        events = [event async for event in client.stream(request)]
        assert events[0].content == 'hello'
        assert events[1].tool_calls[0].name == 'lookup'
        assert events[-1].usage.total_tokens == 10
        assert events[-1].usage.cache_read_tokens == 2
        assert events[-1].usage.reasoning_tokens == 1
        assert 'thinking_blocks' not in requests[0]['messages'][0]
        assert requests[0]['stream_options']['include_usage']
        assert OpenAIClient('local', client=sdk)._build_kwargs(request)['temperature'] == 0


async def test_litellm_adapter_request_stream_and_completion(monkeypatch):
    if sys.version_info >= (3, 11):
        pytest.importorskip('litellm')
    from agento.core.llm import litellm_client as module

    seen = []
    message = {'content': 'answer', 'thinking_blocks': [{'type': 'thinking', 'thinking': 'reason'}],
        'tool_calls': [{'id': 'call', 'index': 0, 'function': {'name': 'lookup', 'arguments': '{}'}}]}
    usage = {'input_tokens': 7, 'output_tokens': 3, 'cache_read_input_tokens': 2,
             'cache_creation_input_tokens': 1}

    async def acompletion(**kwargs):
        seen.append(kwargs)
        if not kwargs['stream']:
            return Obj(choices=[Obj(message=message, finish_reason='tool_use')], usage=usage,
                       _hidden_params={'response_cost': 0.01})

        async def stream():
            yield {'choices': [{'delta': message, 'finish_reason': 'tool_use'}], 'usage': usage}
        return stream()

    fake = Obj(acompletion=acompletion, get_model_info=lambda model: {'max_input_tokens': 8192})
    monkeypatch.setattr(module, '_import_litellm', lambda: fake)
    client = module.LiteLLMClient('local', api_key='offline', api_base='http://example.invalid',
                                 properties=agento.ModelProperties(context_length=8192))
    request = LLMRequest(messages=[{'role': 'user', 'content': 'go'}], tools=[{'type': 'function'}],
                         response_format={'type': 'json_object'})
    streamed = [chunk async for chunk in client.stream(request)]
    assert streamed[0].finish_reason == 'tool_calls'
    assert streamed[0].thinking_blocks[0].thinking == 'reason'
    completed = await client.complete(request)
    assert completed.message.content == 'answer'
    assert completed.usage.cost_usd == 0.01
    assert completed.usage.cache_write_tokens == 1
    assert seen[0]['api_base'] == 'http://example.invalid'
    assert seen[1]['stream'] is False


def test_litellm_python_requirement_is_explicit(monkeypatch):
    from agento.core.llm import litellm_client as module

    monkeypatch.setattr(module, 'sys', Obj(version_info=(3, 10)))
    with pytest.raises(ImportError, match='requires Python 3.11'):
        module._import_litellm()
