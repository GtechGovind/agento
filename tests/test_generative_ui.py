"""Exercise instruction delivery and the opaque-text UI boundary through the runtime."""

from __future__ import annotations

from helpers import build_agent, build_app, collect, streamed_text

import agento
from agento.core.capabilities.builtins.generative_ui import render_openui_specification
from agento.core.tools.base import ToolListing

SECTION_TAGS = (
    'openui-fencing',
    'openui-syntax',
    'openui-components',
    'openui-builtins',
    'openui-streaming',
    'openui-examples',
    'openui-rules',
    'openui-verification',
)

UI_MESSAGE = '''```openui
root = Stack([Button("Inspect", Action([@ToAssistant("Inspect the sample")]))])
```'''


def _agent(*, preload: bool) -> agento.Agent:
    agent = build_agent()
    agent.config.generative_ui = agento.GenerativeUIConfig(enabled=True, preload=preload)
    return agent


async def test_default_guide_tool_keeps_its_public_discovery_contract() -> None:
    capability = agento.GenerativeUI()
    assert capability.name == 'generative_ui'
    tool_sets = capability.tool_sets()
    assert len(tool_sets) == 1
    assert tool_sets[0].name == 'openui'
    listing = await tool_sets[0].list_tools()
    assert isinstance(listing, ToolListing)
    assert len(listing.tools) == 1
    tool = listing.tools[0]
    assert tool.name == 'get_openui_instructions'
    assert tool.annotations is not None and tool.annotations.read_only is True
    assert tool.input_schema['type'] == 'object'
    assert tool.input_schema.get('properties', {}) == {}
    assert tool.input_schema.get('required', []) == []


async def test_preloaded_guide_reaches_the_model_without_an_instruction_tool() -> None:
    artifacts = agento.MemoryArtifactStore()
    app, llm = build_app([UI_MESSAGE], artifacts=artifacts)
    session = await app.sessions.create(agent=_agent(preload=True))
    turn = await session.create_turn('Present a view')
    events = await collect(turn.stream())

    assert len(llm.requests) == 1
    request = llm.requests[0]
    prompt = '\n'.join(message['content'] for message in request.messages if message['role'] == 'system')
    specification = render_openui_specification()
    assert f'<openui>\n{specification}\n</openui>' in prompt
    for tag in SECTION_TAGS:
        assert prompt.count(f'<{tag}>') == 1
        assert prompt.count(f'</{tag}>') == 1
    assert all(tool['function']['name'] != 'get_openui_instructions' for tool in request.tools or [])
    assert agento.GenerativeUI(preload=True).tool_sets() == []
    assert streamed_text(events) == UI_MESSAGE
    assert turn.state.output.content == UI_MESSAGE
    assert not any(isinstance(event, (agento.ToolResult, agento.ArtifactCreated)) for event in events)
    assert await artifacts.list() == []


async def test_deferred_guide_is_complete_in_tool_event_and_next_model_context() -> None:
    artifacts = agento.MemoryArtifactStore()
    app, llm = build_app(
        [agento.say(tool_calls=[('get_openui_instructions', {})]), UI_MESSAGE],
        artifacts=artifacts,
    )
    session = await app.sessions.create(agent=_agent(preload=False))
    turn = await session.create_turn('Prepare an interactive response')
    events = await collect(turn.stream())

    assert len(llm.requests) == 2
    first = llm.requests[0]
    first_prompt = '\n'.join(message['content'] for message in first.messages if message['role'] == 'system')
    assert '<openui>' in first_prompt and 'get_openui_instructions' in first_prompt
    assert all(f'<{tag}>' not in first_prompt for tag in SECTION_TAGS)
    exposed = [tool for tool in first.tools or [] if tool['function']['name'] == 'get_openui_instructions']
    assert len(exposed) == 1
    assert exposed[0]['function']['parameters'].get('properties', {}) == {}

    results = [event for event in events if isinstance(event, agento.ToolResult)]
    assert len(results) == 1
    specification = render_openui_specification()
    assert results[0].content == specification
    for tag in SECTION_TAGS:
        assert f'<{tag}>' in results[0].content and f'</{tag}>' in results[0].content
    delivered = [message['content'] for message in llm.requests[1].messages if message['role'] == 'tool']
    assert delivered == [specification]
    assert streamed_text(events) == UI_MESSAGE
    assert turn.state.output.content == UI_MESSAGE
    assert turn.state.status == 'done'
    assert not any(isinstance(event, agento.ArtifactCreated) for event in events)
    assert await artifacts.list() == []


async def test_ui_fences_are_not_parsed_or_executed_by_the_runtime() -> None:
    # An intentionally unknown component makes this a check of the transport
    # boundary, not an assertion that a renderer accepts the program.
    unparsed = '```openui\nroot = UnknownWidget(@OpenUrl("https://example.invalid"))\n```'
    app, llm = build_app([unparsed])
    session = await app.sessions.create(agent=_agent(preload=False))
    turn = await session.create_turn('Return the supplied text')
    events = await collect(turn.stream())

    assert len(llm.requests) == 1
    assert streamed_text(events) == unparsed
    assert turn.state.output.content == unparsed
    assert turn.state.status == 'done'
    assert not any(isinstance(event, (agento.ToolResult, agento.ArtifactCreated)) for event in events)
