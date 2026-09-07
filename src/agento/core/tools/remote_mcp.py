"""Remote MCP servers, over the official Python SDK.

Model Context Protocol servers are how an agent reaches the outside world —
GitHub, Linear, Notion, Sentry, your own internal service. This module adapts one
to agento's :class:`~agento.core.tools.base.ToolSource`.

Install with::

    pip install "agento[mcp]"

Then::

    app = agento.Agento(
        llm=agento.LiteLLMProvider(),
        mcp={
            "github": agento.MCPServerConfig(
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": f"Bearer {token}"},
            ),
        },
    )
    agent = agento.Agent(
        model="openai/gpt-4o",
        mcp_servers=[agento.MCPServerRef(name="github", enable_tools=["@read-only"])],
    )

**Why there is a worker task in here.** The MCP SDK's transports are anyio
context managers, and anyio requires a cancel scope to be exited by the same task
that entered it. A naive implementation that opens the connection in one call and
closes it in another hits "Attempted to exit cancel scope in a different task" as
soon as anything runs concurrently — which, in an agent that fans out tool calls,
is immediately. So each connection owns one long-lived task: it opens the
session, serves requests from a queue, and closes the session itself. Callers
just await futures.

**Authorization.** ``headers`` may be an async callable, which is re-invoked on
every operation rather than once at connect. That is deliberate: an OAuth token
can expire or be revoked mid-session, and the caller needs to hear
``auth_required`` rather than a generic 401 from somewhere deep in a tool call.

.. note::
   This is one of three adapters that could not be runtime-tested where agento
   was written. Verify with ``python scripts/smoke.py`` before relying on it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from ...errors import McpConnectionError
from ..events import McpServerInit
from ..messages import ApprovalDecision, InternalToolInfo
from .base import (
    AuthRequiredOutcome,
    ToolAnnotations,
    ToolListing,
    ToolListOutcome,
    ToolOutcome,
    ToolSchema,
    ToolSuccess,
)

__all__ = ["HeaderResolver", "RemoteMCP"]

HeaderResolver = Callable[[], Awaitable[dict[str, str] | AuthRequiredOutcome]]
"""Async callable returning headers, or signalling that authorization is needed."""

_MAX_PAGES = 100


class _Request:
    """One operation queued for a connection's worker task."""

    __slots__ = ("op", "payload", "future", "started")

    def __init__(self, op: str, payload: dict[str, Any], future: asyncio.Future[Any]) -> None:
        self.op = op
        self.payload = payload
        self.future = future
        self.started = False


class _Connection:
    """A live MCP session, served by a dedicated task.

    The task opens the transport and session, publishes the session id, then
    serves requests until told to stop — at which point it closes the session
    from inside the same task, which is what anyio requires.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        headers: dict[str, str],
        transport: str,
        session_id: str | None,
        request_timeout: float,
        connect_timeout: float,
    ) -> None:
        self.name = name
        self.url = url
        self.headers = headers
        self.transport = transport
        self.session_id: str | None = session_id
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        self._queue: asyncio.Queue[_Request | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[Any] | None = None

    async def start(self) -> None:
        """Open the connection. Raises :class:`McpConnectionError` on failure."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = loop.create_task(self._run(), name=f"agento-mcp-{self.name}")
        try:
            await asyncio.wait_for(self._ready, timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise McpConnectionError(f"Timed out connecting to MCP server '{self.name}'", 504) from exc

    async def request(self, op: str, **payload: Any) -> Any:
        """Send an operation to the worker and await its result."""
        if self._task is None or self._task.done():
            raise McpConnectionError(f"MCP server '{self.name}' is not connected", 502)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        request = _Request(op, payload, future)
        await self._queue.put(request)
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError as exc:
            detail = "outcome unknown; reconcile before retrying" if request.started else "request never started"
            raise asyncio.TimeoutError(f"MCP {op} timed out: {detail}") from exc

    async def close(self) -> None:
        """Stop the worker and let it tear the session down itself."""
        if self._task is None:
            return
        await self._queue.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        except Exception:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        """Own the session for its whole lifetime."""
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - depends on install
            self._fail(
                ImportError(
                    "Remote MCP support requires the 'mcp' package. Install it with:\n"
                    '    pip install "agento[mcp]"'
                )
            )
            raise exc

        try:
            async with self._open_transport() as streams:
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self.session_id = self._read_session_id(streams)
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(True)
                    await self._serve(session)
        except Exception as exc:
            self._fail(exc)
        finally:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None and not item.future.done():
                    item.future.set_exception(McpConnectionError("MCP connection closed", 502))

    def _fail(self, exc: BaseException) -> None:
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(
                McpConnectionError(f"Failed to connect to MCP server '{self.name}': {exc}", 502)
            )

    @asynccontextmanager
    async def _open_transport(self) -> AsyncIterator[Any]:
        """Open the configured transport as an async context manager."""
        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(self.url, headers=self.headers) as streams:
                yield streams
            return
        from importlib import import_module

        streamable_http = import_module("mcp.client.streamable_http")

        modern = getattr(streamable_http, "streamable_http_client", None)
        if modern is not None:
            async with streamable_http.create_mcp_http_client(headers=self.headers) as client:
                async with modern(self.url, http_client=client) as streams:
                    yield streams
        else:
            legacy = streamable_http.streamablehttp_client
            async with legacy(self.url, headers=self.headers) as streams:
                yield streams

    @staticmethod
    def _read_session_id(streams: Any) -> str | None:
        """Streamable HTTP yields a session-id getter as its third stream item."""
        try:
            if len(streams) >= 3 and callable(streams[2]):
                value = streams[2]()
                return str(value) if value else None
        except Exception:
            pass
        return None

    async def _serve(self, session: Any) -> None:
        """Serve queued operations until asked to stop."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if item.future.done():
                # wait_for cancels an expired caller. A queued write must never
                # start after that caller has already been told it timed out.
                continue
            item.started = True
            try:
                if item.op == "list_tools":
                    result = await self._list_tools(session)
                elif item.op == "call_tool":
                    result = await session.call_tool(item.payload["name"], item.payload["arguments"])
                else:  # pragma: no cover - internal guard
                    raise ValueError(f"Unknown MCP operation: {item.op}")
                if not item.future.done():
                    item.future.set_result(result)
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)

    async def _list_tools(self, session: Any) -> list[Any]:
        """Walk every page of ``tools/list``.

        Guards against a server that returns a cursor it has already issued,
        which would otherwise loop forever.
        """
        tools: list[Any] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            try:
                page = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
            except TypeError:
                # Older SDKs have no cursor parameter; one page is all there is.
                page = await session.list_tools()
            tools.extend(getattr(page, "tools", []) or [])
            cursor = getattr(page, "nextCursor", None) or getattr(page, "next_cursor", None)
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
        return tools


class RemoteMCP:
    """A :class:`~agento.core.tools.base.ToolSource` backed by a remote MCP server.

    Policy-free by design — attach a
    :class:`~agento.core.tools.policy.PolicyToolSet` around it to apply one
    agent's enable/disable/approval rules, so several agents can share a single
    connection without leaking policy into each other.

    Args:
        name: How agents refer to this server.
        url: The server's endpoint.
        headers: Static headers, or an async callable re-invoked on every
            operation (for tokens that expire, or to signal ``auth_required``).
        description: Shown to the agent when this server's tools are deferred.
            Worth writing well: with deferred loading this sentence is *all* the
            agent knows before deciding whether to look inside.
        transport: ``"auto"`` (try streamable-http, fall back to SSE),
            ``"streamable-http"``, or ``"sse"``.
        session_id: Previous session metadata. Reconnect initializes a new server session.
        request_timeout: Seconds for one operation.
        connect_timeout: Seconds to establish the connection.
    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        headers: dict[str, str] | HeaderResolver | None = None,
        description: str = "",
        transport: str = "auto",
        session_id: str | None = None,
        request_timeout: float = 60.0,
        connect_timeout: float = 30.0,
        id: str | None = None,
    ) -> None:
        self._name = name
        self._id = id or name
        self._url = url
        self._headers = headers or {}
        self._description = description
        self._transport = transport
        self._session_id = session_id
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout

        self._connection: _Connection | None = None
        self._connect_lock = asyncio.Lock()
        self._tools: list[ToolSchema] | None = None
        self._reported_init = False

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    @property
    def description(self) -> str:
        return self._description

    @property
    def session_id(self) -> str | None:
        """The live session ID, retained as diagnostic metadata."""
        return self._connection.session_id if self._connection else self._session_id

    # -- connection --------------------------------------------------------- #

    async def _resolve_headers(self) -> dict[str, str] | AuthRequiredOutcome:
        if callable(self._headers):
            resolved = await self._headers()
            return resolved
        return dict(self._headers)

    async def _ensure_connected(self) -> _Connection | AuthRequiredOutcome:
        """Connect if needed, probing transports when set to ``auto``.

        Headers are resolved on every call, not only the first: an authorization
        that lapses mid-session must surface as ``auth_required`` rather than as
        an opaque failure inside a tool call.
        """
        headers = await self._resolve_headers()
        if isinstance(headers, AuthRequiredOutcome):
            return headers

        async with self._connect_lock:
            if self._connection is not None:
                task = self._connection._task
                if self._connection.headers == headers and task is not None and not task.done():
                    return self._connection
                await self.aclose()

            candidates = (
                ["streamable-http", "sse"] if self._transport == "auto" else [self._transport]
            )
            errors: list[str] = []
            for transport in candidates:
                connection = _Connection(
                    name=self._name,
                    url=self._url,
                    headers=headers,
                    transport=transport,
                    session_id=self._session_id,
                    request_timeout=self._request_timeout,
                    connect_timeout=self._connect_timeout,
                )
                try:
                    await connection.start()
                except McpConnectionError as exc:
                    await connection.close()
                    errors.append(f"{transport}: {exc}")
                    continue
                self._connection = connection
                return connection

            raise McpConnectionError(
                f"Could not connect to MCP server '{self._name}' ({'; '.join(errors)})",
                502,
            )

    async def aclose(self) -> None:
        """Close the connection. Safe to call more than once."""
        connection, self._connection = self._connection, None
        self._tools = None
        self._reported_init = False
        if connection is not None:
            self._session_id = connection.session_id
            await connection.close()

    # -- ToolSource surface ------------------------------------------------- #

    async def list_tools(self) -> ToolListOutcome:
        """List every tool the server offers, unfiltered.

        Cached for the life of the connection: a turn may consult the tool list
        many times and a server's catalogue does not change mid-turn.
        """
        connection = await self._ensure_connected()
        if isinstance(connection, AuthRequiredOutcome):
            return connection

        initialized: McpServerInit | None = None
        if not self._reported_init:
            self._reported_init = True
            initialized = McpServerInit(
                id=self._id,
                name=self._name,
                session_id=connection.session_id,
                transport=connection.transport,
            )

        if self._tools is None:
            raw_tools = await connection.request("list_tools")
            self._tools = [_to_schema(tool) for tool in raw_tools]

        return ToolListing(tools=list(self._tools), initialized=initialized)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        """Invoke a tool on the server."""
        connection = await self._ensure_connected()
        if isinstance(connection, AuthRequiredOutcome):
            return connection

        raw = await connection.request("call_tool", name=name, arguments=arguments)
        return _to_outcome(raw)

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        return InternalToolInfo(
            kind="mcp",
            source_id=self._id,
            source_name=self._name,
            original_tool_name=name,
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"RemoteMCP({self._name!r}, url={self._url!r})"


def _to_schema(raw: Any) -> ToolSchema:
    """Translate an MCP tool descriptor into agento's :class:`ToolSchema`."""
    get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
    input_schema = get("inputSchema") or get("input_schema") or {"type": "object", "properties": {}}
    if hasattr(input_schema, "model_dump"):
        input_schema = input_schema.model_dump()
    output_schema = get("outputSchema") or get("output_schema")
    if output_schema is not None and hasattr(output_schema, "model_dump"):
        output_schema = output_schema.model_dump()

    return ToolSchema(
        name=str(get("name", "")),
        description=str(get("description") or ""),
        input_schema=dict(input_schema),
        output_schema=dict(output_schema) if output_schema else None,
        annotations=ToolAnnotations.from_mcp(get("annotations")),
    )


def _to_outcome(raw: Any) -> ToolSuccess:
    """Flatten an MCP ``CallToolResult`` into a single string result.

    Content blocks other than text (images, embedded resources) are represented
    by a short placeholder rather than dumped into the conversation: a base64
    image in the message history is thousands of wasted tokens.
    """
    get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
    is_error = bool(get("isError") or get("is_error") or False)

    structured = get("structuredContent") or get("structured_content")
    if structured is not None:
        return ToolSuccess(
            content=json.dumps(structured, default=str),
            is_error=is_error,
            is_structured=True,
        )

    blocks = get("content") or []
    parts: list[str] = []
    for block in blocks:
        block_get = (
            block.get if isinstance(block, dict) else lambda key, default=None, block=block: getattr(block, key, default)
        )
        block_type = block_get("type", "text")
        if block_type == "text":
            parts.append(str(block_get("text", "")))
        elif block_type == "resource":
            resource = block_get("resource")
            resource_get = (
                resource.get
                if isinstance(resource, dict)
                else lambda key, default=None, resource=resource: getattr(resource, key, default)
            )
            text = resource_get("text") if resource is not None else None
            parts.append(str(text) if text else f"[resource: {resource_get('uri', 'unknown')}]")
        else:
            parts.append(f"[{block_type} content omitted]")

    return ToolSuccess(content="\n".join(parts), is_error=is_error)
