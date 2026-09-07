# API guide

Import public objects from `agento`; implementation modules are useful for
reading source but should not be your default import path. Errors are in
`agento.errors`. The distribution includes `py.typed` for type checkers.

## Application surface

| Object / call | Purpose | Execution |
| --- | --- | --- |
| `Agento(llm=..., store=..., artifacts=..., skills=..., mcp=...)` | Bind host resources and adapters | Configuration |
| `Agent(name=..., model=..., instructions=..., tools=..., config=...)` | Describe one agent | Configuration |
| `await app.run(agent, input)` | One throwaway session; return final text | Runs immediately |
| `await app.stream(agent, input)` | Return an async event iterator | Runs when iterator is consumed |
| `app.run_sync(agent, input)` | Blocking convenience outside an event loop | Runs immediately |
| `await app.sessions.create(agent=...)` | Create a stored conversation | Preparation |
| `await app.sessions.get(id, agent=...)` | Load a conversation; optionally bind tools | Read |
| `await app.sessions.get_or_create_by_external_id(key, agent=...)` | Bind conversation to a host key | Atomic uniqueness in store |
| `await session.create_turn(input, previous_turn_id="auto")` | Prepare an executable turn | Does not execute model/tools |
| `await session.run(input)` | Create, run, return final text | Runs immediately |
| `await session.stream(input)` | Return an async event iterator | Runs when iterator is consumed |
| `turn.stream()` | Async generator of typed events | Consume once |
| `await turn.drain()` / `await turn.wait()` | Run and return terminal state | Consume once |
| `await turn.final_output()` | Run and return final text | Consume once |
| `await session.get_turn(id)` | Load a read-only turn handle | Cannot execute it |
| `await turn.list_events(...)` | Read the durable log | Safe during or after execution |

`run()` and `final_output()` can return an empty string on a paused or errored
turn. Use an explicit turn and inspect its state when failures must be surfaced.
`wait()` is an execution alias, not a way to join an already-running stream.

## Input and output types

`create_turn()` accepts a string, `UserMessage`, `ToolApproval`, `ToolReply`, a
sequence of inputs, or `None` to continue when no user action is pending.
Resolve pending approvals/replies as complete batches for the target thread.
See [events](events.md) for field-level examples.

A `UserMessage` contains text or `TextPart`, `ImagePart`, and `FilePart` objects.
Images use URLs/data URLs. Files use a name and a base64 data URI. Images/PDFs are
converted to provider chat content parts; other files go to the artifact store.
Configure an artifact store when accepting those uploads.

Turn states are `running`, `done`, `cancelled`, and `error`. A `done` turn with
`required_actions` is paused for external input. See [operations](operations.md).

## Extension contracts

| Contract | Built-in implementations | Responsibility |
| --- | --- | --- |
| `LLM` / `LLMProvider` | `ScriptedLLM`, OpenAI, LiteLLM | Requests, streamed chunks, usage |
| `ToolSource` / `ToolSet` | Local functions, MCP, client-side tools, policy wrappers | Tool catalogues, schemas, invocation |
| `SessionStore` | `MemorySessionStore`, `SQLSessionStore` | Atomic turn transitions and event history |
| `ArtifactStore` | `MemoryArtifactStore`, `LocalArtifactStore` | Content storage and metadata |
| `SkillSource` | `FileSkillSource` | Skill discovery and resource reads |
| `Capability` | Built-in context and tool capabilities | Lifecycle hooks and durable state |
| `Tracer` | `NoopTracer`, `OTelTracer` | Spans around execution |

Source signatures and docstrings are the detailed reference:
[session facade](../src/agento/session/agento.py),
[agent configuration](../src/agento/session/agent.py),
[store protocol](../src/agento/session/store/base.py),
[provider contract](../src/agento/core/llm/base.py),
[tool contract](../src/agento/core/tools/base.py).

## Errors and ownership

Catch `AgentoError` for classified framework errors. Handle
`PreviousTurnRunningError`, `SessionStoreConflictError`, `TurnNotRunningError`,
and `ConfigurationError` according to their meaning; they are not universal
retry signals. Ordinary provider/storage exceptions can also propagate.

The turn closes its resource resolver and MCP connections. The host owns its
SQL store and any supplied provider SDK clients: dispose/close them during
shutdown. A reloaded session using Python tools needs the live `Agent` definition.

## Pre-release store-contract update

The durability repair adds optional parameters while preserving existing
application call shapes:

```python
await store.create_turn(record, expected_tip=(last_turn_id,))
await store.update_turn(session_id, turn_id, snapshot=snapshot, events=events)
```

`expected_tip=None` skips the comparison for direct store callers. `(None,)`
means "the session must have no tip"; `(turn_id,)` requires that exact tip.
`SessionHandle` always supplies the comparison. A store must reject an active
session tip, including fresh-root and explicit-branch creation.

`update_turn(..., events=...)` commits all supplied fields and events together
or changes nothing, including metrics. Every mutation after a terminal state
raises `TurnNotRunningError`, even if the proposed terminal state is identical.
Custom implementations must accept and uphold these parameters before use with
this runtime. No database schema change is required for the bundled stores.
