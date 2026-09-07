# Events

`turn.stream()` yields events. This is agento's real public contract — if you are
integrating it into an application, this page is the one to read.

```python
async for event in turn.stream():
    match event:
        case agento.ModelMessageDelta(content=text) if text:
            await websocket.send_text(text)
        case agento.ApprovalRequired():
            await ask_a_human(event)
        case agento.TurnDone(state=state):
            await mark_complete(state)
```

Every event is a pydantic model, so `event.model_dump()` gives you JSON for a
websocket or an SSE frame without any conversion of your own.

---

## Durable vs transient

**Durable** events are written to the session store *before* they are yielded.
Replay them with `turn.list_events()` or `session.list_events()`. Complete model
messages, tool results, lifecycle, artifact, and compaction events are durable.
The opening empty `ModelMessage` and token deltas are transient. Standalone
approval/client-action signals are streamed; their required actions are preserved
in the terminal `TurnDone` state for replay.

**`ModelMessageDelta` is transient.** It is the token-by-token stream, and it is
never persisted, because the complete `ModelMessage` that follows carries
everything it did. Render deltas for responsiveness; treat the `ModelMessage` as
the truth.

---

## Ordering

`event.id` is a monotonic ULID: lexicographic order **is** creation order, within
a stream and after a reload. Sort by `id`, never by `created_at` — two events in
the same millisecond have the same timestamp and different ids.

---

## The events

### `TurnCreated`

First on the live execution stream. Carries `turn_id`, `previous_turn_id` and
the input that started the turn. Upload events can have earlier IDs because
attachments are prepared before execution; stored history is ordered by ID.

### `ModelMessage`

Emitted **twice per model call**, and the difference matters:

- **At the start**, carrying only `id` — this opens a delta stream.
- **At the end**, fully populated with content, tool calls, finish reason and
  usage.

Every `ModelMessageDelta` in between shares that `id`. So a client creates one
message bubble on the first, appends deltas to it, and replaces its contents with
the second.

```python
if isinstance(event, agento.ModelMessage):
    if event.content is None and not event.tool_calls:
        ui.start_message(event.id)          # placeholder
    else:
        ui.finish_message(event.id, event)  # complete
```

`usage.input_tokens_breakdown` splits the input tokens across `harness`,
`instructions`, `tool_definitions`, `skills` and `messages` — estimates, but the
fastest way to answer "why is this agent expensive?".

### `ModelMessageDelta`

One streaming chunk: `content`, `reasoning_content`, tool-call fragments, and on
the final frame `finish_reason` and `usage`.

Tool-call fragments arrive keyed by index, with the id and name in the first and
the arguments accumulating across many. Rendering "calling `search`…" from the
first fragment is fine; do not try to parse partial arguments.

### `ToolResult`

One executed tool call. `content` is what the model sees — already shortened if
the result was offloaded, with `artifact_id` pointing at the full payload.

`is_error` means the tool ran and failed. The content still reaches the model,
because a model that can see the error usually fixes its arguments and retries.

### `ThreadCreated` / `ThreadDone`

A sub-agent started and finished. `ThreadDone.state` is `done` with the child's
final message, or `error` with a reason. The root thread does not emit these —
its outcome is the turn's outcome.

Use `thread_id` to render delegated work in its own pane; `"main"` is the root.

At a child join, the parent reply and child retirement are checkpointed together
with `ToolResult` and `ThreadDone`. The live order is `ToolResult`, then
`ThreadDone`. Both remain available in stored history if the stream closes
between them. Older already-joined snapshots are retired without replaying work;
when their terminal details are absent, recovery cannot reconstruct the missing
`ThreadDone` status. See [recovery](operations.md).

### `ApprovalRequired`

One or more tool calls need a human decision. The turn ends here.

```python
resumed = await session.create_turn([
    agento.ToolApproval(
        thread_id=event.thread_id,
        tool_call_id=call.id,
        decision="allow",          # or "deny"
        reason=None,               # shown to the model when denying
    )
    for call in event.tool_calls
])
```

Answer **every** pending call in one batch — a partial batch is rejected with
`InvalidSendInputError`, because a half-answered pause has no well-defined
continuation.

### `ClientToolRequired`

Tool calls your application must execute — a genuine client-side tool, or the
built-in `ask_user_question`. Answer with `ToolReply`:

```python
resumed = await session.create_turn([
    agento.ToolReply(
        thread_id=event.thread_id,
        tool_call_id=call.id,
        content="the user chose: refund in full",
    )
    for call in event.tool_calls
])
```

For `ask_user_question`, the arguments on the originating `ModelMessage` carry
`question` and `options` — render them however you like.

### `McpInitialized` / `McpAuthRequired`

Servers that connected (with session IDs kept as diagnostic metadata), and
servers that need authorization. Each `McpAuthRequired` entry carries an
`auth_url` to send the user to; once authorized, a new turn proceeds.

### `ArtifactCreated`

Content was written to the artifact store — an offloaded tool result, or a file
the user attached that could not go inline. `artifact_id`, `name`, `size_bytes`
and `source_tool`. Your application can fetch it from the store directly; the
agent reads it with `read_artifact` and `search_artifact`.

### `ContextCompacted`

The conversation was summarized and the older history replaced. Informational,
but worth surfacing: the agent now works from a summary, so a user asking "what
did I say at the start?" may get a condensed answer even though the full event
log is intact.

### `TurnDone`

Last on a normally consumed stream. Explicit closure/cancellation finalizes in
storage without yielding during cleanup; storage outages can prevent finalization.
`state` is one of:

| Status | Meaning |
| --- | --- |
| `done` | Finished — **or paused**. Check `required_actions`. |
| `cancelled` | Stopped, with a `reason`. |
| `error` | Failed, with a `message`. |

`done` with a non-empty `required_actions` is the case worth being careful about:
the agent did not fail, it is waiting for you.

```python
if state.status == "done" and state.required_actions:
    await handle_pause(state.required_actions)
elif state.status == "done":
    await deliver(state.output)
```

`state.metrics` carries tokens, cost, iterations, tool calls, sub-agents and
compactions for the whole turn, summed across every thread.

---

## Input items

What goes **into** `create_turn`:

| Type | Answers |
| --- | --- |
| `UserMessage` | — (a new instruction; a plain string is shorthand) |
| `ToolApproval` | `ApprovalRequired` |
| `ToolReply` | `ClientToolRequired` |

A batch must be homogeneous: all user messages, or all approvals and replies.
Mixing them is rejected, because "here is my answer, and also a new request" has
no defined ordering against the pending call.

Attachments go on a `UserMessage`:

```python
await session.create_turn(agento.UserMessage(content=[
    agento.TextPart(text="What changed between these?"),
    agento.FilePart(name="q3.csv", data="data:text/csv;base64,..."),
]))
```

Images and PDFs go to the model inline. Anything else is stored as an artifact
and the model is told the id — so a 40 MB CSV is readable without being pasted
into the context window.

---

## A complete handler

```python
async def run_turn(session, user_input) -> dict:
    turn = await session.create_turn(user_input)
    text_parts: list[str] = []

    async for event in turn.stream():
        match event:
            case agento.ModelMessageDelta(content=text) if text:
                text_parts.append(text)
                await push_to_client(text)

            case agento.ToolResult(is_error=True, content=content):
                log.warning("tool failed", content=content[:200])

            case agento.ThreadCreated(agent_info=info):
                await push_status(f"delegating to {info.name}…")

            case agento.ApprovalRequired() | agento.ClientToolRequired():
                await push_pause(event)

            case agento.TurnDone(state=state):
                return {"status": state.status, "text": "".join(text_parts)}

    return {"status": "unknown", "text": "".join(text_parts)}
```
