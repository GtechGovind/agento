# Operating agento

agento is an embedded library. The host application runs the worker, authenticates
users, authorizes access to sessions and artifacts, and supervises execution.
Start with the [validation record](validation.md); passing local tests does not
establish readiness for your provider, workload, or deployment.

## Own the execution lifetime

`create_turn()` prepares a stored turn; `stream()` or `drain()` executes it once.
An unconsumed turn stays `running`. Keep a strong reference to background tasks,
await them during graceful shutdown, and surface their exceptions. A client
disconnection does not independently create a background worker.

```python
workers = set()

async def submit(session, user_input):
    turn = await session.create_turn(user_input)
    task = asyncio.create_task(turn.drain())
    workers.add(task)
    task.add_done_callback(workers.discard)
    return turn.id
```

This sketch shows ownership, not a complete job queue: your supervisor must also
collect exceptions, handle shutdown, and recover work after process death. For
streaming requests, use `contextlib.aclosing(turn.stream())` on early exit.

## Cancellation and recovery

| Situation | Runtime behavior | Application action |
| --- | --- | --- |
| Normal completion | Final state and event are stored together | Inspect `state.status` and `required_actions` |
| Human approval needed | Turn finishes with pending required actions | Collect all decisions and create the next turn |
| `turn.cancel()` | Cooperative stop checked between execution steps | Wait for the stream/task to finish |
| Task cancellation or `stream.aclose()` | Cleanup attempts a cancelled terminal checkpoint | Await cleanup and inspect stored state |
| Process crash | No Python cleanup can run; the turn may remain running | Confirm the worker is dead, reconcile tools, explicitly cancel the stored turn |
| Database unavailable | Persistence raises; terminal storage cannot be guaranteed | Alert, restore storage, inspect the last committed checkpoint |
| MCP request expires in its queue | The request is skipped before execution | Retry only if the operation is still wanted |
| MCP timeout after dispatch | Outcome is unknown | Query operation status before retrying |

`session.cancel_active_turn()` records cancellation. It does **not** stop another
process or undo an external action. Coordinate worker shutdown before using it
for recovery. A new turn rejects an active parent/tip with
`PreviousTurnRunningError`; it no longer silently cancels it.

An interrupted tool call with no durable result is repaired with an
**outcome-unknown** result. It may have executed. The model is instructed to
reconcile, but this instruction is not an exactly-once execution mechanism.

## Make side effects idempotent

Use the tool context's `tool_call_id` to correlate attempts and maintain an
operation record in your service. A model can issue a fresh call ID for the same
business action, so important actions also need a stable business operation key.

```python
@agento.tool(destructive=True)
async def dispatch_order(order_id: str, ctx: agento.ToolContext) -> str:
    """Dispatch an approved order exactly once through the order service."""
    # order_service must implement idempotency and authorization itself.
    return await order_service.dispatch(
        order_id=order_id,
        operation_key=f"dispatch:{order_id}",
        trace_id=ctx.tool_call_id,
        actor_id=ctx.metadata["user_id"],
    )
```

The service in this integration sketch is supplied by your application. Check
the authoritative service record after an ambiguous timeout. Do not implement
blanket retries around tool execution, `session.run`, or turn creation.
Provider SDK retry behavior is configured on the provider adapter/client; the
optional `tenacity` dependency does not install a universal retry policy.

## Storage and concurrency

Use one active executor per session. Handles refresh the current tip before
preparing a turn; stores compare that expected tip while creating the turn. A
competing creation fails with `SessionStoreConflictError`. Reload and reassess
the request rather than silently branching. Explicit branching remains available.

Memory stores are process-local and disposable. SQLite suits a local deployment;
PostgreSQL has a dedicated driver extra. Test the workload on your chosen backend.
SQL writes serialize updates on the session row before changing its tip, turn
state, or accumulated metrics. Terminal turn fields are immutable.

Run schema creation or migrations separately from general request handling.
`schema_sql()` emits table DDL; include the metadata's indexes in a managed
migration too. There is no automatic versioned migration framework here. Back up
session, turn, and event tables together and test restoration before upgrading.
Local artifact storage is separate from SQL: back up its root too. Token/cost
metrics are folded at terminal transition; abrupt process death can leave partial
usage unaccounted for. Provider billing remains authoritative.

## MCP connections

MCP connections are scoped to a turn's resource resolver and closed with it.
Headers can come from an async resolver. If headers change, the adapter reconnects
and refreshes its tool catalogue; an authorization-required outcome pauses use.
Treat header changes as a connection boundary when operations are in flight.

Stored MCP session IDs are diagnostic metadata. **The current adapter initializes
a fresh server session on reconnect**; it does not resume a remote server's
application state. Keep durable business state in your service.

## Data and trust boundaries

- Authorize session IDs and external IDs in the host. Metadata provides context,
  not access control. Tool code must check the authenticated actor's permissions.
- Tools execute in the host process or worker threads. This is not a sandbox;
  expose only the operations the agent is allowed to perform.
- Keep artifact and skill roots under host ownership. Artifact IDs reject path
  traversal and existing symlinks; the store is not an isolation boundary against
  another process that can concurrently replace files in its root.
- Model/tool text, prompts, and attachments can contain private data. Apply your
  retention and redaction policy to snapshots, event logs, artifacts, and traces.
  The tracing adapter can record model/tool output; use a redacting tracer when needed.
- Uploaded artifacts may be created before turn validation/storage completes.
  Artifact storage and SQL do not share a transaction. Provide retention or
  orphan cleanup for interrupted preparation; session deletion does not remove
  separate artifact files automatically.
- Limit input sizes, tool runtime, model budget, and queued requests in the host.
  Synchronous tools may continue in a worker thread after task cancellation.

## Release acceptance

Before exposing a deployment, verify live provider streaming, approvals and
recovery, tool idempotency, remote MCP authentication, database contention,
backup/restore, retention, load, and worker shutdown under your configuration.
Record versions and evidence in your release notes. MySQL support is not part of
the currently verified backend matrix.
