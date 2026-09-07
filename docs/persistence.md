# Persistence

agento keeps two things durable — the **event log** (what happened) and the
**turn snapshot** (enough state to resume). Everything else is derived.

```python
# In-memory: tests, scripts, anything that need not outlive the process
app = agento.Agento(llm=...)

# SQLite / Postgres / MySQL
store = agento.SQLSessionStore("sqlite+aiosqlite:///./agento.db")
await store.create_tables()
app = agento.Agento(llm=..., store=store)
```

---

## The shape

```
Session          one conversation
 ├── Turn        one exchange; turns form a tree
 │    ├── snapshot   thread state — how the next turn resumes
 │    └── events     append-only log, ordered by ULID
 └── metadata    your identifiers, visible to every tool
```

**A session** holds the agent it runs, a title, your `metadata`, an optional
`external_id`, and `last_turn_id` — the tip.

**A turn** holds its input, its state, and a snapshot of every thread's messages.
`previous_turn_id` is a *parent pointer*, so turns form a tree rather than a list.

**Events** are append-only and primary-keyed by a monotonic ULID, which is also
the ordering key: lexicographic order is creation order. Sort by `id`, never by
timestamp.

---

## Sessions

```python
session = await app.sessions.create(
    agent=agent,
    metadata={"user_id": "u_42", "tenant": "acme"},   # visible to tools
    external_id="slack:C0123:1699999999.123456",      # your key, unique
)

session = await app.sessions.get(session_id)
session, created = await app.sessions.get_or_create_by_external_id(thread_key, agent=agent)
```

`get_or_create_by_external_id` is what makes agento fit an existing application:
you already have an identifier for the conversation — a Slack thread, a ticket, a
document — and you want the agent's memory keyed to it without a second mapping
table. It is race-safe, so two events for the same thread arriving together
produce one session.

### Agents and reload

An agent's `tools` are Python functions and cannot be serialized. Everything else
about an `Agent` is stored, so a session reloaded in a fresh process needs its
live agent from somewhere:

```python
app = agento.Agento(llm=..., store=store, agents=[support_agent])   # by name
session = await app.sessions.get(session_id)                        # found automatically

# or supply it explicitly
session = await app.sessions.get(session_id, agent=support_agent)
```

An agent whose tools are all MCP servers and skills reconstructs from storage
alone. Newly stored agents record whether live tools are required. Older records
without that marker must be loaded with an explicit agent when they used Python
tools. One marked as requiring Python tools raises `ConfigurationError` at `create_turn` with a
message saying exactly this.

---

## Turns

```python
turn = await session.create_turn("hello")                          # continue the tip
turn = await session.create_turn("start over", previous_turn_id="none")
turn = await session.create_turn("rewritten", previous_turn_id=earlier.id)   # branch
```

Branching is how you implement "edit and resend": nothing is deleted, the
session's tip moves onto the new branch, and `list_events()` follows the ancestor
chain so sibling branches stay invisible.

```
turn-1 ──▶ turn-2 ──▶ turn-3          original
   └─────▶ turn-2b ─▶ turn-3b         branch — this is what the session shows now
```

### Reading

```python
turns = await session.list_turns(limit=50)
feed = await session.list_events(limit=100)        # across turns, active branch
events = await turn.list_events(limit=100)         # one turn, oldest first

while feed.next_cursor:
    feed = await session.list_events(limit=100, cursor=feed.next_cursor)
```

`list_events()` works *during* a run as well as after it, which is what lets a
second client follow along without holding the stream.

---

## Running in the background

The HTTP response often needs to return before the agent finishes:

```python
turn = await session.create_turn(user_input)
task = asyncio.create_task(turn.drain())      # runs and persists; you don't consume
return {"turn_id": turn.id}
```

The client then polls `list_events(cursor=...)` or subscribes through your own
transport. Everything is persisted either way — `stream()` and `drain()` are the
same execution, differing only in whether you consume the events.

---

## Cancellation

```python
turn.cancel()                                  # same process, holds the handle
await session.cancel_active_turn()             # another process, records the outcome
```

Cancellation is cooperative and checked between steps, so an in-flight model call
or tool completes and the turn is left resumable. The next turn continues from
the cancelled one's context rather than starting blank.

A new turn rejects a running tip with `PreviousTurnRunningError`. If its worker
has died, reconcile any in-flight actions and explicitly call
`cancel_active_turn()` before continuing. This records cancellation; it cannot
stop a remote worker. Stale handles refresh the session tip and an atomic
expected-tip check prevents concurrent requests from silently forking.

---

## The invariants

Any store must uphold three things, and the shipped ones are tested against them:

**1. Turn creation is atomic with the session tip.** Inserting a turn and
advancing `last_turn_id` is one unit. Two concurrent turns must not leave a tip
pointing at a turn that does not exist.

**2. The first terminal state wins.** Once a turn is `done`, `cancelled` or
`error`, a later write is rejected with `TurnNotRunningError` carrying the state
that won — which is how a cancellation is not overwritten by a completion that
arrived late.

**3. Events order by id.** ULIDs, lexicographically. Never by timestamp.

---

## Writing your own store

Twelve methods, in
[`agento/session/store/base.py`](../src/agento/session/store/base.py). The
contract test suite runs against any implementation:

```python
from tests.test_store import run_contract

async def test_my_store():
    await run_contract(MyStore())
```

That covers the invariants above plus pagination, branch-aware feeds, external-id
uniqueness and cascade delete.

### Notes for a real deployment

**Snapshots are rewritten on every context change.** Simple and correct, and fine
at the scale an embedded agent runs at. If you have very long turns and heavy
concurrency, a store that appends messages instead of rewriting the JSON blob
will do less I/O — the protocol leaves you free to implement `update_turn` that
way.

**The event log is append-only and never updated**, so it partitions and archives
cleanly. Turn state and snapshots are hot; events are cold after the turn ends.

**Index what you query.** The shipped SQL store indexes `sessions.updated_at`,
`turns.session_id`, `turns.created_at`, `events.turn_id` and `events.session_id`.
The event primary key already provides ordering.

---

## The SQL store

```python
store = agento.SQLSessionStore("postgresql+asyncpg://user:pass@host/db")
await store.create_tables()          # or manage DDL yourself
print(store.schema_sql("postgresql"))  # the CREATE TABLE statements
```

Three tables — `agento_sessions`, `agento_turns`, `agento_events` — with a
configurable prefix if you share a schema. Written against SQLAlchemy Core rather
than the ORM, so there is no identity map between agento and the tables and the
SQL is easy to read.

`create_tables()` is fine to call at startup every time; for anything with a real
deployment process, take `schema_sql()` once and manage it with your own
migrations.

## Atomic checkpoints and custom-store migration

`update_turn(..., events=[...])` must commit its snapshot, state, metrics, custom
fields, and events in one transaction or change nothing. Terminal states reject
all subsequent turn mutations, including repeats of the same status.
`create_turn(..., expected_tip=(last_turn_id,))` compares the observed session tip
inside the insert transaction. `(None,)` checks for an empty session;
`expected_tip=None` omits that comparison for direct store callers.

Custom stores must implement these optional parameters; existing application
calls are unchanged. See [API migration details](api.md) and
[operations](operations.md). These changes require no new SQL columns.
