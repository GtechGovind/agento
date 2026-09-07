> Historical design plan. For current behavior and verified scope, see the
> [documentation index](README.md) and [validation record](validation.md).

# agento — build plan

> An embeddable **agent harness for Python**. Everything TrueForge's runtime does,
> exposed as an SDK your application calls directly — no HTTP server, no chat UI,
> no multi-tenancy, no sandbox.

---

## 1. What agento is, and what it is not

TrueForge is three things stacked: a **harness** (the agent execution loop), a
**server** (HTTP + SSE + auth + multi-tenant stores + catalogs), and a **UI**
(React chat app + embeddable SDK). Roughly 120k lines of TypeScript, of which the
harness itself — `trueforge-core` — is ~16k.

agento is a Python re-implementation of **the harness only**, redesigned as a
library rather than the engine behind a server.

| | TrueForge | agento |
| --- | --- | --- |
| Consumed via | HTTP API + SSE, or its TS SDK over HTTP | direct Python calls, `async for` over events |
| Agent definition | rows in a DB, referenced by id | a Python object you construct (or a Pydantic model you store yourself) |
| Tenancy | `tenant_id` on every store call | none — sessions carry `metadata` + `external_id` |
| Models | catalog rows → Vercel AI SDK | `LLM` Protocol + adapters |
| Tools | remote MCP servers, configured by name | remote MCP **and** plain Python functions |
| Skills | git repos sparse-cloned into a sandbox | local directories, read on demand |
| Code execution | Daytona / local sandbox, Code Mode | **out of scope for v1** |
| Big tool results | dumped into a sandbox file | dumped into a pluggable `ArtifactStore` |
| Persistence | SQLite + Postgres, hand-written SQL per backend | `SessionStore` Protocol + in-memory + SQL connector |
| UI | React chat app | none (Generative UI **prompt pack** is included, rendering is yours) |

### Locked decisions

- **Python, async-first.** `async for event in turn.stream()`. Target **≥ 3.10**
  (your Cowork VM runs 3.10, your macOS venv runs 3.14 — 3.10 covers both).
- **No UI code.** Generative UI ships as an instruction pack + one tool; your
  app renders the `openui` blocks the model emits.
- **No sandbox.** Not a stub, not dead code — the seams that a sandbox would
  plug into (`ToolSet`, `ArtifactStore`, `SkillSource`) are public Protocols, so
  adding one later is additive, not a rewrite.
- **Interfaces first for LLMs and storage.** Adapters are optional extras.
- **Skills** = filesystem directories with `SKILL.md`, progressive disclosure.
- **Large tool responses** = offloaded to an `ArtifactStore`, replaced in context
  with an id + preview, readable back through a built-in tool.

---

## 2. Environment constraint you should know about

PyPI is unreachable from both this session's cloud sandbox and the Cowork VM on
your Mac (`403` from the egress policy; `tunnel error: unsuccessful` from `uv`).
`pydantic 2.13` happens to be pre-installed in the sandbox; nothing else can be
installed or runtime-tested here.

This does not block the build, but it does dictate the dependency policy:

> **Anything agento's core imports must be either stdlib or pydantic.**
> Every third-party library lives behind a Protocol, in an adapter module that is
> imported lazily and only when you opt into it.

The practical effect: the ~85% of agento I can fully test here (loop, tools,
capabilities, sessions, in-memory store) is tested here. The ~15% I cannot (the
LiteLLM shim, the MCP shim, the SQLAlchemy store) are small, isolated wrappers —
each one file, each verifiable on your machine with a single `uv pip install`.
I will mark those files explicitly and ship a smoke script that exercises them.

---

## 3. Dependency policy — reuse over rebuild

You asked to use open-source libraries wherever they exist. Here is every place
that decision applies, and the two places I recommend **not** taking a library.

### Use the library

| Concern | Library | Why, and where it plugs in |
| --- | --- | --- |
| **Model providers** | **LiteLLM** | One `acompletion(model=..., stream=True)` covers OpenAI, Anthropic, Gemini, Bedrock, Vertex, Azure, Ollama, vLLM, OpenRouter, TrueFoundry — 100+ providers, normalized to OpenAI chunk shape, with tool calls, reasoning content, cache tokens and cost already parsed. TrueForge hand-writes this per provider against the Vercel AI SDK; LiteLLM is that layer, for Python, maintained. → `agento.llm.LiteLLMClient` |
| Model providers (alt) | `openai` SDK | For teams that only ever talk to one OpenAI-compatible endpoint and don't want LiteLLM's weight. → `agento.llm.OpenAICompatibleClient` |
| **Remote MCP** | **`mcp`** (official Python SDK) | `streamablehttp_client` + `sse_client`, session lifecycle, `list_tools` pagination, `call_tool`. Exactly what TrueForge uses the TS MCP SDK for. → `agento.tools.RemoteMCP` |
| **SQL persistence** | **SQLAlchemy 2.0 (async)** | One store implementation gives SQLite (`aiosqlite`), Postgres (`asyncpg`) and MySQL. TrueForge maintains two parallel hand-written SQL backends plus two migration sets; this collapses that to one. → `agento.session.store.SQLSessionStore` |
| Token counting | `tiktoken` | Real counts instead of `len/4` when installed. Heuristic fallback otherwise. |
| SKILL.md frontmatter | `PyYAML` | Parses the standard Anthropic `SKILL.md` header. ~15-line stdlib fallback so skills work without it. |
| Event ids | `python-ulid` | Monotonic ULIDs are the durable event ordering key (TrueForge's choice, and it's right). Stdlib fallback included since it's 30 lines and I need ordering to work in tests. |
| Retries | `tenacity` | Backoff around provider and MCP calls, when installed. |
| JSON Schema for tool args | **pydantic** (already a dep) | `TypeAdapter(...).json_schema()` on a function's signature. No extra dependency, and it is *better* than a bespoke schema generator. |
| Tracing | **OpenTelemetry API** | `agento.tracing.OTelTracer` implements the `Tracer` Protocol against `opentelemetry-api`. No-op by default. |

### Don't use a library (deliberately)

- **The agent loop itself.** LangGraph / LlamaIndex / Agents SDK / CrewAI each
  bring their own loop, their own state model and their own opinions. You asked
  for *TrueForge's* behaviour — its state machine, its approval semantics, its
  compaction and offloading policy, its persist-before-yield durability. Building
  that loop on top of another framework's loop means fighting it forever. This is
  the ~2,500 lines that are genuinely agento's own, and they are a direct port of
  code we just read end to end.
- **Async fan-out.** stdlib `asyncio` merges the sub-agent generators fine.
  `anyio` would be a dependency for ~40 lines.

---

## 4. Architecture

Four layers. Each one only knows about the layer below it.

```
┌──────────────────────────────────────────────────────────────────┐
│  session/      Agento · Sessions · SessionHandle · TurnHandle    │  durability,
│                ResourceResolver · SessionStore                   │  turn lifecycle
├──────────────────────────────────────────────────────────────────┤
│  core/runtime/ AgentThread · Orchestrator · DeferredTools        │  the loop
├──────────────────────────────────────────────────────────────────┤
│  core/capabilities/  compaction · subagents · large results ·    │  behaviour that
│                      skills · approvals · questions · openui     │  hooks the loop
├──────────────────────────────────────────────────────────────────┤
│  core/llm/     LLM Protocol + adapters                           │  the outside
│  core/tools/   ToolSet Protocol + local / MCP / client-side      │  world
│  skills/ artifacts/ store/   Protocols + default implementations │
└──────────────────────────────────────────────────────────────────┘
```

### Module map

```
src/agento/
  __init__.py               public surface: Agent, Agento, tool, MCPServer, ...
  errors.py                 every exception, each with a stable .code
  _ids.py                   monotonic ULID (python-ulid when present)
  tracing.py                Tracer Protocol, NoopTracer, OTelTracer

  core/
    messages.py             LLM message / tool-call / usage models
    events.py               the streamed event union — the SDK's real contract
    instructions.py         InstructionBuilder (XML-sectioned system prompt)
    tokens.py               token estimation (tiktoken when present)

    llm/
      base.py               LLM Protocol, LLMRequest, StreamChunk, LLMResponse
      litellm_client.py     ← LiteLLM adapter (default)
      openai_client.py      ← openai SDK adapter
      echo.py               scripted fake LLM — makes the whole loop testable

    tools/
      base.py               ToolSet / ToolSource Protocols, tool-result envelopes
      local.py              @tool decorator, LocalToolSet, schema from signature
      client_side.py        tools your app executes (agent pauses for the result)
      selectors.py          @all / @read-only / @write / @destructive
      policy.py             per-agent enable/disable/preload/approval policy
      remote_mcp.py         ← official mcp SDK adapter
      registry.py           name sanitizing, collision suffixes, LLM schema conv.
      execute.py            parallel tool execution, result envelopes

    capabilities/
      base.py               Capability Protocol + the four processor hooks
      builtins/
        current_datetime.py
        ask_user_question.py     structured clarification (client-side tool)
        subagents.py             create_sub_agent + delegation guidance
        compaction.py            summarize-and-replace when context grows
        large_tool_response.py   offload to ArtifactStore + read_artifact
        skills.py                <skill> prompt sections + read_skill
        generative_ui.py         full openui pack, preload or deferred

    runtime/
      agent_thread.py       the loop: state machine, streaming, approvals
      orchestrator.py       root + sub-agent threads, parallelism, cancellation
      deferred_tools.py     list_tools / get_tool_info / call_tool
      tool_call_repair.py   closes dangling tool calls so replay never 400s
      metrics.py            token / iteration / tool-call / subagent counters

  skills/
    base.py                 Skill model, SkillSource Protocol
    filesystem.py           FileSkillSource — a directory of SKILL.md dirs

  artifacts/
    base.py                 ArtifactStore Protocol, Artifact model
    memory.py               in-process (tests, short-lived work)
    local.py                a directory on disk (default)

  session/
    agent.py                Agent + RuntimeConfig (fully serializable)
    agento.py               Agento — the façade you construct once
    sessions.py             create / get / get_or_create_by_external_id
    session_handle.py       createTurn: build threads, validate, persist
    turn_handle.py          stream(): persist-before-yield, terminal state
    resolver.py             per-run wiring: model → client, name → MCP, skills
    store/
      base.py               SessionStore Protocol + all input models
      memory.py             reference implementation, fully tested
      sql.py                ← SQLAlchemy async: SQLite / Postgres / MySQL
```

---

## 5. The public API

Everything below is what a user of the SDK actually types. This is the surface I
will optimize for, and the docs will be written against it.

### The 6-line version

```python
import asyncio, agento

app = agento.Agento(llm=agento.LiteLLMClient())     # models via LiteLLM
agent = agento.Agent(model="openai/gpt-4o", instructions="You are concise.")

async def main():
    print(await app.run(agent, "What is an agent harness?"))

asyncio.run(main())
```

### Tools are plain functions

```python
from agento import tool

@tool
async def search_orders(customer_id: str, limit: int = 10) -> str:
    """Look up recent orders for a customer.

    Args:
        customer_id: The customer's id.
        limit: How many orders to return.
    """
    return json.dumps(await db.orders(customer_id, limit))

agent = agento.Agent(model="openai/gpt-4o", tools=[search_orders])
```

The JSON Schema comes from the type hints; the description and per-argument docs
come from the docstring. Mark side effects so approval policy can see them:

```python
@tool(destructive=True)
async def refund(order_id: str) -> str: ...
```

### Streaming, and the events you get

```python
session = await app.sessions.create(agent=agent)
turn = await session.create_turn(input="refund order 42 and tell me why it failed")

async for event in turn.stream():
    match event:
        case agento.ModelMessageDelta(content=c) if c:
            print(c, end="", flush=True)
        case agento.ToolApprovalRequired(tool_calls=calls):
            ...   # your app decides
        case agento.TurnDone(state=state):
            print(state.status)
```

### Human in the loop

The turn ends with `required_actions`; you answer by creating the next turn.

```python
turn = await session.create_turn(input=[
    agento.ToolApproval(thread_id="main", tool_call_id=id, decision="allow"),
])
```

Same shape for `ask_user_question` and any client-side tool: the loop pauses,
emits a `ToolResponseRequired`, and resumes when you send a `ToolResponse`.

### Resources are wired once, on the façade

```python
app = agento.Agento(
    llm=agento.LiteLLMClient(),                          # or a dict / callable
    mcp={"github": agento.MCPServerConfig(url=..., headers=...)},
    skills=agento.FileSkillSource("./skills"),
    artifacts=agento.LocalArtifactStore("./.agento/artifacts"),
    store=agento.SQLSessionStore("sqlite+aiosqlite:///./agento.db"),
)
```

An `Agent` then refers to MCP servers and skills **by name**, which keeps agent
definitions serializable — you can store them in your own database as JSON and
hand them back to `Agento` later.

---

## 6. Feature parity map

Every harness feature, and how it lands in agento.

| TrueForge feature | agento | Notes |
| --- | --- | --- |
| Agent loop + state machine | `runtime/agent_thread.py` | Direct port, including the `llm-call` / `tool-response` / `user-input` transitions and their legality table |
| Streaming deltas | `ModelMessageDelta` events | Passed through, never persisted (same as TrueForge) |
| Tool approval | `capabilities` + policy | `@write` / `@destructive` / `@all` / literal names |
| Ask-user-question | `builtins/ask_user_question.py` | Client-side tool; loop pauses |
| Client-side tools | `tools/client_side.py` | Your app supplies the result |
| Sub-agents | `builtins/subagents.py` + orchestrator | Parallel, capped, isolated context, only the summary returns |
| Deferred tool loading | `runtime/deferred_tools.py` | `list_tools` / `get_tool_info` / `call_tool` |
| Large tool responses | `builtins/large_tool_response.py` | → `ArtifactStore` instead of sandbox file |
| Context compaction | `builtins/compaction.py` | Same trigger maths (80% of context length, 50k fallback) and summary prompt |
| Skills | `skills/` + `builtins/skills.py` | Filesystem, progressive disclosure |
| MCP (remote) | `tools/remote_mcp.py` | streamable-http + SSE, session resume, pagination guard |
| MCP OAuth | header resolver hook | The `headers` callable can raise "auth required" and surface it as an event; you own the OAuth dance |
| Generative UI | `builtins/generative_ui.py` | Full openui pack, both preload and deferred modes |
| Sessions / turns / events | `session/` | persist-before-yield, forkable turns, resumable event log |
| Metrics | `runtime/metrics.py` | tokens, cost, iterations, tool calls, subagents, compactions |
| Tracing | `tracing.py` | Protocol + no-op + OTel adapter |
| Sandbox, Code Mode | — | Out of scope; seams left open |
| HTTP server, auth, catalogs, UI | — | Out of scope by design |

---

## 7. Build phases

Each phase leaves the repo importable and tested. You can start using it after
phase 4.

| # | Phase | Deliverable |
| --- | --- | --- |
| 1 | **Foundations** | `pyproject`, errors, ids, tracing, `InstructionBuilder`, message + event models, token estimation |
| 2 | **LLM layer** | `LLM` Protocol, normalized chunk/usage types, `ScriptedLLM` (test double), LiteLLM adapter, openai adapter |
| 3 | **Tool layer** | `@tool`, `LocalToolSet`, client-side tools, selectors + policy, registry/schema conversion, parallel execution, `RemoteMCP` |
| 4 | **Runtime** | `AgentThread`, `Orchestrator`, deferred tools, tool-call repair, metrics — **the loop runs end to end here** |
| 5 | **Capabilities** | datetime, ask-question, subagents, compaction, artifacts + large results, skills, generative UI |
| 6 | **Session layer** | `Agent`, `Agento`, `Sessions`, `SessionHandle`, `TurnHandle`, resolver, `SessionStore` + memory + SQL |
| 7 | **Docs, examples, tests** | README, architecture guide, one doc per subsystem, runnable examples, pytest suite, smoke script for the un-testable adapters |

---

## 8. Testing

- **`ScriptedLLM`** — a fake LLM you hand a list of turns to (text, tool calls,
  usage, finish reasons). Every loop behaviour becomes a deterministic unit test
  with no network: multi-step tool use, approval pause/resume, sub-agent fan-out,
  compaction trigger, offload threshold, iteration limit, cancellation mid-stream.
- **In-memory store** carries the full `SessionStore` contract test; the SQL store
  is expected to pass the identical suite once you can install SQLAlchemy.
- **`scripts/smoke.py`** — the only script that needs real credentials and real
  packages. Runs one live turn against a real model, one real MCP server, and one
  SQLite-backed session, so you can verify the three untestable adapters in one go.

---

## 9. Things I'd like your call on (non-blocking — I'll use the first option)

1. **Package/import name** — `agento` throughout. Confirm you want the PyPI-style
   name and `import agento`, not something like `from agento_sdk import ...`.
2. **License header** — TrueForge is MIT. agento is a re-implementation, not a
   copy, but the design lineage is real. I'll add a short attribution note in the
   README ("architecture derived from TrueForge, MIT") unless you'd rather not.
3. **Cost tracking** — LiteLLM can compute per-call USD cost. I'll surface it in
   metrics when the adapter provides it; free to ignore.
4. **Sync façade** — everything is async. I'll add a thin `agento.sync` wrapper
   (`app.run_sync(...)`) so scripts and Django views can call it without an event
   loop dance. Say if you'd rather not have it.

---

## 10. What happens next

On your go-ahead I build phases 1 → 7 in order, writing files into
`/Volumes/workspace/agento` as each phase completes, so you can read and run the
code while the rest is still being written.
