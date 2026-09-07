# Getting started

## 1. Install from a checkout

Use Python 3.10 or newer and an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate`. Choose extras as needed:

| Extra | Purpose |
| --- | --- |
| `openai` | Direct OpenAI SDK / compatible chat-completion endpoint |
| `litellm` | Provider routing through LiteLLM; Python 3.11+ |
| `mcp` | Remote MCP tools; tested SDK range declared in `pyproject.toml` |
| `sqlite` | SQLAlchemy and the SQLite driver |
| `postgres` | SQLAlchemy and the PostgreSQL driver |
| `otel` | OpenTelemetry tracing API |
| `extras` | Optional token counting, ULIDs, YAML, and supporting libraries |
| `all` | Both model adapters, MCP, SQLite, tracing, and niceties; add `postgres` for its driver |
| `dev` | Test, coverage, type-checking, lint, and local test-server dependencies |

Combine them with `python -m pip install -e '.[openai,sqlite]'`.
The core imports without those extras. A missing adapter dependency produces an
installation hint when that adapter is used.

## 2. Run without an account

Save this as `hello.py` and run `python hello.py`:

```python
import asyncio
import agento

async def main():
    llm = agento.ScriptedLLM([agento.say("Hello from agento.")])
    app = agento.Agento(llm=llm)
    agent = agento.Agent(name="assistant", model="scripted/demo")
    print(await app.run(agent, "Hello"))

asyncio.run(main())
```

`ScriptedLLM` follows your supplied responses. It is useful for learning and
repeatable tests; it is not an AI model.

## 3. Connect a real model

Install the `openai` extra. Set credentials in your shell or secret manager,
and choose a model your endpoint supports:

```bash
export AGENTO_MODEL="your-model-id"
# Set OPENAI_API_KEY through your normal secret-management workflow.
```

Inside your async function:

```python
import os

app = agento.Agento(llm=agento.OpenAIProvider())
agent = agento.Agent(name="assistant", model=os.environ["AGENTO_MODEL"])
print(await app.run(agent, "Explain this project's purpose in one sentence."))
```

For an OpenAI-compatible endpoint, supply `base_url` and the endpoint's required
credentials to `OpenAIProvider`. For multiple providers, install `litellm`, use
`LiteLLMProvider`, and set `Agent.model` to the provider's LiteLLM model string.
Capabilities such as image/PDF inputs and structured output depend on the chosen
model and endpoint; check them against that provider before rollout.

## 4. Add a typed tool

A complete, offline example:

```python
import asyncio
import agento

@agento.tool(read_only=True)
async def shipment_status(order_id: str) -> str:
    """Look up a shipment by order ID."""
    return {"ORD-7": "out for delivery"}.get(order_id, "unknown order")

async def main():
    app = agento.Agento(llm=agento.ScriptedLLM([
        agento.say(tool_calls=[("shipment_status", {"order_id": "ORD-7"})]),
        agento.say("Your order is out for delivery."),
    ]))
    agent = agento.Agent(name="shipping", model="scripted/demo", tools=[shipment_status])
    print(await app.run(agent, "Where is ORD-7?"))

asyncio.run(main())
```

Type hints become the argument schema. Docstrings describe the tool to the model.
Arguments are validated before invocation. Synchronous tools run in a worker
thread. Mark writes explicitly and read [approval policy](tools.md) before giving
an agent tools with side effects.

## 5. Stream and inspect a turn

The following snippets run inside an async function with `app` and `agent` defined:

```python
from contextlib import aclosing

session = await app.sessions.create(agent=agent, metadata={"user_id": "u42"})
turn = await session.create_turn("Where is ORD-7?")

async with aclosing(turn.stream()) as stream:
    async for event in stream:
        if isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)
        elif isinstance(event, agento.TurnDone):
            print("\n", event.state.status)
```

Creating a turn prepares and stores it. **Iterating executes it**, exactly once.
Use `turn.drain()` when you want persistence without consuming events.
A plain `break` does not guarantee immediate async-generator cleanup; `aclosing`
closes the stream reliably on early exit.

The convenience methods retain their awaitable API:

```python
stream = await session.stream("Hello")
async with aclosing(stream):
    async for event in stream:
        print(event.type)
```

Use `await app.stream(agent, input)` the same way. `turn.stream()` itself needs
no `await`. These distinctions are also listed in the [API guide](api.md).

## 6. Keep the conversation

Install `sqlite`, then create and close the store explicitly:

```python
store = agento.SQLSessionStore("sqlite+aiosqlite:///./agento.db")
await store.create_tables()
try:
    app = agento.Agento(llm=agento.OpenAIProvider(), store=store, agents=[agent])
    session, created = await app.sessions.get_or_create_by_external_id(
        "ticket:123", agent=agent,
    )
    print(await session.run("What do you remember?"))
finally:
    await store.dispose()
```

To resume after a restart, construct a fresh runtime with the same store and
register the live agent again. Python functions are not serialized into the
snapshot. See [persistence](persistence.md) for branches, active-turn conflicts,
and the custom-store contract.

## Next steps

Run `AGENTO_OFFLINE=1 python examples/03_approval.py` for a complete approval
cycle. Explore [capabilities](capabilities.md) for context management and
[operations](operations.md) before integrating with a long-lived service.
