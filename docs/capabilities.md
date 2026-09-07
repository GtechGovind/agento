# Capabilities

The loop is deliberately small: call the model, run the tools, repeat.
*Everything else* agento does is a capability hooked onto it — compaction,
offloading, sub-agents, skills, approvals, generative UI. None of them uses a
private API, so anything they can do, you can do.

---

## The hooks

```
send()  ─── pre_send ──┐
                       │   repair dangling state before new input lands
┌──────────────────────┘
│
├─ pre_llm ──────────── just before each model call        (compaction)
├─ prepare_request ──── last edit of the outgoing messages (ephemeral)
│
│  ... model call, tool execution ...
│
├─ process_tool_results  results in hand, before context   (offloading)
└─ post_tool_call ────── after results are in context
```

Plus two non-hook contributions:

- `tool_sets()` — tools the capability provides
- `build_instructions(builder)` — a section in the system prompt

### Hooks yield, they do not mutate

A hook is an async generator yielding outputs; the loop applies them:

| Output | Effect |
| --- | --- |
| `AppendContext(messages=[...], events=[...])` | add messages |
| `ReplaceContext(messages=[...], usage=...)` | replace the whole context |
| `EmitEvent(event=...)` | emit without touching context |
| `SetState(key=..., value=...)` | persist durable state |

The loop applies the in-memory transition, then the turn layer checkpoints the
new snapshot and associated events before publishing the durable public event.

`process_tool_results` is the exception — it may mutate `results` in place, which
is how offloading replaces a huge payload with a preview. It must not add or
remove results: every tool call needs exactly one, or the next provider request
is invalid.

### `pre_llm` vs `prepare_request`

Both run before a model call, and the difference is durability:

- `pre_llm` **changes the conversation**. What it appends is stored and will be
  there next turn.
- `prepare_request` **changes one request**. The edit reaches the model and then
  disappears.

Use `prepare_request` for anything that should not accumulate: a just-in-time
reminder, a cache-control marker, redaction. Doing that in `pre_llm` means twenty
copies of the reminder by turn twenty.

---

## A worked example

```python
class BudgetGuard(agento.Capability):
    """Warns the agent as it approaches a token budget."""

    name = "budget_guard"
    state_key = "example.budget"       # declaring this makes state durable

    def __init__(self, limit_tokens: int) -> None:
        self._limit = limit_tokens
        self._spent = 0

    def load_state(self, value):
        """Called at construction with last turn's value."""
        self._spent = int(value or 0)

    def build_instructions(self, builder):
        builder.add_section(
            "budget",
            f"This conversation has a budget of {self._limit:,} tokens. Be direct.",
        )

    async def pre_llm(self, context):
        self._spent = context.usage.total()
        yield agento.SetState(key=self.state_key, value=self._spent)

    def prepare_request(self, messages):
        if self._spent < self._limit * 0.8:
            return None
        return [*messages, {"role": "user", "content": "[budget nearly spent — wrap up]"}]


app = agento.Agento(llm=..., capabilities=[BudgetGuard(limit_tokens=50_000)])
```

`capabilities=` on `Agento` applies to every agent it runs. For one agent, pass
them through a `ResourceResolver` subclass — see [architecture](architecture.md).

---

## Durable state

Set `state_key`, yield `SetState`, implement `load_state`. The value is persisted
per thread and handed back on the next turn.

Rules the loop enforces:

- **The key must be declared.** Yielding `SetState` for an undeclared key raises
  `CapabilityStateError`, which catches typos and stops one capability writing
  over another's state.
- **Keys are unique per thread.** Two capabilities declaring the same key is an
  error at construction.
- **Values are JSON.** `None` clears; there is no "undefined".
- **Orphaned keys are dropped.** Turning a capability off does not leave its
  state to be silently resurrected later.

---

## The built-ins

Each is worth reading as an example of one hook done well.

### `CurrentDateTime`

One tool. Models have no clock, and a model left to itself answers "what is
today's date?" from training data — confidently wrong, and silently wrong for
anything relative like "overdue" or "last quarter".

### `AskUserQuestion`

A client-side tool. The loop pauses and emits `ClientToolRequired`; your app
renders the question and answers with `ToolReply`. The tool description carries
the guidance that keeps it from being annoying: ask only when the answer changes
what happens next, make options mutually exclusive, never add "Other".

Not given to sub-agents — a delegated thread has no user to ask.

### `SubAgents`

Adds `create_sub_agent`. See [architecture](architecture.md) for how a child
joins back. Two constraints worth designing around: the child cannot see the
conversation (its `input` is its whole world), and it cannot ask the user
anything.

`model_choices` lets an agent route cheap work to a cheap model:

```python
agento.SubAgentConfig(
    enabled=True,
    model_choices={"fast": "Quick lookups.", "thorough": "Deep analysis."},
    models={"fast": "openai/gpt-4o-mini", "thorough": "anthropic/claude-sonnet-4-5"},
)
```

### `DeferredTools`

Adds `list_tools`, `get_tool_info` and `call_tool` for sources with
`preload=False`. Three extra round trips in exchange for a prompt that stays
small no matter how many connectors are attached.

Approval survives the wrapper: `call_tool` reports the *underlying* tool when
asked what it is about to run, so a destructive tool reached this way pauses
exactly as it would if preloaded.

### `ContextCompaction`

`pre_llm`. At 80% of the model's context window — or 50,000 tokens when that is
unknown — asks the model for a structured summary and replaces the working
history with it.

The full event log is untouched, but the *agent* now works from the summary. That
is a real trade, and it is why the prompt asks for verbatim user messages and
specific identifiers rather than a tidy paragraph.

### `LargeToolResponse`

`process_tool_results`. Two thresholds, because they catch different failures:
`individual_token_threshold` (6,000) for one huge result, and
`total_token_threshold` (10,000) for many medium ones that are individually fine
and collectively ruinous.

Adds `read_artifact` (ranged reads) and `search_artifact` (regex with line
numbers). Without an artifact store configured it still works, truncating to a
preview — lossier, but the context window survives.

### `Skills`

`build_instructions` plus `read_skill` / `read_skill_file`. Only each skill's
name and description reach the prompt; the body loads on demand. See
[skills](#skills-format) below.

### `GenerativeUI`

Teaches the model the `openui` language so it can emit charts, tables and forms.
**agento renders nothing** — the block arrives as ordinary assistant text and
your application renders it. The capability module defines the grammar,
component signatures, and built-in functions for implementing a compatible renderer.

Off by default. `preload=True` puts the full spec in every prompt (about 3,100
tokens with `o200k_base`; provider tokenization varies). The default defers it
behind a tool call.

---

## Skills format

A skill is a folder with a `SKILL.md`:

```
skills/
  refund-policy/
    SKILL.md
    limits.md          # referenced, read on demand
```

```markdown
---
name: refund-policy
description: How to decide and process a customer refund. Use whenever a
  customer asks for a refund, a chargeback, or money back.
---

## Procedure

1. Check the order date. Refunds are automatic within 30 days.
...
```

The `description` is the most important line in the file: it is the only thing
the model sees before deciding whether to open the skill. Write it as *when to
use this*, not as a title.

Front matter is optional — without it, the directory name becomes the skill name
and the first paragraph becomes the description.

```python
app = agento.Agento(llm=..., skills=agento.FileSkillSource("./skills"))
agent = agento.Agent(model=..., skills=["refund-policy"])
```

For skills from git, S3 or a database, implement `SkillSource` — three methods.

---

## Turning things off

```python
agento.Agent(
    model="openai/gpt-4o",
    config=agento.RuntimeConfig(
        iteration_limit=50,
        current_datetime=True,
        ask_user_questions=True,
        sub_agents=agento.SubAgentConfig(enabled=False),
        compaction=agento.CompactionConfig(enabled=True, threshold_tokens=80_000),
        large_tool_response=agento.LargeToolResponseConfig(
            enabled=True, individual_token_threshold=4_000
        ),
        generative_ui=agento.GenerativeUIConfig(enabled=True, preload=False),
    ),
)
```

Every capability adds tools and prompt text, and both cost tokens on every call.
For a narrow agent — one tool, short answers — turning off sub-agents, questions
and the clock measurably shrinks the prompt.
