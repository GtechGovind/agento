# Tools

Four kinds, one interface:

| Kind | Runs where | Use for |
| --- | --- | --- |
| Python functions | your process | your own business logic |
| Remote MCP | a remote server | GitHub, Linear, Notion, your internal service |
| Client-side | your application | file pickers, browser actions, asking the user |
| Built-ins | agento | `read_artifact`, `read_skill`, `create_sub_agent`, … |

---

## Python functions

```python
from agento import tool

@tool
async def search_orders(customer_id: str, status: str = "any", limit: int = 10) -> str:
    """Search a customer's orders.

    Args:
        customer_id: The customer's id.
        status: Filter by status, or "any".
        limit: Maximum results to return.
    """
    return json.dumps(await db.orders(customer_id, status, limit))
```

**The schema comes from the type hints**, through pydantic. Anything pydantic can
validate works: `str`, `int`, `list[str]`, `Literal["a", "b"]`, an `Enum`, a
nested `BaseModel`.

**The descriptions come from the docstring** — summary for the tool, `Args:` for
each parameter. Write them for the model, not for a colleague who can read the
code. "The customer's id" is weaker than "The customer's id, e.g. `cus_8812`.
Get it from `list_customers` if you do not have it."

**Arguments are validated before your function runs.** A hallucinated argument
becomes a message the model can act on:

```json
{"error": "Invalid arguments", "details": [{"field": "customer_id", "problem": "Field required"}]}
```

Sync functions work and run on a worker thread, so a blocking call cannot stall
streaming or a parallel tool call.

### Return values

| You return | The model sees |
| --- | --- |
| `str` | the string |
| `dict` / `list` | JSON |
| a pydantic model | its JSON |
| `None` | `""` |
| an exception | a JSON error message; the turn continues |

### Marking behaviour

```python
@tool(read_only=True)      # matches @read-only selectors
@tool(destructive=True)    # pauses for approval by default
@tool(requires_approval=True)   # always pause, ignoring policy
@tool(requires_approval=False)  # never pause, ignoring policy
```

Annotating is worth the two seconds: it is what makes `enable_tools=["@read-only"]`
and the default approval policy work without naming individual tools.

---

## `ToolContext`

A tool usually needs to know *who* it is acting for, and you cannot ask the model
— it would invent a customer id. Annotate a parameter and agento injects it,
leaving it out of the schema:

```python
@tool
async def list_my_orders(status: str, ctx: agento.ToolContext) -> str:
    """List the current customer's orders.

    Args:
        status: Filter by status.
    """
    customer = ctx.metadata["user_id"]      # from the session, not the model
    return json.dumps(await db.orders(customer, status))
```

The model sees a one-argument tool. `ctx.metadata` is the session's metadata,
which is where your identifiers belong:

```python
session = await app.sessions.create(agent=agent, metadata={"user_id": "u_42"})
```

`ToolContext` also carries `session_id`, `turn_id`, `thread_id` (so you can tell
delegated work apart), `tool_call_id`, `agent_name`, `artifacts` and `approval`.

Anywhere deeper in a call stack, `agento.current_tool_context()` reads the same
value — it is a `ContextVar`, so parallel tool calls each see their own.

---

## Policy

Selectors name groups of tools without listing them:

| Selector | Matches |
| --- | --- |
| `@all` | every tool |
| `@read-only` | `read_only is True` |
| `@write` | `read_only is False` and not destructive |
| `@destructive` | `destructive is True` |
| `"tool_name"` | that tool |

Four lists, on an `MCPServerRef` or a `PolicyToolSet`:

```python
agento.MCPServerRef(
    name="github",
    enable_tools=["@all"],                      # what may be used
    disable_tools=["delete_repository"],        # subtracted; disable always wins
    preload_tools=["search_issues"],            # in the prompt while the rest defer
    require_approval_for_tools=["@write", "@destructive"],   # the default
    preload=False,                              # the default
)
```

Two things to know:

**Unannotated tools are exempt from tag matching** (except `@all`). A server that
ships no annotations gets no automatic approval gate — for those, name the tools
explicitly.

**An empty list is honoured.** `require_approval_for_tools=[]` disables approval
for that source. That is why the default is expressed as absence rather than as
an empty list.

Policy is **enforced**, not just applied to the listing: a disabled tool is
refused at call time even if the model names it through a deferred wrapper or
hallucinates it.

---

## Remote MCP

```python
app = agento.Agento(
    llm=agento.LiteLLMProvider(),
    mcp={
        "github": agento.MCPServerConfig(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {token}"},
            description="Issues, pull requests, repository files and CI status.",
        ),
    },
)

agent = agento.Agent(
    model="openai/gpt-4o",
    mcp_servers=[agento.MCPServerRef(name="github", enable_tools=["@read-only"])],
)
```

Needs `pip install "agento[mcp]"`. Streamable-HTTP and SSE, probed in that order
unless you set `transport`.

**Write the `description` carefully.** With deferred loading it is the only thing
the agent knows about the server before deciding whether to look inside.

### Expiring credentials and OAuth

`headers` may be an async callable, re-invoked on every operation rather than
once at connect — so a token that lapses mid-session surfaces as an
authorization requirement rather than an opaque 401:

```python
async def github_headers():
    token = await tokens.get("github")
    if token is None:
        return agento.core.tools.base.AuthRequiredOutcome(
            servers=[agento.core.events.McpServerAuth(
                id="github", name="github", auth_url="https://example.com/oauth/github",
            )]
        )
    return {"Authorization": f"Bearer {token}"}

agento.MCPServerConfig(url=..., headers=github_headers)
```

The turn then ends with an `McpAuthRequired` event carrying the URL.

---

## Client-side tools

Tools agento cannot run — the run pauses and your application supplies the
result:

```python
@tool
async def pick_file(prompt: str) -> str:
    """Ask the user to choose a file from their computer.

    Args:
        prompt: What to show in the picker.
    """
    raise NotImplementedError      # never called; the host answers this

agent = agento.Agent(
    model="openai/gpt-4o",
    tools=[agento.ClientSideToolSet("ui", [pick_file])],
)
```

The loop emits `ClientToolRequired`; you answer with `ToolReply`. See
[events](events.md).

---

## Naming

Provider tool names must match `[a-zA-Z0-9_-]{1,64}`, and two servers can both
expose `search`. agento sanitizes illegal characters, truncates to 64, and
de-duplicates with a numeric suffix — first claim wins, and agento's own
built-ins are registered first so they keep their natural names.

The owning set's name is prefixed onto each description (`mcp server: github`),
because with several connectors attached the model otherwise cannot tell two
similarly-named tools apart — and picking the wrong `search` looks like the agent
being stupid rather than the prompt being ambiguous.

Ordering is deterministic (built-ins first, then alphabetical), which keeps the
prompt prefix stable and lets providers serve most of a long conversation from
their prompt cache.

---

## Writing a `ToolSet`

For tools from somewhere agento does not know about — a plugin system, a
database of definitions, another protocol:

```python
class MyToolSet:
    name = "mine"
    id = "mine"
    description = "Tools from my registry."
    preload = True
    has_preloaded_tools = True

    def allowed_tool_names(self): return None

    async def list_tools(self):
        return agento.core.tools.base.ToolListing(tools=[
            agento.ToolSchema(
                name="do_thing",
                description="Does the thing.",
                input_schema={"type": "object", "properties": {"x": {"type": "string"}},
                              "required": ["x"]},
                annotations=agento.core.tools.base.ToolAnnotations(read_only=True),
            )
        ])

    async def call_tool(self, name, arguments, approval=None):
        return agento.text_result(await registry.run(name, arguments))

    async def tool_info(self, name, arguments=None, resolve_underlying=False):
        return agento.core.messages.InternalToolInfo(
            kind="local", source_id=self.id, source_name=self.name,
            original_tool_name=name,
        )
```

Wrap it in a `PolicyToolSet` to get selectors and approval:

```python
agent = agento.Agent(
    model="openai/gpt-4o",
    tools=[agento.PolicyToolSet(MyToolSet(), agento.ToolSelectors(), preload=True)],
)
```

`call_tool` may also return `ApprovalRequiredOutcome`, `ClientSideRequiredOutcome`,
`CreateSubAgentOutcome` or `AuthRequiredOutcome` — that union is how the built-in
capabilities are implemented, and it is open to you.

## Timeout, authentication, and reconnect behavior

MCP requests share a connection worker. Requests that time out before dispatch
are skipped. A timeout after dispatch has an unknown outcome: reconcile before
retrying side effects. Changed headers cause a new connection and a refreshed
catalogue. Reconnect initializes a fresh server session; persisted session IDs
are metadata, not a promise of remote-session resumption.

Local tools with side effects should use explicit annotations and service-level
idempotency. Unannotated tools are not automatically classified as writes.
`idempotent=True` describes a tool; it does not implement deduplication. See
[operations](operations.md) for host responsibilities and recovery examples.
