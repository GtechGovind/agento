# Token efficiency: mechanisms, measurements, and tradeoffs

agento manages what enters an agent's working context and what remains there.
This can reduce repeated input tokens as tool catalogs, results, and conversations
grow. The amount depends on the workload and what the agent needs to read next.

## Four mechanisms that work together

| Mechanism | What happens in the runtime | When it helps | Cost or limitation |
| --- | --- | --- | --- |
| **Deferred tool schemas** | Advertise a source, then expose `list_tools`, `get_tool_info`, and `call_tool` instead of every tool schema | Many available tools, few needed for a particular task | Discovery adds model/tool interactions and context; preload frequently used tools |
| **On-demand skills** | Advertise skill names, descriptions, and resource names; load procedures and files when requested | A library of procedures that rarely all apply at once | Reading a skill still consumes context; loaded content remains until replaced or compacted |
| **Result offloading** | Replace an oversized successful result with an artifact reference and preview; offer selective reads and searches | Large query results, logs, or connector responses | Requires an artifact store to retain the full result; retrieval adds context and may require more calls |
| **History compaction** | Ask the model for a structured summary, then replace working history when the threshold is reached | Conversations approaching the input budget | Summarization costs tokens and can lose detail; a smaller prompt does not establish equivalent answer quality |

The mechanisms are independently configurable. The default large-result thresholds
are 6,000 estimated tokens for one result and 10,000 for a result batch, with a
400-character preview from each end. Compaction defaults to 80% of the reported
model context window, constrained by the output reservation, or 50,000 tokens
when the window is unknown. See [configuration](capabilities.md).

The runtime estimates thresholds using `o200k_base` when tiktoken is available,
and a character heuristic otherwise. Neither method establishes exact token
usage for every provider. Provider-reported usage is kept separately.

## What we measured

On 2026-09-07, the [measurement script](../scripts/measure_context.py) ran paired
executions through agento using deterministic synthetic data and `ScriptedLLM`.
It captured the actual `LLMRequest.messages` and `LLMRequest.tools`, serialized
them as compact JSON, and counted that text with tiktoken's `o200k_base` encoding.

**These are local context measurements.** They do not measure provider billing,
model quality, latency, or total tokens for a completed real-world task. The
scripted responses exercise real runtime transformations; no model inference or
external service is involved. Additional tool reads, discovery calls, output
tokens, summarization, and provider prompt caching can change the overall cost.

Both sides use the same input and tool definitions. Unrelated capabilities are
disabled equally. The enabled side includes its helper tool schemas, instructions,
artifact references, and previews. We do not compare a full request against only
a shortened result fragment.

### Large results: context on the next model request

The tool returns generated support-ticket records containing identifiers, subject,
status, priority, and update time. Each pair runs the same two scripted model
calls: request the tool, then finish. The only changed capability is offloading.
The artifact store is present on both sides.

| Records returned | Payload bytes | Offloading disabled | Offloading enabled | Next-request context reduction |
| --- | ---: | ---: | ---: | ---: |
| 20 | 3,569 | 1,494 | 1,831 | **−22.56%**: helpers add 337 tokens; the small result stays inline |
| 200 | 35,707 | 12,834 | 970 | **92.44%** |
| 2,000 | 357,093 | 126,234 | 967 | **99.23%** |

Counts include messages and all exposed tool schemas. For the two offloaded
cases, the script reads the stored artifact and asserts byte-for-byte equality
with the original result. The smaller request contains a preview, not all the
information in the original result. Selective retrieval is the next step when the
answer needs data outside that preview; those later calls are not measured here.

Across both recorded input requests, including helper overhead before the tool
runs, the respective reductions are **−40.10%, 88.53%, and 98.82%**. These totals
still exclude future retrieval, generated output, and any real model's decisions.

### Tool catalogs: context on the first model request

Each synthetic catalog tool has a name, description, and a schema for `query`,
`status`, and `limit`. Both runs expose the same source through `PolicyToolSet`;
the pair changes only `preload=True` versus `preload=False`.

| Available tools | All schemas preloaded | Deferred discovery | First-request context reduction |
| --- | ---: | ---: | ---: |
| 1 | 280 | 492 | **−75.71%**: discovery adds 212 tokens |
| 10 | 1,459 | 492 | **66.28%** |
| 100 | 13,249 | 492 | **96.29%** |

This is the request before discovery or business-tool execution. The deferred
request exposes three discovery/invocation helpers and the source description.
When a tool is needed, its name/schema and subsequent results enter the
conversation. The constant 492 in this fixture describes the initial request
for one source, not unlimited catalogs or a fixed size throughout execution.

**The small cases matter.** Deferring one tool can be more expensive than preloading
it. Enabling artifact helpers for small results can add context without removing
anything. Choose features for the workload and measure a complete task before
making a cost claim.

## Reproduce the measurements

From a checkout, in an isolated Python environment:

```bash
python -m pip install -e '.[extras]'
python scripts/measure_context.py --output /tmp/agento-context-results.json
```

The [recorded JSON](data/context-measurements.json) includes environment versions,
payload hashes, per-request counts, and measurement scope. This run used Python
3.14.7, Pydantic 2.13.5, and tiktoken 0.14.0. Generated artifact IDs can change a
few token counts between runs. The script does not require provider credentials
and does not invoke a model or business service.

We have not measured a token-saving percentage for compaction or skills here.
A fixed scripted summary would not validate a real model's summary quality or
total cost, so the README does not turn the compaction regression into a savings
benchmark. No competitor or live-task comparison is implied by these results.

## Enable the pieces you need

Inside your application setup, with your `provider` already configured:

```python
import agento

app = agento.Agento(
    llm=provider,
    artifacts=agento.LocalArtifactStore("./artifacts"),
)
agent = agento.Agent(
    name="research",
    model="your-model-id",
    config=agento.RuntimeConfig(
        large_tool_response=agento.LargeToolResponseConfig(
            individual_token_threshold=6_000,
            total_token_threshold=10_000,
        ),
        compaction=agento.CompactionConfig(threshold_tokens=50_000),
    ),
)
```

An MCP reference defaults to deferred discovery. Set `preload=True` for a source
used on most calls, or use `preload_tools` for selected tools. Attach skills by
name through a configured `SkillSource`. See [tools](tools.md),
[capabilities](capabilities.md), and the [context example](../examples/04_context_management.py).

## Inspect the implementation

| Feature | Source |
| --- | --- |
| Result thresholds, previews, selective reads, and searches | [large_tool_response.py](../src/agento/core/capabilities/builtins/large_tool_response.py) |
| Deferred discovery and invocation | [deferred_tools.py](../src/agento/core/capabilities/builtins/deferred_tools.py) |
| Skill metadata and demand loading | [skills.py](../src/agento/core/capabilities/builtins/skills.py) |
| Summary generation and context replacement | [compaction.py](../src/agento/core/capabilities/builtins/compaction.py) |
| Threshold token estimates | [tokens.py](../src/agento/core/tokens.py) |
| Defaults and runtime configuration | [agent.py](../src/agento/session/agent.py) |
| Provider usage, reasoning/cache tokens, and optional costs | [metrics.py](../src/agento/core/runtime/metrics.py) |

Child agents can keep detailed work in separate contexts and return focused
results. They also make model calls of their own: delegation is a way to organize
work and route models, not an automatic total-token reduction. Cache-token and
cost fields are reported when the provider supplies them; agento does not itself
guarantee prompt caching or billing savings.
