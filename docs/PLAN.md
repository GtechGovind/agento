# Engineering plan

This plan describes the contracts agento maintains and the evidence needed for
future releases. The [usage guides](README.md) document the current API; the
[validation record](validation.md) records completed checks and remaining work.

## Product scope

agento is a Python library embedded in a host application. It coordinates model
requests, tool execution, human responses, child agents, context management, and
session persistence. The host supplies its interface, authentication, access
control, deployment supervision, and business rules.

The core targets Python 3.10 and depends on Pydantic. Provider, MCP, database,
tokenizer, and tracing integrations remain optional. LiteLLM requires Python
3.11 or later. Installing an extra must not silently enable an integration.

## Contracts to preserve

| Boundary | Required behavior | Where to inspect it |
| --- | --- | --- |
| Application calls | Agent definitions, sessions, turns, and streaming remain typed and explicit | [Public API](api.md) |
| Model requests | Provider adapters normalize streaming and usage while keeping provider-specific replay fields | [Architecture](architecture.md) |
| Tool execution | Argument validation and source policy apply to both preloaded and deferred calls | [Tools and approvals](tools.md) |
| Persistence | Snapshots and durable events commit atomically; stale session tips and terminal mutations are rejected | [Persistence](persistence.md) |
| Interruption | Unconfirmed external effects remain unknown until reconciled; recovery must not silently retry them | [Operations](operations.md) |
| Capabilities | Hooks return declared context, event, or state changes; state survives checkpoints | [Capabilities](capabilities.md) |
| Child agents | Contexts stay separate; completion reaches the parent before the child is retired | [Events](events.md) |
| Package contents | Distributions contain source, license information, and typing metadata; local IDE data stays outside them | [Release process](releases.md) |

## Runtime organization

A thread records conversation and capability state in its journal. Each execution
builds a queue of operations for preparation, model requests, tool work, and
continuation. Stream ownership belongs to that execution, including cleanup when
a consumer closes the iterator or cancels its task.

The orchestrator schedules runnable threads with bounded concurrency. Each active
source has its own task and result slot. The consumer advances a source after
handling its output, preserving backpressure and task-local tracing state. Joined
child results are checkpointed before the completion becomes visible.

Context helpers project the journal into outstanding calls, latest approval
decisions, and estimated usage. Instruction sections retain live child handles
and render in order. Deferred discovery publishes a small gateway while the
underlying connector remains responsible for execution policy.

Compaction sends a structured journal to a summarization model and accepts only
a nonblank handover. The record separates user text, calls, results, and approval
decisions. A summary can lose information, so workload-specific evaluation is
required before relying on it for long-running tasks.

## Evidence for changes

1. Add a behavioral regression for a changed execution or persistence boundary.
   Use scripted models and disposable resources for repeatable local tests.
2. Run lint, strict typing, the supported Python test matrix, and the 85% combined
   coverage gate. Exercise PostgreSQL durability separately from in-memory tests.
3. Execute the README example and complete examples, check documentation links,
   and inspect the built wheel and source distribution.
4. Reproduce context measurements when prompts, schemas, or context processing
   change. Publish request scope, helper overhead, and small-workload cases beside
   any reduction percentage.
5. Merge through the protected pull-request workflow only after CI and security
   checks pass for the candidate commit.

## Work before a stable release

- Exercise selected live providers with the host's real structured, multimodal,
  reasoning, timeout, and rate-limit requirements.
- Test remote MCP authentication, reconnection, and side-effect reconciliation
  against the actual connectors used by the application.
- Validate storage contention, process interruption, backup and restore,
  retention, tenant isolation, and resource cleanup in the target deployment.
- Evaluate compaction quality and complete-task token use with representative
  tasks. Synthetic request reductions alone do not establish billing savings.
- Validate generated OpenUI output with the renderer the host chooses. The SDK
  supplies instructions and events; rendering belongs to the host.
- Complete source and dependency license review, packaging checks, and the
  documented release checklist before a stable release.

These are acceptance tasks, not claims of completed live-service or production
validation. Track confirmed results in the validation record as work progresses.
