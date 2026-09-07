# Validation and release status

This is a pre-release engineering validation record, not a production acceptance
certificate. Checks use scripted models, disposable storage, offline provider
responses, and a local MCP server. No paid model request or external business tool
was invoked for this repair.

## Scope

The original study inventoried 80 Python files, 62 library modules, and the project
documents. Its 63 passing tests missed several lifecycle and adapter failures.
The repair focuses on those failures, their regression tests, documentation,
package contents, and repeatable development checks. Existing user-owned staged
and untracked work was preserved; the original study began before the repository's
first commit.

## Runtime replacement validation — 2026-09-07

Seven modules were reimplemented around local interface and behavioral contracts:
thread execution, orchestration, context queries, instruction composition,
compaction, deferred tools, and the OpenUI instruction pack. Public call shapes
and persisted data models remain compatible. The engineering plan and prompt
examples were rewritten, and current source and documentation branding was removed.

| Check | Result |
| --- | --- |
| Python 3.14.7 with locked optional integrations | 179 tests passed |
| Python 3.10.20 with MCP 1.26.0 and offline provider adapters | 179 tests passed; the MCP test server emits the settings warning described below |
| Combined statement and branch coverage | 88.18%; configured floor remains 85% (90.62% statements, 78.84% branches) |
| Runtime contract regressions | 52 additional cases across execution, orchestration, instructions, discovery, compaction, and OpenUI |
| Minimum core: Python 3.10.20 / Pydantic 2.7.0 | 66 tests passed; seven pytest modules and one optional SQL test explicitly skipped |
| Ruff and strict mypy | Passed; 62 library modules type-checked |
| README quickstart and six examples | Passed in both full and minimal environments |
| Context measurements | All six synthetic cases rerun; tables and README chart updated from the recorded JSON |

The new cases cover iterator ownership, cancellation, approval publication,
capability-state recovery, child-result replay, source backpressure, discovery
policy, and safe quoted instruction content. Discovery now preserves MCP
initialization metadata when it opens a connection. Capability append events
reach both stored and live streams after checkpointing, and state writes enforce
the emitting capability's own key. Successful and failed child joins are atomic;
legacy answered-child snapshots retire without repeating their work.

These checks establish behavior in the stated environments. They are not a formal
clean-room process or a repository-wide provenance certification. Live-model
summary quality and host-rendered OpenUI output still require acceptance testing.
The immutable v0.1.0 release and historical commits retain their original contents.

## Original preview baseline — 2026-09-07

| Check | Evidence / result |
| --- | --- |
| Python 3.14.7, current adapter dependencies | 127 tests passed |
| Combined statement and branch coverage | 85.65%; configured floor 85% (88.75% statements, 74.08% branches) |
| Ruff configured rules | Passed |
| Strict mypy | Passed across 62 library modules |
| README quickstart and six examples | Passed with forced offline execution |
| MCP | Actual localhost HTTP initialization, tool listing/call, credential reconnect; current SDK 2.1.1 |
| OpenAI SDK 2.54.0 | Stream parsing and outgoing request verified with offline HTTP transport |
| LiteLLM 1.100.0 | Adapter translation tested with deterministic injected completion responses on Python 3.14 |
| Python 3.10.20 + PostgreSQL 17 + MCP 1.26.0 | 142 tests passed; 85.84% combined coverage; temporary PostgreSQL container removed after testing |
| Minimum core: Python 3.10.20 / Pydantic 2.7.0 | 62 original tests passed; SQLite check and four pytest-only modules skipped explicitly; all examples passed |
| Wheel and source distribution | Both built and inspected; license, notice, and typing marker present; IDE/assistant/audit data excluded |
| Clean wheel install | Imported and executed a scripted agent outside the source checkout |
| Documentation | Local Markdown links checked; README quickstart executed |

Python 3.10 emits a Pydantic-settings warning from the MCP 1.26 test server;
Pydantic 2.7 emits two protected-namespace warnings. These runs passed. LiteLLM
requires Python 3.11+; its adapter translation is also tested with injected
responses on Python 3.10, but live LiteLLM construction is explicitly rejected.

The test counts and coverage above describe the stated environment. They do not
mean every branch is covered or every provider/database combination is verified.
The compatibility workflow is in [CI](../.github/workflows/ci.yml). The preview table above
is the original local baseline; current hosted results are available in
[Verify](https://github.com/GtechGovind/agento/actions/workflows/ci.yml) and
[Security](https://github.com/GtechGovind/agento/actions/workflows/security.yml).
The [release pipeline](releases.md) requires both gates to pass on its exact source commit.

## Failure-to-regression map

| Original finding | Repair / verification |
| --- | --- |
| Artifact root traversal | ID and symlink validation; [artifact tests](../tests/test_artifacts_inputs.py) |
| Duplicate SQLite compaction event | One insertion owner and snapshot checkpoint; [durability tests](../tests/test_durability.py) |
| Complete message visible before persistence | Inspect event log and snapshot at each publication boundary; durability tests |
| Capability state snapshot lag | Apply state before checkpoint; durability tests |
| Completed tool lost on stream closure | Checkpoint results before publishing; close-and-resume tests |
| Incorrect “not executed, retry” recovery | Unknown-outcome repair; interrupted approved-action regression |
| MCP queued write runs after timeout | Skip expired queue entries; [adapter tests](../tests/test_adapters.py) |
| MCP SDK transport import failure | Compatible transport selection; SDK loopback integration |
| Terminal writes mutate result/double metrics | Immutable terminal turn, serialized SQL transition; concurrent-write tests |
| Stale handles silently branch | Refresh and compare session tip atomically; concurrent-create tests |
| Async-generator closure fails | No yield during teardown; created/delta/message/result closure tests |
| Convenience stream API docs fail | Preserve awaitable calls and test both entry points |
| Python tools silently disappear on reload | Store live-tool requirement marker and require agent rebinding |
| User-upload artifact event disappears | Carry preparation events into the durable stream |
| Image/file wire parts fail provider schema | Normalize content parts; validate exact OpenAI SDK content-part schemas |
| Child completion/retirement can replay | Persist completion and retire before checkpoint; [boundary tests](../tests/test_boundaries.py) |
| Package includes local workspace state | Explicit sdist allowlist, license/typing marker, [distribution check](../scripts/check_dist.py) |
| Relationship and documentation drift | Public export/dependency tests, local Markdown link checks, Graphify navigation (the local graph predates the runtime replacement) |

## Reproduce

Follow [contributing](../CONTRIBUTING.md) for installation and check commands.
Run `python -m pytest --cov=agento --cov-report=term-missing` to inspect coverage
by module. Add `--cov-report=html` for a local browsable report. The PostgreSQL
fixture is opt-in and uses unique temporary table prefixes.

## Remaining release work

- Confirm live provider behavior for the selected models, especially multimodal
  inputs, structured output, reasoning content, rate limits, and retry policies.
- Exercise real remote MCP authentication/session behavior and any side effects
  through the host application's idempotency/reconciliation workflow.
- Verify workload-specific contention, load, process-crash recovery, backup/restore,
  retention, tenant authorization, and resource cleanup in the target deployment.
- MySQL is outside the locally verified set. The source supports SQLAlchemy
  dialects, but that is not evidence of backend acceptance. Hosted checks are
  tracked separately through the workflow links above.
- Repository security reports use [private advisories](https://github.com/GtechGovind/agento/security/advisories/new).
  Review source provenance and complete the release checklist before a stable
  release. Package-index publication remains separate from GitHub prereleases.

Graphify is a navigation aid. Unresolved extraction edges and undirected edge
coalescing are recorded in the local graph diagnostics; they are not silently
relabelled as valid runtime relationships. The raw extraction is retained.
