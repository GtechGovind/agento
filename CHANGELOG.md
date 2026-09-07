# Changelog

## Unreleased

### Changed

- Reimplemented thread execution, orchestration, context queries, instruction
  composition, compaction, deferred discovery, and the OpenUI instruction pack
  while preserving public interfaces and persisted data shapes.
- Replaced the historical design plan with current engineering contracts and
  release acceptance work; refreshed README context measurements and diagrams.
- Keep SQLAlchemy, PyYAML, and OpenAI tracing instrumentation in optional extras;
  the core continues to depend only on Pydantic.

### Fixed

- Preserve MCP connection-initialization metadata from deferred discovery.
- Safely quote instruction content containing a CDATA closing delimiter.
- Publish capability append events after their checkpoint and reject writes to
  another capability's state key.
- Commit child joins and retirement atomically; retire already-answered children
  from older snapshots without repeating model or tool work.

### Added

- 52 behavioral regression cases for stream ownership, cancellation, approvals,
  recovery, child coordination, prompt composition, discovery, and OpenUI.

## 0.1.0 — 2026-09-07 (preview)

### Fixed

- Checkpoint complete model/tool events together with the current snapshot before
  publication; prevent duplicate compaction events in SQL.
- Finalize explicitly closed streams without yielding during generator teardown;
  preserve capability state and child completion through recovery.
- Reject terminal turn mutations and concurrent stale-tip creation; count terminal
  metrics once. Refresh stale handles and require explicit active-turn cancellation.
- Preserve upload events and require live Python tools when reloading agent definitions.
- Validate local artifact IDs and symlinks; serialize inline images/files to chat
  provider content-part shapes.
- Skip expired queued MCP requests, distinguish uncertain in-flight timeouts,
  support tested MCP SDK 1.x/2.x transports, and reconnect on credential changes.
- Classify concurrent SQL external-ID conflicts so get-or-create returns the winner.

### Added

- Regression tests across memory/SQLite, optional PostgreSQL, local MCP SDK
  integration, offline provider transport tests, coverage and CI checks.
- Brief README, detailed usage/operations guides, contribution and security
  policies, license files, typed-package marker, and distribution allowlists.
- Three architecture/context visuals and six reproducible context-efficiency
  measurements with their scope, overhead, and limitations.
- GitHub CI for Python 3.10–3.14, CodeQL and dependency audits, coding-agent
  guidance, Dependabot updates, and a verified prerelease pipeline with checksums,
  a core-runtime SBOM, and artifact attestations.

### Compatibility notes

- Application `run`/`stream` call shapes are retained. Session/application stream
  convenience calls are awaited; `turn.stream()` is iterated directly.
- Custom stores must support `create_turn(..., expected_tip=...)` and atomic
  `update_turn(..., events=...)`. See the [store migration guide](docs/api.md).
- A new turn no longer silently cancels a running turn. Repeated terminal updates
  now raise `TurnNotRunningError` instead of mutating completed state.
- The LiteLLM extra requires Python 3.11+ because recent upstream releases fail
  to import on Python 3.10. Core, OpenAI, and MCP retain Python 3.10 support.
- MCP session IDs are retained as metadata; reconnect initializes a new remote
  session. MCP support is bounded to `>=1.26,<3` and tested by version in CI.
- Existing stored agents from before the live-tool marker need an explicit
  `agent=` binding on reload when they used Python tools.
