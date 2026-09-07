# Changelog

## Unreleased

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
  policies, license/attribution files, typed-package marker, and distribution allowlists.

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
