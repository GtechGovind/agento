# Working on agento

agento is an embeddable Python agent runtime. Keep changes focused, preserve the
host application's control, and substantiate behavior and performance claims.
Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/architecture.md](docs/architecture.md).

## Repository map

- `src/agento/core/`: messages, events, model/tool contracts, runtime, capabilities.
- `src/agento/session/`: public facade, turn lifecycle, resource resolution, stores.
- `src/agento/artifacts/`: host-owned artifact storage and access.
- `tests/`: contract, recovery, adapter, and dependency-boundary scenarios.
- `examples/`: runnable usage; `docs/`: detailed guides and evidence.

Use a local `graphify-out/` graph, when available, to locate relationships; verify
the relevant source and tests before editing. Generated graphs are navigation
aids, may be stale, and are excluded from version control and packages.

## Design rules

- Keep the core importable with only Pydantic. Import optional dependencies in
  their adapters, lazily. Core code must not depend on the session facade.
- Support Python 3.10+ in core, OpenAI, and MCP paths. Only the LiteLLM adapter
  requires Python 3.11+. Preserve that distinction in dependencies and examples.
- Preserve public application call shapes. For a necessary contract change,
  update the protocol, all implementations, relevant tests, and migration docs.
  Clarify a material unresolved compatibility decision before implementing it.
- Keep durable snapshots and their public events atomic. Completed state must
  be committed before durable events are published; terminal records are immutable.
  Protect session-tip updates against stale handles and concurrent writers.
- An interrupted external tool call can have an unknown outcome. Reconcile before
  retrying a side effect; never interpret a timeout as proof it did not execute.
  Do not silently cancel an active turn to start another.
- Keep credentials and live Python callables outside serialized agent definitions.
  Authentication, authorization, tenancy, idempotency, and sandboxing belong to
  the host/service boundary. Metadata and approval labels do not provide them.
- Keep provider wire formats in adapters and context strategies in capabilities.
  Avoid broad refactors or new hard dependencies for a local behavior fix.

## Verification

Use an isolated environment. `uv sync --extra all --extra dev --extra postgres`
installs the development dependencies from `uv.lock`. The equivalent pip setup
and full checks are in CONTRIBUTING.md. Common commands:

```bash
python -m pytest --cov=agento --cov-report=term-missing
python -m ruff check src tests scripts examples
python -m mypy src/agento
python scripts/check_docs.py
python scripts/check_examples.py
```

For behavior changes, add a meaningful regression around the observable failure.
Storage changes need memory and SQL coverage; event-ordering tests should inspect
the store at publication time. Test cancellation, restart, and concurrent writes
when those boundaries change. Keep the 85% combined branch/statement coverage
gate; do not hide production modules or weaken assertions to pass it.

Use `AGENTO_OFFLINE=1` for examples and deterministic adapters for tests. Tests
must not require model credentials or paid calls. PostgreSQL tests accept
`AGENTO_TEST_POSTGRES_URL` only for a disposable database. Verify package changes
with `uv build` and `python scripts/check_dist.py dist`.

## Documentation and delivery

Keep the README concise and link detailed instructions in `docs/`. Update docs
and examples when behavior changes. For token-efficiency claims, preserve the
measured scope, baseline, overhead, and limitations; local context counts do not
establish billing savings or equivalent model quality.

Report what changed, checks actually run, and remaining uncertainty. Preserve
unrelated work. Never commit credentials, IDE/assistant state, databases, local
artifacts, or generated audit output. Work through pull requests to protected
`main`; agent suggestions still require the repository's normal checks and review.
