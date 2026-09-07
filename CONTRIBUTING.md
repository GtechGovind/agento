# Contributing to agento

Start with the [architecture](docs/architecture.md), choose a focused change, and
include a failing regression test when fixing behavior. Documentation and
runnable examples are first-class contributions.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all,dev,postgres]'
```

With uv, `uv sync --extra all --extra dev --extra postgres` uses the checked-in
lockfile. Runtime dependencies remain optional; the complete development
installation is used for strict type checking and adapter tests.

## Required local checks

```bash
python -m pytest --cov=agento --cov-report=term-missing --cov-report=xml
python -m ruff check src tests scripts examples
python -m mypy src/agento
python scripts/check_docs.py
python scripts/check_examples.py
uv build
python scripts/check_dist.py dist
```

Coverage includes branches and has an 85% combined threshold. Do not exclude
production modules or weaken assertions merely to meet it. Add scenarios around
observable contracts, failure paths, and external boundaries. Adapter tests use
offline responses and a temporary MCP server on loopback; no provider key is needed.

The compatibility CI checks Python 3.10–3.14, minimal core imports, MCP 1.x/2.x,
and PostgreSQL. CI configuration is not evidence that a hosted run has passed;
see the [current local verification record](docs/validation.md).

To run PostgreSQL contract/regression tests against a **disposable** database,
install `postgres` and set `AGENTO_TEST_POSTGRES_URL`. Tests create isolated table
prefixes and drop those test tables afterward. Do not point this at a production
database. The normal test command does not require PostgreSQL.

## Change boundaries

- Preserve public application call shapes unless a migration is discussed.
- Add a regression that fails before a behavior fix, including memory and SQL
  when storage is involved. Test the instant an event is published, not only the
  final drained state.
- Keep imports optional for the Pydantic-only core. Update the store contract and
  migration notes if custom implementations must change.
- Keep README usage brief. Put detail in `docs/` and add a navigation link.
- Preserve attribution and dependency licenses. Never commit credentials, IDE
  assistant state, databases, local artifacts, or generated analysis output.

## Pull requests

Describe the concrete problem, resulting behavior, and checks run. Include
remaining limitations and reproducible failures. New features should explain
which application needs them and why they belong in the runtime.

## Release checklist

1. Run the checks above on a clean checkout and inspect wheel/sdist contents.
2. Review source provenance, dependency licenses, package metadata, and notices.
3. Confirm repository ownership, intended package-index name, and private
   vulnerability reporting contact. No publication destination is assumed here.
4. Record supported Python, adapter, and database versions and run live provider
   acceptance, recovery, and deployment checks for the release scope.
5. Update `CHANGELOG.md` and `docs/validation.md`, review the version, then build
   signed/provenance-attested artifacts using the release infrastructure chosen
   by the maintainers.
6. Publish only after maintainer review. The supplied CI builds and validates;
   it has no package publishing credentials or automatic publishing step.
