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

Core, OpenAI, and MCP support Python 3.10+. LiteLLM requires Python 3.11+ and is
omitted by its dependency marker on 3.10. Use Python 3.11+ to work on LiteLLM.

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

Fork the repository or create a focused branch and open a pull request against
`main`. Changes to protected `main` go through pull requests and the required
status checks. Resolve review conversations before merging. Default review
ownership is `@GtechGovind`; a maintainer reviews compatibility, scope, and
release impact. Dependency updates follow the same process and are not
automatically merged.

The repository currently has one maintainer, so protection does not demand a
second approving reviewer on the owner's own pull requests. Required CI/security
checks, resolved conversations, and the pull-request path still apply; only
users with repository merge permission can merge. Revisit the approval count
when additional maintainers join.

Describe the concrete problem, resulting behavior, and checks run. Include
remaining limitations and reproducible failures. New features should explain
which application needs them and why they belong in the runtime.

## Coding agents and dependency maintenance

[AGENTS.md](AGENTS.md) provides shared guidance for coding agents, and
[Copilot instructions](.github/copilot-instructions.md) describe repository
contracts to prioritize during implementation and review. Give an agent a
bounded problem, acceptance criteria, and relevant failure evidence. Review its
diff and validation results through the same pull-request process.

Two optional custom profiles are available in supported Copilot environments:

- [runtime-maintainer](.github/agents/runtime-maintainer.md): focused runtime and
  adapter fixes with regression tests; local read/search/edit/execute tools.
- [docs-reviewer](.github/agents/docs-reviewer.md): checks documentation and token
  claims against source and evidence; read/search tools only.

These files provide context; they do not activate a hosted coding agent or
automatic AI reviews. A maintainer must enable an available agent integration
for their account/repository and explicitly assign a task or request its review.
No model API key or paid AI service is required for the normal CI checks.

Dependabot checks `uv` dependencies and pinned GitHub Actions weekly. Minor and
patch updates are grouped; major updates remain separate for compatibility
review. Keep `pyproject.toml` and `uv.lock` consistent and rerun the compatibility
matrix when updating adapters. See GitHub's
[supported ecosystems](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories)
and [custom agent setup](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/create-custom-agents).

## GitHub prereleases

1. Run the checks above, inspect package contents, and review source provenance,
   dependency licenses, metadata, and notices. Confirm that private vulnerability
   reporting works.
2. Update the version, `CHANGELOG.md`, and validation documentation through a
   pull request to `main`. Record the release's tested scope and unverified
   live-provider/deployment boundaries.
3. Wait for **CI gate** and **Security gate** to pass on the intended main-branch
   push. In **Actions → GitHub prerelease → Run workflow**, select `main` and
   enter `version` as `X.Y.Z` and `commit` as the current full 40-character SHA.
4. Review the resulting workflow and assets. The pipeline verifies the exact
   source, builds and checks the packages, records checksums/SBOM/attestations,
   and publishes a new GitHub prerelease. It does not publish to PyPI.

See [the release guide](docs/releases.md) for exact requirements, download
verification, and recovery from an interrupted attempt. Never move a published
tag or replace release assets. Publish a new version for a correction.
