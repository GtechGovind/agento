# Security

agento is pre-release. There is no supported stable release or response-time
commitment yet. Review [operations](docs/operations.md) before exposing an agent
to external users or attaching tools with important side effects.

## Reporting

Report vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/GtechGovind/agento/security/advisories/new).
This sends the report to the repository maintainer without publishing a public
issue. Do not put exploit details, credentials, or private data in public issues
or pull requests. If the private form is unavailable, open an issue asking the
maintainer to restore the reporting channel, without disclosing the vulnerability.

Provide affected versions or commits, the smallest safe reproduction,
expected/actual behavior, and impact. Use disposable files and synthetic
credentials. Include recovery or mitigation ideas if known. The maintainer will
coordinate investigation, fixes, and disclosure through the private report.

## Supported versions

Security fixes currently target the latest `main` revision and the latest
prerelease. Older prereleases have no guaranteed backport support. A prerelease
is evaluation software; the test matrix and its limits are recorded in
[validation](docs/validation.md).

## Responsibility boundaries

The host owns authentication, authorization, tenancy, sandboxing, network egress,
secret handling, and data retention. Metadata is not an authorization check.
Tools run with the host process's permissions. Approval labels do not replace
service-side permission checks or idempotency.

Local artifacts reject malformed IDs and symlinks but require a host-owned root;
they are not a sandbox against another local process with write access. MCP
credentials stay in runtime configuration; logs, snapshots, and model/tool
outputs may still contain sensitive application data and need host redaction.
