# Security

agento is pre-release. There is no supported stable release or response-time
commitment yet. Review [operations](docs/operations.md) before exposing an agent
to external users or attaching tools with important side effects.

## Reporting

Use the repository's private vulnerability-reporting channel when it is enabled.
If no private channel is configured, ask the repository maintainer for one
without posting exploit details, credentials, or private data publicly. A
maintainer must configure and publish that contact before a public release.

Provide affected versions, the smallest safe reproduction, expected/actual
behavior, and impact. Use disposable files and synthetic credentials.

## Responsibility boundaries

The host owns authentication, authorization, tenancy, sandboxing, network egress,
secret handling, and data retention. Metadata is not an authorization check.
Tools run with the host process's permissions. Approval labels do not replace
service-side permission checks or idempotency.

Local artifacts reject malformed IDs and symlinks but require a host-owned root;
they are not a sandbox against another local process with write access. MCP
credentials stay in runtime configuration; logs, snapshots, and model/tool
outputs may still contain sensitive application data and need host redaction.
