# agento documentation

Start with a runnable agent, then add the pieces your application needs.
All installation commands assume a checkout of this repository. Package-index
publication is a separate release step.

| I want to… | Read |
| --- | --- |
| Run my first agent | [Getting started](getting-started.md) |
| Add functions, approvals, or MCP servers | [Tools](tools.md) |
| Save a conversation and continue it later | [Persistence](persistence.md) |
| Handle the event stream in my application | [Events](events.md) |
| Use compaction, artifacts, skills, or child agents | [Capabilities](capabilities.md) |
| Understand token reduction and reproduce the measurements | [Token efficiency](token-efficiency.md) |
| Find the public classes and their responsibilities | [API guide](api.md) |
| Understand how the runtime fits together | [Architecture](architecture.md) |
| Handle cancellation, outages, and deployment | [Operations](operations.md) |
| Check what has actually been verified | [Validation](validation.md) |
| Contribute or prepare a release | [Contributing](../CONTRIBUTING.md) |
| Download, verify, or publish a GitHub prerelease | [Releases](releases.md) |

The [examples directory](../examples) contains complete scripts. Set
`AGENTO_OFFLINE=1` to force their scripted mode even when provider credentials
are present. The first example works with the Pydantic-only installation;
SQLite examples require the `sqlite` extra.

The [original design plan](PLAN.md) is historical background. Current API and
runtime behavior are described by these guides and the regression tests.
