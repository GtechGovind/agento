# Releases

agento uses GitHub prereleases while its public APIs and operational guarantees are being validated. A prerelease is a versioned preview, not a claim of deployment or live-provider acceptance. See the [validation report](validation.md) for the evidence and remaining boundaries.

## Publish a GitHub prerelease

1. Merge the intended version, changelog, and documentation into protected `main`. The version in `pyproject.toml` must be an unprefixed `X.Y.Z` value.
2. Wait for the **CI gate** and **Security gate** from the main-branch push to succeed on that exact commit.
3. Open **Actions → GitHub prerelease → Run workflow**. Choose `main`, enter the package version (initially `0.1.0`), and paste the full 40-character SHA at the current tip of `main`.

The same dispatch is available through the GitHub CLI:

```sh
gh workflow run release.yml --repo GtechGovind/agento --ref main \
  -f version=0.1.0 -f commit=FULL_40_CHARACTER_MAIN_COMMIT_SHA
```

The workflow checks the repository, branch protection, exact commit, package version, latest successful verification runs, and both gate jobs. It builds the wheel and source distribution, checks their contents, installs the wheel into a clean environment, runs the documented examples, and generates checksums, a CycloneDX SBOM, and GitHub artifact attestations. Before publishing, a separate job repeats the release checks and verifies the downloaded assets against their checksums and provenance.

The workflow creates a new `vX.Y.Z` tag, uploads all assets to a draft, and then publishes it as a prerelease. Existing tags or releases cause it to stop. It does not overwrite published assets, promote a prerelease to a stable release, or publish to PyPI.

## What is included

| Asset | Purpose |
| --- | --- |
| `agento-*.whl` | Installable Python package. |
| `agento-*.tar.gz` | Source distribution, including examples and documentation. |
| `SHA256SUMS` | SHA-256 digests of the wheel, source distribution, and SBOM. |
| `core-runtime.cdx.json` | CycloneDX 1.6 inventory of the clean wheel installation and its resolved core dependencies. |

The SBOM describes that specific core environment, including its installation tooling. It does not inventory every optional provider, database, or MCP extra, and is not a vulnerability-free certification. Optional extras resolve separately when installed. The SBOM retains its generated UUID serial number, which the pinned attestation action requires, and its generation timestamp. The build tools have pinned direct versions; transitive package dependencies and hosted runner images can change, so byte-for-byte reproduction across dates is not guaranteed.

GitHub hosts the signed provenance attestations separately from release assets. They bind asset digests to the workflow and source commit; they do not certify application behavior. The workflow also binds the core-runtime SBOM to the wheel. This follows GitHub's [artifact attestation workflow](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations) and the official [CycloneDX Python environment scanner](https://cyclonedx-bom-tool.readthedocs.io/en/latest/usage.html).

## Verify a download

Download all assets into a new directory, then verify their checksums and provenance:

```sh
gh release download v0.1.0 --repo GtechGovind/agento --dir agento-release
cd agento-release
sha256sum --check SHA256SUMS
gh attestation verify agento-0.1.0-py3-none-any.whl \
  --repo GtechGovind/agento \
  --signer-workflow GtechGovind/agento/.github/workflows/release.yml \
  --source-digest FULL_40_CHARACTER_RELEASE_COMMIT_SHA \
  --deny-self-hosted-runners
```

On macOS, use `shasum -a 256 --check SHA256SUMS`. Compare the expected commit to the published tag and the release workflow run before installation. GitHub's [verification documentation](https://cli.github.com/manual/gh_attestation_verify) describes the available identity restrictions.

## Interrupted releases

A failure before tag creation leaves no release. A failure after tag creation can leave an unused tag or an unpublished draft. The workflow deliberately refuses to reuse either automatically. A maintainer must inspect the failed run, the tag's commit, the draft, and any uploaded assets before deciding whether to remove an unpublished attempt or use a new version. Never move a published release tag or replace its assets; publish a new version for corrections.

Repository administrators control branch rules and release settings. Enable GitHub's immutable-release setting when available so the platform also locks published tags and assets. This pipeline performs no automatic release deletion, rollback, or rule bypass.
