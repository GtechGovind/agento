"""Fail closed before publishing a new GitHub prerelease (Python 3.11+)."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import tomllib

REPOSITORY = 'GtechGovind/agento'
REQUIRED_GATES = {'ci.yml': 'CI gate', 'security.yml': 'Security gate'}


def api(path: str, *, missing_ok: bool = False) -> object:
    request = Request(
        f'https://api.github.com/repos/{REPOSITORY}/{path}',
        headers={
            'Authorization': f'Bearer {os.environ["GH_TOKEN"]}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise SystemExit(f'GitHub API failed with HTTP {exc.code}: {path}') from exc


def validate_identity(version: str, commit: str) -> str:
    if not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', version):
        raise SystemExit('Version must be an unprefixed X.Y.Z version')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise SystemExit('Commit must be a complete lowercase 40-character SHA')
    metadata = tomllib.loads(Path('pyproject.toml').read_text())['project']
    if metadata['version'] != version:
        raise SystemExit('Requested version does not match pyproject.toml')
    checked_out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if checked_out != commit:
        raise SystemExit('Checked-out source does not match the requested commit')
    return f'v{version}'


def preflight(version: str, commit: str) -> str:
    tag = validate_identity(version, commit)
    if (
        os.environ.get('GITHUB_REPOSITORY') != REPOSITORY
        or os.environ.get('GITHUB_REF') != 'refs/heads/main'
        or os.environ.get('GITHUB_SHA') != commit
        or os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
    ):
        raise SystemExit('Dispatch this workflow on main in the canonical repository at the requested SHA')
    branch = api('branches/main')
    if not isinstance(branch, dict) or not branch.get('protected') or branch['commit']['sha'] != commit:
        raise SystemExit('Requested commit must be the current tip of protected main')
    for workflow, gate in REQUIRED_GATES.items():
        query = urlencode({'branch': 'main', 'event': 'push', 'head_sha': commit, 'per_page': 100})
        payload = api(f'actions/workflows/{workflow}/runs?{query}')
        runs = payload['workflow_runs'] if isinstance(payload, dict) else []
        if not runs:
            raise SystemExit(f'No main push verification found for {workflow} at {commit}')
        run = max(runs, key=lambda item: (item['run_number'], item['run_attempt']))
        if run['status'] != 'completed' or run['conclusion'] != 'success' or run['head_sha'] != commit:
            raise SystemExit(f'Latest {workflow} verification is not successful at {commit}')
        payload = api(f'actions/runs/{run["id"]}/jobs?filter=latest&per_page=100')
        jobs = payload['jobs'] if isinstance(payload, dict) else []
        matching = [job for job in jobs if job['name'] == gate]
        if len(matching) != 1 or matching[0]['conclusion'] != 'success':
            raise SystemExit(f'Required job {gate!r} did not succeed')
        print(f'PASS {gate}: {run["html_url"]}')
    if api(f'git/ref/tags/{tag}', missing_ok=True) is not None:
        raise SystemExit(f'Tag {tag} already exists; releases never overwrite tags')
    if api(f'releases/tags/{tag}', missing_ok=True) is not None:
        raise SystemExit(f'Release {tag} already exists; inspect it before retrying')
    print(f'PASS release preflight: {tag} at {commit}')
    return tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    preflight(args.version, args.commit)
