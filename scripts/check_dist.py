"""Inspect built distribution contents before sharing or publishing them."""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN = {'.idea', '.slingshot', 'graphify-out', '.venv', '__pycache__', '.git'}


def main(directory: str) -> None:
    artifacts = [*Path(directory).glob('*.whl'), *Path(directory).glob('*.tar.gz')]
    if not any(p.suffix == '.whl' for p in artifacts) or not any(p.name.endswith('.tar.gz') for p in artifacts):
        raise SystemExit('Build both wheel and sdist before checking')
    for artifact in artifacts:
        if artifact.suffix == '.whl':
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(artifact) as archive:
                names = archive.getnames()
        for name in names:
            parts = Path(name).parts
            if FORBIDDEN.intersection(parts) or any(part == '.env' or part.endswith(('.db', '.iml', '.pyc')) for part in parts):
                raise SystemExit(f'Forbidden package content: {name}')
        for required in ['LICENSE', 'NOTICE', 'py.typed']:
            if not any(Path(name).name == required for name in names):
                raise SystemExit(f'{artifact.name}: missing {required}')
        print(f'PASS {artifact.name}: {len(names)} entries; license, notice, and typing marker present')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'dist')
