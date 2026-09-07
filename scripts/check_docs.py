"""Check local Markdown links without a network dependency."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / name for name in ['README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md']]
FILES += sorted((ROOT / 'docs').glob('*.md'))


def main() -> None:
    errors = []
    checked = 0
    for path in FILES:
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            target = link.split(' "', 1)[0].strip('<>')
            parts = urlsplit(target)
            if parts.scheme or not parts.path:
                continue
            checked += 1
            resolved = (path.parent / unquote(parts.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing local target {target}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'{checked} local documentation links valid across {len(FILES)} files')


if __name__ == '__main__':
    main()
