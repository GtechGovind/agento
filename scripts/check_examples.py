"""Run every example and the README quickstart with forced offline providers."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = {**os.environ, 'AGENTO_OFFLINE': '1', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
           'PYTHONPATH': str(ROOT / 'src')}
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY']:
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix='agento-examples-') as directory:
        readme = Path(directory) / 'readme_quickstart.py'
        readme.write_text(re.search(r'```python\n(.*?)```', (ROOT / 'README.md').read_text(), re.S)[1])
        for script in [readme, *sorted((ROOT / 'examples').glob('[0-9]*.py'))]:
            result = subprocess.run([sys.executable, str(script)], cwd=directory, env=env,
                                    text=True, capture_output=True, timeout=60)
            if result.returncode:
                raise SystemExit(f'{script.name} failed:\n{result.stdout}\n{result.stderr}')
            print(f'PASS {script.name}')


if __name__ == '__main__':
    main()
