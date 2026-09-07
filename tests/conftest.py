"""pytest configuration.

Puts ``src/`` and the tests directory on the path so ``import agento`` and
``import helpers`` work in a checkout without installing anything. The suite also
runs without pytest — see ``python tests/run_tests.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
