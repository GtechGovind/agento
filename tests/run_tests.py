#!/usr/bin/env python3
"""Run the original fixture-free test subset without pytest.

pytest is nicer, and ``pytest`` runs these same files unchanged. This exists so
the suite is runnable in a bare checkout — before anything is installed, and in
an environment where installing is not possible.

::

    python tests/run_tests.py                 # everything
    python tests/run_tests.py test_loop       # one module
    python tests/run_tests.py -k approval     # matching tests

It discovers ``test_*.py`` modules, runs every ``async def test_*`` (and plain
``def test_*``) in them, and reports failures with tracebacks.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TESTS_DIR))

from helpers import Skip  # noqa: E402  (path set up above)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _discover(selectors: list[str]) -> list[Path]:
    modules = sorted(TESTS_DIR.glob("test_*.py"))
    if not selectors:
        return modules
    return [path for path in modules if any(sel in path.stem for sel in selectors)]


def _run_one(func: object) -> None:
    if inspect.iscoroutinefunction(func):
        asyncio.run(func())  # type: ignore[arg-type]
    else:
        func()  # type: ignore[operator]


def main(argv: list[str]) -> int:
    keyword: str | None = None
    if "-k" in argv:
        index = argv.index("-k")
        keyword = argv[index + 1] if index + 1 < len(argv) else None
        argv = argv[:index] + argv[index + 2 :]

    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    started = time.perf_counter()

    for path in _discover(argv):
        module_name = path.stem
        if "import pytest" in path.read_text():
            skipped.append(f"{module_name} (requires pytest fixtures; run python -m pytest)")
            continue
        try:
            module = __import__(module_name)
        except Exception:
            failed.append((f"{module_name} (import)", traceback.format_exc()))
            continue

        tests = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        for name, func in tests:
            label = f"{module_name}::{name}"
            if keyword and keyword not in label:
                continue
            try:
                _run_one(func)
            except Skip as skip:
                skipped.append(f"{label} ({skip})")
                print(f"{YELLOW}s{RESET}", end="", flush=True)
            except Exception:
                failed.append((label, traceback.format_exc()))
                print(f"{RED}F{RESET}", end="", flush=True)
            else:
                passed.append(label)
                print(f"{GREEN}.{RESET}", end="", flush=True)

    elapsed = time.perf_counter() - started
    print("\n")

    for label, tb in failed:
        print(f"{RED}FAILED{RESET} {label}")
        print(f"{DIM}{tb}{RESET}")

    for label in skipped:
        print(f"{YELLOW}SKIPPED{RESET} {label}")

    summary = f"{len(passed)} passed"
    if failed:
        summary += f", {RED}{len(failed)} failed{RESET}"
    if skipped:
        summary += f", {YELLOW}{len(skipped)} skipped{RESET}"
    print(f"{summary} in {elapsed:.2f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
