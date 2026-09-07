"""Skills read from a directory on disk.

Point it at a folder whose subdirectories each contain a ``SKILL.md``::

    source = FileSkillSource("./skills")
    app = agento.Agento(llm=..., skills=source)
    agent = agento.Agent(model=..., skills=["contract-review"])

Front matter is parsed with PyYAML when it is installed, and with a small
built-in parser otherwise — which handles the flat ``key: value`` and folded
multi-line values that a ``SKILL.md`` header actually uses. Skills therefore work
with no dependencies at all, and get full YAML if you want it.

Missing front matter is not an error: the directory name becomes the skill name
and the first paragraph of the body becomes its description. A skill that is just
a markdown file still works, which is worth more than strictness here.

Skill bodies are cached after first read, keyed by the file's modification time,
so a long-running process picks up edits without a restart.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import Skill

__all__ = ["FileSkillSource", "parse_skill_markdown"]

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_SIMPLE_KEY = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")
_RESOURCE_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py", ".sql", ".toml"}


def _parse_front_matter(raw: str) -> dict[str, Any]:
    """Parse a front-matter block into a dict.

    Uses PyYAML when available. The fallback understands ``key: value`` plus
    continuation lines, which covers the shape ``SKILL.md`` headers actually take.
    """
    try:  # pragma: no cover - optional dependency
        import yaml

        loaded = yaml.safe_load(raw)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    data: dict[str, Any] = {}
    key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _SIMPLE_KEY.match(line)
        if match and not line[:1].isspace():
            key = match.group(1)
            data[key] = match.group(2).strip().strip("\"'")
        elif key is not None:
            # Continuation of a folded value.
            data[key] = f"{data[key]} {line.strip()}".strip()
    return data


def parse_skill_markdown(text: str, *, fallback_name: str) -> tuple[dict[str, Any], str]:
    """Split a ``SKILL.md`` into front matter and body.

    Args:
        text: The file's contents.
        fallback_name: Used when the front matter has no ``name``.

    Returns:
        ``(front_matter, body)``. When there is no front matter, a name and
        description are derived from the directory name and first paragraph.
    """
    match = _FRONT_MATTER.match(text)
    if match:
        front = _parse_front_matter(match.group(1))
        body = match.group(2)
    else:
        front = {}
        body = text

    front.setdefault("name", fallback_name)
    if not front.get("description"):
        first_paragraph = next(
            (block.strip() for block in body.split("\n\n") if block.strip() and not block.startswith("#")),
            "",
        )
        front["description"] = " ".join(first_paragraph.split())[:400]
    return front, body


class FileSkillSource:
    """Skills stored as directories under a root folder.

    Args:
        root: The folder to scan. Each subdirectory containing ``SKILL.md`` is a
            skill. A ``SKILL.md`` directly in ``root`` also counts, so a single
            skill folder can be passed directly.
        max_resource_bytes: Refuse to read a resource file larger than this.
            Stops a stray large file in a skill folder from being pulled into
            context wholesale.
    """

    def __init__(self, root: str | Path, *, max_resource_bytes: int = 512 * 1024) -> None:
        self._root = Path(root).expanduser().resolve()
        self._max_resource_bytes = max_resource_bytes
        self._cache: dict[str, tuple[float, str]] = {}

    # -- discovery ---------------------------------------------------------- #

    def _skill_dirs(self) -> dict[str, Path]:
        """Map skill name to directory, by scanning for ``SKILL.md``."""
        found: dict[str, Path] = {}
        if not self._root.is_dir():
            return found
        if (self._root / "SKILL.md").is_file():
            found[self._root.name] = self._root
            return found
        for entry in sorted(self._root.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                found[entry.name] = entry
        return found

    def _load(self, directory: Path) -> Skill:
        path = directory / "SKILL.md"
        front, body = parse_skill_markdown(path.read_text(encoding="utf-8"), fallback_name=directory.name)

        resources = [
            str(item.relative_to(directory))
            for item in sorted(directory.rglob("*"))
            if item.is_file() and item.name != "SKILL.md" and item.suffix.lower() in _RESOURCE_SUFFIXES
        ]

        # The body is deliberately not kept here — listing a skill must stay
        # cheap, and the body is only read when the agent asks for it.
        del body

        known = {"name", "description"}
        return Skill(
            name=str(front.get("name") or directory.name),
            description=str(front.get("description") or ""),
            location=str(directory),
            resources=resources,
            metadata={k: str(v) for k, v in front.items() if k not in known},
        )

    # -- SkillSource -------------------------------------------------------- #

    async def list_skills(self, names: Sequence[str] | None = None) -> list[Skill]:
        return await asyncio.to_thread(self._list_sync, names)

    def _list_sync(self, names: Sequence[str] | None) -> list[Skill]:
        wanted = set(names) if names is not None else None
        skills: list[Skill] = []
        for name, directory in self._skill_dirs().items():
            if wanted is not None and name not in wanted:
                continue
            try:
                skills.append(self._load(directory))
            except OSError:
                # An unreadable skill folder should not break the whole listing.
                continue
        return skills

    async def read_skill(self, name: str) -> str:
        return await asyncio.to_thread(self._read_skill_sync, name)

    def _read_skill_sync(self, name: str) -> str:
        directory = self._skill_dirs().get(name)
        if directory is None:
            raise KeyError(f"No skill named {name!r}")
        path = directory / "SKILL.md"
        mtime = path.stat().st_mtime
        cached = self._cache.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        _, body = parse_skill_markdown(path.read_text(encoding="utf-8"), fallback_name=name)
        self._cache[name] = (mtime, body)
        return body

    async def read_resource(self, name: str, resource: str) -> str:
        return await asyncio.to_thread(self._read_resource_sync, name, resource)

    def _read_resource_sync(self, name: str, resource: str) -> str:
        directory = self._skill_dirs().get(name)
        if directory is None:
            raise KeyError(f"No skill named {name!r}")

        target = (directory / resource).resolve()
        # The path comes from the model, so containment is checked rather than
        # assumed: a skill may only read its own files.
        if not target.is_relative_to(directory.resolve()):
            raise ValueError(f"Resource path escapes the skill directory: {resource!r}")
        if not target.is_file():
            raise KeyError(f"Skill {name!r} has no resource {resource!r}")
        if target.stat().st_size > self._max_resource_bytes:
            raise ValueError(
                f"Resource {resource!r} is larger than the {self._max_resource_bytes}-byte limit."
            )
        return target.read_text(encoding="utf-8", errors="replace")

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"FileSkillSource({str(self._root)!r})"
