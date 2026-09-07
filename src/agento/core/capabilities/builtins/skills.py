"""Skills as a capability — advertise in the prompt, load on demand.

The system prompt gets one small block per attached skill: its name, its
description, and the files it contains. That is a couple of lines each. The
procedure itself — which may be pages — is read only when the agent decides the
skill applies, through the ``read_skill`` tool.

That split is the whole idea. Twenty skills cost twenty lines rather than twenty
documents, so an agent can carry a large library of institutional knowledge and
still start each turn with a small prompt.

See :mod:`agento.skills.base` for the ``SKILL.md`` format.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ...instructions import InstructionBuilder
from ...tools.base import ToolSet
from ...tools.local import LocalToolSet, Tool
from ..base import Capability

__all__ = ["Skills"]

SKILLS_PREAMBLE = """\
Skills are procedures written for this agent. Each entry below gives a name, a
description of when it applies, and any extra files it contains.

- Judge from the description whether a skill applies to the current task.
- If it does, call read_skill(name) to read the procedure BEFORE starting, and
  then follow it.
- Only read a skill's extra files (read_skill_file) when the procedure refers to
  one and you need it.
- Do not guess what a skill says from its name."""


class _SkillTools(LocalToolSet):
    """``read_skill`` and ``read_skill_file``, bound to one source."""

    def __init__(self, source: Any, allowed: Sequence[str]) -> None:
        self._source = source
        self._allowed = set(allowed)
        super().__init__(
            "skills",
            [
                Tool(self._build_read_skill(), name="read_skill", read_only=True),
                Tool(self._build_read_file(), name="read_skill_file", read_only=True),
            ],
            description="Read the procedures attached to this agent.",
            kind="builtin",
        )

    def _build_read_skill(self) -> Any:
        async def read_skill(name: str) -> str:
            """Read a skill's full procedure.

            Call this before following a skill. The system prompt lists only each
            skill's name and description; the procedure lives here.

            Args:
                name: The skill name, exactly as listed in the prompt.
            """
            if name not in self._allowed:
                return json.dumps(
                    {
                        "error": f"Skill {name!r} is not attached to this agent",
                        "available_skills": sorted(self._allowed),
                    }
                )
            try:
                return await self._source.read_skill(name)
            except KeyError:
                return json.dumps({"error": f"Skill {name!r} could not be found"})

        return read_skill

    def _build_read_file(self) -> Any:
        async def read_skill_file(name: str, path: str) -> str:
            """Read one of a skill's extra files.

            Only use this for files the skill's procedure actually refers to.

            Args:
                name: The skill name.
                path: The file path, relative to the skill, as listed in the prompt.
            """
            if name not in self._allowed:
                return json.dumps({"error": f"Skill {name!r} is not attached to this agent"})
            try:
                return await self._source.read_resource(name, path)
            except KeyError:
                return json.dumps({"error": f"Skill {name!r} has no file {path!r}"})
            except ValueError as exc:
                return json.dumps({"error": str(exc)})

        return read_skill_file


class Skills(Capability):
    """Attaches skills to an agent.

    Args:
        source: Where to read skill content from.
        skills: The skills to attach, already resolved to metadata by the session
            layer (which is what turns the agent's list of names into objects).
    """

    name = "skills"

    def __init__(self, source: Any, skills: Sequence[Any]) -> None:
        self._source = source
        self._skills = list(skills)
        self._tools = (
            _SkillTools(source, [skill.name for skill in self._skills]) if self._skills else None
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools] if self._tools is not None else []

    def build_instructions(self, builder: InstructionBuilder) -> None:
        if not self._skills:
            return
        section = builder.begin_section("skills")
        section.add_content(SKILLS_PREAMBLE)
        for skill in self._skills:
            body = f"name: {skill.name}\ndescription: {skill.description}"
            if skill.resources:
                listed = ", ".join(skill.resources[:20])
                if len(skill.resources) > 20:
                    listed += f", and {len(skill.resources) - 20} more"
                body += f"\nfiles: {listed}"
            section.add_section("skill", body)
