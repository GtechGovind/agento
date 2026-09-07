"""Skills — reusable instruction packs the agent loads only when relevant.

A skill is a folder with a ``SKILL.md`` in it. The front matter names it and says
when it applies; the body is a procedure — how to review a contract, how to
format a report, how your team wants a migration written.

The point is **progressive disclosure**. Only each skill's name and description
go into the system prompt, which costs a line or two. The body is read on demand,
through the ``read_skill`` tool, when the agent judges the skill relevant. So
twenty skills cost twenty lines of context rather than twenty documents, and the
agent still has all twenty available.

::

    skills/
      contract-review/
        SKILL.md          # front matter + procedure
        checklist.md      # referenced, read on demand
      quarterly-report/
        SKILL.md

``SKILL.md`` follows the widely used format::

    ---
    name: contract-review
    description: Review a commercial contract for unusual liability, term and
      termination clauses. Use when asked to check or summarize a contract.
    ---

    ## Procedure

    1. Read the whole document before commenting...

The ``description`` is the single most important line in the file: it is the only
thing the model sees before deciding whether to open the skill. Write it as *when
to use this*, not as a title.

:class:`SkillSource` is the interface. The bundled implementation reads a local
directory; a source backed by git, S3 or a database is a small class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Skill", "SkillSource"]


class Skill(BaseModel):
    """A skill's identity and location — not its body.

    Attributes:
        name: Unique name. Also how the agent refers to it in ``read_skill``.
        description: When to use this skill. Goes in the system prompt.
        location: Where it lives, for display — a path, a URL, a row id.
        resources: Other files in the skill, relative to it. Advertised so the
            agent knows what it may ask for, without their contents costing
            anything until it does.
        metadata: Anything else the front matter carried.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    location: str | None = None
    resources: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class SkillSource(Protocol):
    """Where skills come from."""

    async def list_skills(self, names: Sequence[str] | None = None) -> list[Skill]:
        """List available skills — metadata only, never bodies.

        Args:
            names: Restrict to these names. Unknown names are skipped rather than
                raising; the agent layer decides how strict to be.

        Returns:
            Matching skills.
        """
        ...

    async def read_skill(self, name: str) -> str:
        """Read a skill's ``SKILL.md`` body.

        Raises:
            KeyError: No such skill.
        """
        ...

    async def read_resource(self, name: str, resource: str) -> str:
        """Read one of a skill's other files.

        Args:
            name: The skill.
            resource: Path relative to the skill's directory.

        Raises:
            KeyError: No such skill or resource.
            ValueError: The path escapes the skill's directory.
        """
        ...
