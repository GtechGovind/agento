"""Compose ordered, tagged instruction sections without recursive rendering.

Builders retain live child sections, so a capability can populate its section
before or after its siblings. Blank sections do not contribute to the output.
Use ``escape=True`` to place arbitrary text inside a CDATA section.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["InstructionBuilder", "ROOT_AGENT_IDENTITY", "SUB_AGENT_IDENTITY"]

ROOT_AGENT_IDENTITY = (
    "You are an assistant working within the host application's agent runtime. "
    "Carry out the user's task using the tools and guidance provided."
)
SUB_AGENT_IDENTITY = (
    "You are a delegated worker. Complete the assigned task using your available "
    "context and tools, then send a concise result to the coordinating agent. "
    "Stay within that assignment and do not ask the user questions."
)


@dataclass
class _Section:
    name: str
    quoted: bool = False
    entries: list[str | _Section] = field(default_factory=list)


def _serialize(root: _Section) -> str:
    completed: dict[int, str] = {}
    pending = [(root, False)]
    while pending:
        section, ready = pending.pop()
        if not ready:
            pending.append((section, True))
            pending.extend((entry, False) for entry in reversed(section.entries) if isinstance(entry, _Section))
            continue
        pieces = [entry if isinstance(entry, str) else completed[id(entry)] for entry in section.entries]
        body = "\n\n".join(piece for piece in pieces if piece)
        if not body:
            completed[id(section)] = ""
            continue
        if section.quoted:
            body = body.replace("]]>", "]]]]><![CDATA[>")
            completed[id(section)] = f"<{section.name}><![CDATA[\n{body}\n]]></{section.name}>"
        else:
            completed[id(section)] = f"<{section.name}>\n{body}\n</{section.name}>"
    return completed[id(root)]


class InstructionBuilder:
    """Mutable handle to one section of an instruction document."""

    __slots__ = ("_section",)

    def __init__(self, tag: str, escape: bool = False) -> None:
        self._section = _Section(tag, escape)

    @classmethod
    def system_prompt(cls, identity: str) -> InstructionBuilder:
        """Start a document containing its agent identity."""
        return cls("system-prompt").add_section("agent-identity", identity)

    def begin_section(self, tag: str) -> InstructionBuilder:
        """Append a child and return its handle; subsequent edits remain visible."""
        child = InstructionBuilder(tag)
        self._section.entries.append(child._section)
        return child

    def add_content(self, content: str) -> InstructionBuilder:
        """Append nonblank text and return this handle."""
        if value := content.strip():
            self._section.entries.append(value)
        return self

    def add_section(self, tag: str, content: str, escape: bool = False) -> InstructionBuilder:
        """Append a populated child and return this handle."""
        if content.strip():
            child = self.begin_section(tag)
            child._section.quoted = escape
            child.add_content(content)
        return self

    def is_empty(self) -> bool:
        """Check for any text without producing the serialized document."""
        remaining = [self._section]
        while remaining:
            for entry in remaining.pop().entries:
                if isinstance(entry, str):
                    return False
                remaining.append(entry)
        return True

    def build(self) -> str:
        """Render populated sections in insertion order."""
        return _serialize(self._section)

    def __str__(self) -> str:
        return self.build()
