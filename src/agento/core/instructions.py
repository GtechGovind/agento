"""The system-prompt builder.

agento assembles its system prompt as a tree of named XML sections rather than
by concatenating strings. There are two reasons for that, both practical:

1. **Models follow structured prompts more reliably.** A tag like ``<skills>``
   gives the model an unambiguous boundary for a block of guidance, which matters
   a great deal once the prompt contains instructions from four or five different
   capabilities at once.
2. **Contributors do not have to coordinate.** A capability calls
   ``builder.add_section("my-thing", ...)`` and does not care what else is in the
   prompt or in what order. Empty sections vanish, so a capability that has
   nothing to say this run costs nothing.

Example::

    root = InstructionBuilder.system_prompt("You are the Agent...")
    caps = root.begin_section("agent-capabilities")
    caps.add_section("skills", "1. The skill is in `path`...")
    caps.add_section("sub-agents", "The Agent can delegate...")
    root.add_section("user-instructions", user_text, escape=True)
    print(root.build())

produces::

    <system-prompt>
    <agent-identity>
    You are the Agent...
    </agent-identity>

    <agent-capabilities>
    <skills>
    1. The skill is in `path`...
    </skills>

    <sub-agents>
    The Agent can delegate...
    </sub-agents>
    </agent-capabilities>

    <user-instructions><![CDATA[
    ...
    ]]></user-instructions>
    </system-prompt>

User-authored text is wrapped in ``CDATA`` (``escape=True``) so that a prompt
containing ``</skills>`` or any other tag-like text cannot terminate a section
early and confuse the model about where its instructions end.
"""

from __future__ import annotations

__all__ = ["InstructionBuilder", "ROOT_AGENT_IDENTITY", "SUB_AGENT_IDENTITY"]


ROOT_AGENT_IDENTITY = (
    "You are the Agent, an AI assistant that helps users accomplish tasks by "
    "leveraging available tools and capabilities."
)

SUB_AGENT_IDENTITY = (
    "You are the Agent operating as a sub-agent that has been delegated a specific task. "
    "The Agent has access to the same tools as the parent agent. The Agent must focus on "
    "completing the delegated task and return a concise result. The Agent cannot ask "
    "questions to the user."
)


class InstructionBuilder:
    """A node in the system-prompt tree.

    Each instance is one XML section. It can hold raw text, nested sections, or
    both, and children always render in insertion order.

    Args:
        tag: The XML tag name for this section.
        escape: When true, the body is wrapped in a CDATA block. Use for any text
            that did not come from agento itself.
    """

    __slots__ = ("_tag", "_escape", "_children")

    def __init__(self, tag: str, escape: bool = False) -> None:
        self._tag = tag
        self._escape = escape
        self._children: list[str | InstructionBuilder] = []

    @classmethod
    def system_prompt(cls, identity: str) -> InstructionBuilder:
        """Create the root ``<system-prompt>`` node with an ``<agent-identity>``.

        Args:
            identity: Who the agent is. Use :data:`ROOT_AGENT_IDENTITY` or
                :data:`SUB_AGENT_IDENTITY` unless you have a reason not to.
        """
        builder = cls("system-prompt")
        builder.add_section("agent-identity", identity)
        return builder

    def add_content(self, content: str) -> InstructionBuilder:
        """Append raw text to this section, with no wrapping tag.

        Blank content is ignored, so callers never need to guard against it.

        Returns:
            ``self``, for chaining.
        """
        trimmed = content.strip()
        if trimmed:
            self._children.append(trimmed)
        return self

    def add_section(self, tag: str, content: str, escape: bool = False) -> InstructionBuilder:
        """Append a leaf section — ``<tag>content</tag>``.

        Args:
            tag: The section's XML tag.
            content: The section body. Blank content is ignored and the section
                is not emitted at all.
            escape: Wrap the body in CDATA. Set this for user-authored text.

        Returns:
            ``self``, for chaining.
        """
        trimmed = content.strip()
        if not trimmed:
            return self
        child = InstructionBuilder(tag, escape)
        child._children.append(trimmed)
        self._children.append(child)
        return self

    def begin_section(self, tag: str) -> InstructionBuilder:
        """Create a nested section and return it, so you can add children to it.

        Args:
            tag: The nested section's XML tag.

        Returns:
            The **child** builder — not ``self``. Keep a reference to the parent
            if you need to add siblings afterwards.
        """
        child = InstructionBuilder(tag)
        self._children.append(child)
        return child

    def is_empty(self) -> bool:
        """True when this section would render to nothing."""
        return not self.build()

    def build(self) -> str:
        """Render this node and everything under it to a string.

        Sections that would be empty render as the empty string and are dropped
        by their parent, so a capability that contributes nothing leaves no trace
        in the prompt.
        """
        parts: list[str] = []
        for child in self._children:
            if isinstance(child, str):
                parts.append(child)
            else:
                rendered = child.build()
                if rendered:
                    parts.append(rendered)

        if not parts:
            return ""

        body = "\n\n".join(parts)
        if self._escape:
            return f"<{self._tag}><![CDATA[\n{body}\n]]></{self._tag}>"
        return f"<{self._tag}>\n{body}\n</{self._tag}>"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.build()
