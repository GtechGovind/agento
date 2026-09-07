"""Artifacts — where content too large for the context window goes.

TrueForge writes oversized tool results into a sandbox filesystem. agento has no
sandbox, so it writes them into an **artifact store**: a small key/value
interface for named blobs, with a built-in tool that lets the agent read parts of
one back.

The store is used in three places:

* A tool result over the size threshold is stored, and replaced in the
  conversation with an id, a preview, and instructions for reading more.
* A file the user attaches that cannot be sent to the model inline.
* Anything a tool of yours chooses to store, through
  :attr:`~agento.core.tools.context.ToolContext.artifacts`.

Two implementations ship: :class:`~agento.artifacts.memory.MemoryArtifactStore`
(process-local, good for tests) and
:class:`~agento.artifacts.local.LocalArtifactStore` (a directory on disk). The
protocol is five methods, so S3, a database blob column or your existing document
service is a short adapter.

**Range reads matter.** ``read(artifact_id, offset=..., length=...)`` is what
makes offloading useful rather than just tidy: an agent facing a 40 MB JSON file
reads the first 2 KB, works out its shape, and pulls only the part it needs —
which is the whole point of getting it out of the context window.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..core.events import now_iso

__all__ = ["Artifact", "ArtifactStore"]


class Artifact(BaseModel):
    """Metadata for one stored blob.

    Attributes:
        id: Stable identifier. This is what the agent is told and what it passes
            to ``read_artifact``.
        name: Human-readable name — a filename, or a name derived from the tool
            that produced it.
        size_bytes: Size of the content.
        mime_type: Best-known type, when there is one.
        created_at: ISO-8601 creation time.
        source: Where it came from — a tool name, or ``"user-upload"``.
        metadata: Anything the producer wants to keep alongside it.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    size_bytes: int
    mime_type: str | None = None
    created_at: str = Field(default_factory=now_iso)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ArtifactStore(Protocol):
    """Somewhere to keep content that should not live in the context window."""

    async def write(
        self,
        *,
        name: str,
        content: bytes,
        mime_type: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Store content and return its metadata.

        Args:
            name: Human-readable name. Not required to be unique — the returned
                ``id`` is the identity.
            content: The bytes.
            mime_type: Type, if known.
            source: Producer — a tool name, or ``"user-upload"``.
            metadata: Anything else worth keeping.

        Returns:
            The stored :class:`Artifact`, with its assigned ``id``.
        """
        ...

    async def read(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read all or part of an artifact.

        Args:
            artifact_id: Which artifact.
            offset: Byte offset to start at.
            length: How many bytes; ``None`` reads to the end.

        Raises:
            KeyError: No such artifact.
        """
        ...

    async def stat(self, artifact_id: str) -> Artifact | None:
        """Metadata for an artifact, or ``None`` if it does not exist."""
        ...

    async def list(self, *, source: str | None = None, limit: int = 100) -> list[Artifact]:
        """List artifacts, newest first.

        Args:
            source: Only artifacts from this producer.
            limit: Maximum to return.
        """
        ...

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact. Returns whether it existed."""
        ...
