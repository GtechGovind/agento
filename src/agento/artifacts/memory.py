"""An artifact store that lives in the process.

Good for tests, notebooks, and short-lived work where nothing needs to outlive
the process. Not good for anything else: the contents disappear when the process
does, and nothing is shared between workers.

``max_bytes`` guards against the failure this store makes easy — an agent that
offloads a hundred large tool results into RAM. When the cap is exceeded the
oldest artifacts are evicted, which is the right trade here: an agent that can no
longer read an old artifact degrades, whereas one that exhausts memory takes the
process with it.
"""

from __future__ import annotations

from typing import Any

from .._ids import new_id
from .base import Artifact

__all__ = ["MemoryArtifactStore"]


class MemoryArtifactStore:
    """In-process artifact storage.

    Args:
        max_bytes: Soft cap on total stored bytes. Oldest artifacts are evicted
            when it is exceeded. ``None`` disables eviction.
    """

    def __init__(self, *, max_bytes: int | None = 256 * 1024 * 1024) -> None:
        self._max_bytes = max_bytes
        self._blobs: dict[str, bytes] = {}
        self._meta: dict[str, Artifact] = {}
        self._total = 0

    async def write(
        self,
        *,
        name: str,
        content: bytes,
        mime_type: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact_id = new_id()
        artifact = Artifact(
            id=artifact_id,
            name=name,
            size_bytes=len(content),
            mime_type=mime_type,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._blobs[artifact_id] = content
        self._meta[artifact_id] = artifact
        self._total += len(content)
        self._evict_if_needed()
        return artifact

    async def read(self, artifact_id: str, *, offset: int = 0, length: int | None = None) -> bytes:
        blob = self._blobs.get(artifact_id)
        if blob is None:
            raise KeyError(f"No artifact with id {artifact_id!r}")
        if length is None:
            return blob[offset:]
        return blob[offset : offset + length]

    async def stat(self, artifact_id: str) -> Artifact | None:
        return self._meta.get(artifact_id)

    async def list(self, *, source: str | None = None, limit: int = 100) -> list[Artifact]:
        items = [
            artifact
            for artifact in self._meta.values()
            if source is None or artifact.source == source
        ]
        items.sort(key=lambda artifact: artifact.created_at, reverse=True)
        return items[:limit]

    async def delete(self, artifact_id: str) -> bool:
        blob = self._blobs.pop(artifact_id, None)
        self._meta.pop(artifact_id, None)
        if blob is None:
            return False
        self._total -= len(blob)
        return True

    def _evict_if_needed(self) -> None:
        """Drop the oldest artifacts until under the cap."""
        if self._max_bytes is None:
            return
        while self._total > self._max_bytes and self._blobs:
            oldest = min(self._meta.values(), key=lambda artifact: artifact.created_at)
            self._total -= len(self._blobs.pop(oldest.id, b""))
            self._meta.pop(oldest.id, None)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"MemoryArtifactStore({len(self._blobs)} artifacts, {self._total} bytes)"
