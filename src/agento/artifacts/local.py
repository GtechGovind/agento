"""An artifact store backed by a directory on disk.

The default for real use. Each artifact becomes two files::

    <root>/<id>.bin     the content
    <root>/<id>.json    the metadata

Two files rather than one because metadata must be readable without loading the
content — an agent listing artifacts, or checking a size before deciding how much
to read, should not pull a 40 MB blob into memory to find out it is 40 MB.

IDs are generated independently of names. Read/stat/delete validate every ID
and reject existing symlinks; the root must remain owned by the host.

All I/O runs on a worker thread, so a large write cannot stall the event loop
while an agent is streaming.
"""

from __future__ import annotations

import asyncio
import builtins
import re
from pathlib import Path
from typing import Any

from .._ids import new_id
from .base import Artifact

__all__ = ["LocalArtifactStore"]


class LocalArtifactStore:
    """Artifact storage in a local directory.

    Args:
        root: Directory to use. Created if missing.
        max_bytes_per_artifact: Reject writes larger than this. The default of
            256 MB is a guard against an agent looping on a runaway tool, not a
            considered storage policy — raise it if you have a reason.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes_per_artifact: int = 256 * 1024 * 1024,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes_per_artifact

    # -- paths -------------------------------------------------------------- #

    def _blob_path(self, artifact_id: str) -> Path:
        return self._path(artifact_id, ".bin")

    def _meta_path(self, artifact_id: str) -> Path:
        return self._path(artifact_id, ".json")

    def _path(self, artifact_id: str, suffix: str) -> Path:
        if re.fullmatch(r"[0-9a-hjkmnp-tv-z]{26}", artifact_id) is None:
            raise ValueError("Invalid artifact ID")
        path = self._root / f"{artifact_id}{suffix}"
        if path.is_symlink() or not path.resolve().is_relative_to(self._root):
            raise ValueError("Artifact path escapes the store or is a symlink")
        return path

    # -- ArtifactStore ------------------------------------------------------ #

    async def write(
        self,
        *,
        name: str,
        content: bytes,
        mime_type: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        if len(content) > self._max_bytes:
            raise ValueError(
                f"Artifact {name!r} is {len(content)} bytes, over the "
                f"{self._max_bytes}-byte limit for this store."
            )

        artifact = Artifact(
            id=new_id(),
            name=name,
            size_bytes=len(content),
            mime_type=mime_type,
            source=source,
            metadata=dict(metadata or {}),
        )
        await asyncio.to_thread(self._write_sync, artifact, content)
        return artifact

    def _write_sync(self, artifact: Artifact, content: bytes) -> None:
        # Content first, then metadata: a reader that sees the metadata file can
        # rely on the content being complete.
        self._blob_path(artifact.id).write_bytes(content)
        self._meta_path(artifact.id).write_text(artifact.model_dump_json(), encoding="utf-8")

    async def read(self, artifact_id: str, *, offset: int = 0, length: int | None = None) -> bytes:
        path = self._blob_path(artifact_id)
        if not await asyncio.to_thread(path.exists):
            raise KeyError(f"No artifact with id {artifact_id!r}")
        return await asyncio.to_thread(self._read_sync, path, offset, length)

    @staticmethod
    def _read_sync(path: Path, offset: int, length: int | None) -> bytes:
        with path.open("rb") as handle:
            if offset:
                handle.seek(offset)
            return handle.read() if length is None else handle.read(length)

    async def stat(self, artifact_id: str) -> Artifact | None:
        path = self._meta_path(artifact_id)
        if not await asyncio.to_thread(path.exists):
            return None
        return await asyncio.to_thread(self._stat_sync, path)

    @staticmethod
    def _stat_sync(path: Path) -> Artifact | None:
        try:
            return Artifact.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt or half-written metadata file should read as "missing"
            # rather than crash an agent mid-turn.
            return None

    async def list(self, *, source: str | None = None, limit: int = 100) -> list[Artifact]:
        return await asyncio.to_thread(self._list_sync, source, limit)

    def _list_sync(self, source: str | None, limit: int) -> builtins.list[Artifact]:
        artifacts: builtins.list[Artifact] = []
        for path in self._root.glob("*.json"):
            try:
                self._meta_path(path.stem)
            except ValueError:
                continue
            artifact = self._stat_sync(path)
            if artifact is None:
                continue
            if source is not None and artifact.source != source:
                continue
            artifacts.append(artifact)
        artifacts.sort(key=lambda _artifact: _artifact.created_at, reverse=True)
        return artifacts[:limit]

    async def delete(self, artifact_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, artifact_id)

    def _delete_sync(self, artifact_id: str) -> bool:
        blob = self._blob_path(artifact_id)
        meta = self._meta_path(artifact_id)
        existed = blob.exists() or meta.exists()
        blob.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return existed

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"LocalArtifactStore({str(self._root)!r})"
