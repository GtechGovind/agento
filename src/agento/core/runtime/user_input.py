"""Turning user input into context messages.

A plain string is trivial. Attachments are not, and the interesting decision is
what to do with each one:

* **Images and PDFs** go to the model inline. Modern models read them natively,
  and a picture described in words is a worse picture.
* **Everything else** — a 40 MB CSV, a zip, a log file — is written to the
  artifact store, and the model is told the id, name, type and size. It reads
  what it needs through ``read_artifact`` instead of having the whole thing
  pasted into its context window, where it would crowd out the actual task.

If no artifact store is configured, a non-inline attachment is rejected with a
clear message rather than silently dropped: losing a file the user attached is
worse than telling them it cannot be handled.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from ...errors import InvalidFileInputError
from ..events import ArtifactCreated, UserMessage
from ..messages import FilePart, ImagePart, LLMUserMessage, TextPart
from .context_utils import internal_message

__all__ = ["ProcessedUserInput", "process_user_message"]

_INLINE_MIME_PREFIXES = ("image/", "application/pdf")
_DATA_URI_MIME = re.compile(r"^data:([^;,]+)")


class ProcessedUserInput:
    """The result of turning one user message into context.

    Attributes:
        messages: What to append — the user message, plus a note about any
            stored attachments.
        events: :class:`~agento.core.events.ArtifactCreated` for each stored file.
    """

    __slots__ = ("messages", "events")

    def __init__(self, messages: list[LLMUserMessage], events: list[Any]) -> None:
        self.messages = messages
        self.events = events


def _mime_of(data_uri: str) -> str:
    match = _DATA_URI_MIME.match(data_uri)
    if not match:
        raise InvalidFileInputError(
            "File data must be a data URI of the form 'data:<mime>;base64,<payload>'"
        )
    return match.group(1)


def _decode(data_uri: str) -> bytes:
    comma = data_uri.find(",")
    if comma == -1:
        raise InvalidFileInputError("File data URI has no payload")
    payload = data_uri[comma + 1 :]
    if not payload:
        raise InvalidFileInputError("File data URI has an empty payload")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise InvalidFileInputError(f"File data URI is not valid base64: {exc}") from exc


def _safe_name(name: str) -> str:
    """Reject path traversal in an attachment name.

    Attachment names reach an artifact store, and a store backed by a directory
    would happily write ``../../etc/something`` if allowed to.
    """
    cleaned = name.lstrip("/")
    if not cleaned:
        raise InvalidFileInputError("File name must not be empty")
    for segment in cleaned.replace("\\", "/").split("/"):
        if segment == "..":
            raise InvalidFileInputError(f"File name contains path traversal: {name}")
    return cleaned


async def process_user_message(
    message: UserMessage,
    *,
    artifacts: Any = None,
    source: str = "user-upload",
) -> ProcessedUserInput:
    """Convert a :class:`~agento.core.events.UserMessage` into context messages.

    Args:
        message: The input item.
        artifacts: The configured artifact store, or ``None``.
        source: Recorded on stored artifacts.

    Returns:
        A :class:`ProcessedUserInput`.

    Raises:
        InvalidFileInputError: An attachment is malformed, or needs an artifact
            store and none is configured.
    """
    if isinstance(message.content, str):
        return ProcessedUserInput([LLMUserMessage(content=message.content)], [])

    inline_parts: list[Any] = []
    text_parts: list[TextPart] = []
    stored: list[dict[str, Any]] = []
    events: list[Any] = []

    for part in message.content:
        if isinstance(part, TextPart):
            text_parts.append(part)
            continue
        if isinstance(part, ImagePart):
            inline_parts.append(part)
            continue
        if not isinstance(part, FilePart):
            continue

        name = _safe_name(part.name)
        mime = _mime_of(part.data)

        if mime.startswith(_INLINE_MIME_PREFIXES):
            if mime.startswith("image/"):
                inline_parts.append(ImagePart(url=part.data))
            else:
                inline_parts.append(part)
            continue

        if artifacts is None:
            raise InvalidFileInputError(
                f"Attachment {name!r} ({mime}) cannot be sent to the model inline and no artifact "
                "store is configured. Pass artifacts=... to Agento to enable file attachments."
            )

        content = _decode(part.data)
        artifact = await artifacts.write(
            name=name, content=content, mime_type=mime, source=source
        )
        stored.append(
            {"id": artifact.id, "name": name, "mime": mime, "size": artifact.size_bytes}
        )
        events.append(
            ArtifactCreated(
                artifact_id=artifact.id,
                name=name,
                size_bytes=artifact.size_bytes,
                mime_type=mime,
                source_tool=source,
            )
        )

    parts: list[Any] = [*inline_parts, *text_parts]
    messages: list[LLMUserMessage] = []
    if parts:
        messages.append(LLMUserMessage(content=parts))
    elif not stored:
        messages.append(LLMUserMessage(content=""))

    if stored:
        listing = "\n\n".join(
            f"[file_{index + 1}]\n  name: {item['name']}\n  artifact_id: {item['id']}\n"
            f"  type: {item['mime']}\n  size: {item['size']} bytes"
            for index, item in enumerate(stored)
        )
        messages.append(
            internal_message(
                f"The user attached {len(stored)} file(s). Read them with the read_artifact tool.\n\n{listing}"
            )
        )

    return ProcessedUserInput(messages, events)
