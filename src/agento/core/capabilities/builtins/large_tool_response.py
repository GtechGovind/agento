"""Keeping oversized tool results out of the context window.

A single tool call can return more than the agent can afford to read. A search
API returning 200 results, a database query returning 50,000 rows, a log file —
dropped into the conversation, one of these can consume most of the window, and
it stays there for every subsequent model call in the session.

This capability intercepts results before they reach the context. Anything over
the threshold is written to the artifact store, and the conversation gets an id,
a size, and a preview of the first and last few hundred characters. The agent
then reads what it actually needs with two tools:

``read_artifact(artifact_id, offset, length)``
    A window into the content. Start at the beginning to see the shape, then seek.

``search_artifact(artifact_id, pattern, ...)``
    Regex search returning matching lines with line numbers — grep, essentially.
    This is what makes a 40 MB file genuinely workable rather than merely stored.

Two thresholds, because they catch different failures:

* ``individual_token_threshold`` (default 6,000) — one huge result.
* ``total_token_threshold`` (default 10,000) — many medium results in one batch,
  none individually alarming, together ruinous. The largest are offloaded until
  the batch fits.

Failed results are truncated rather than stored: a stack trace has no long tail
worth reading, and storing it would waste an artifact id.

With no artifact store configured the capability still works — results are
truncated to a preview with guidance to narrow the call. Lossier, but it keeps
the context window intact, which is the point.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ...events import ArtifactCreated
from ...tokens import estimate_tokens
from ...tools.base import ToolSet
from ...tools.context import ToolContext
from ...tools.local import LocalToolSet, Tool
from ..base import Capability, CapabilityOutput, EmitEvent, ExecutionContext

__all__ = [
    "DEFAULT_INDIVIDUAL_TOKEN_THRESHOLD",
    "DEFAULT_PREVIEW_CHARS",
    "DEFAULT_TOTAL_TOKEN_THRESHOLD",
    "LargeToolResponse",
]

DEFAULT_INDIVIDUAL_TOKEN_THRESHOLD = 6_000
DEFAULT_TOTAL_TOKEN_THRESHOLD = 10_000
DEFAULT_PREVIEW_CHARS = 400
_FAILURE_TRUNCATION_CHARS = 800
_MAX_READ_CHARS = 20_000
_MAX_SEARCH_MATCHES = 100


def _preview(content: str, chars: int) -> str:
    """First and last ``chars`` characters, with the middle elided.

    Both ends, not just the head: the shape of a JSON payload is often only
    obvious from its closing structure, and the last rows of a table are as
    informative as the first.
    """
    if len(content) <= chars * 2:
        return content
    return f"{content[:chars]}\n\n... [{len(content) - chars * 2} characters omitted] ...\n\n{content[-chars:]}"


# --------------------------------------------------------------------------- #
# Tools the agent uses to read back what was offloaded                         #
# --------------------------------------------------------------------------- #


async def _read_artifact(
    artifact_id: str,
    ctx: ToolContext,
    offset: int = 0,
    length: int = 4000,
) -> str:
    """Read part of a stored artifact.

    Large tool results and uploaded files are stored rather than pasted into the
    conversation. Use this to read the parts you need.

    Start at offset 0 to see the beginning and work out the structure, then read
    further, or use search_artifact to jump to what matters.

    Args:
        artifact_id: The artifact id given in the tool result or file listing.
        offset: Character offset to start reading from.
        length: How many characters to read. Capped at 20000.
    """
    store = ctx.artifacts
    if store is None:
        return json.dumps({"error": "No artifact store is configured."})

    meta = await store.stat(artifact_id)
    if meta is None:
        return json.dumps({"error": f"No artifact with id {artifact_id!r}"})

    length = max(1, min(length, _MAX_READ_CHARS))
    try:
        raw = await store.read(artifact_id, offset=offset, length=length)
    except KeyError:
        return json.dumps({"error": f"No artifact with id {artifact_id!r}"})

    text = raw.decode("utf-8", errors="replace")
    return json.dumps(
        {
            "artifact_id": artifact_id,
            "name": meta.name,
            "total_bytes": meta.size_bytes,
            "offset": offset,
            "returned_bytes": len(raw),
            "has_more": offset + len(raw) < meta.size_bytes,
            "content": text,
        }
    )


async def _search_artifact(
    artifact_id: str,
    pattern: str,
    ctx: ToolContext,
    max_matches: int = 20,
    context_lines: int = 0,
) -> str:
    """Search an artifact with a regular expression, like grep.

    Returns matching lines with their line numbers, so you can then read around
    them precisely instead of scanning the whole artifact.

    Args:
        artifact_id: The artifact to search.
        pattern: A Python regular expression.
        max_matches: Maximum matching lines to return. Capped at 100.
        context_lines: Lines of context to include either side of each match.
    """
    store = ctx.artifacts
    if store is None:
        return json.dumps({"error": "No artifact store is configured."})

    meta = await store.stat(artifact_id)
    if meta is None:
        return json.dumps({"error": f"No artifact with id {artifact_id!r}"})

    try:
        expression = re.compile(pattern)
    except re.error as exc:
        return json.dumps({"error": f"Invalid regular expression: {exc}"})

    raw = await store.read(artifact_id)
    lines = raw.decode("utf-8", errors="replace").splitlines()
    limit = max(1, min(max_matches, _MAX_SEARCH_MATCHES))

    matches: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not expression.search(line):
            continue
        entry: dict[str, Any] = {"line": number, "text": line[:1000]}
        if context_lines > 0:
            start = max(0, number - 1 - context_lines)
            end = min(len(lines), number + context_lines)
            entry["context"] = [item[:500] for item in lines[start:end]]
        matches.append(entry)
        if len(matches) >= limit:
            break

    return json.dumps(
        {
            "artifact_id": artifact_id,
            "name": meta.name,
            "total_lines": len(lines),
            "matches": matches,
            "truncated": len(matches) >= limit,
        }
    )


class LargeToolResponse(Capability):
    """Offloads oversized tool results and gives the agent tools to read them.

    Args:
        individual_token_threshold: One result larger than this is offloaded.
        total_token_threshold: A batch larger than this has its biggest results
            offloaded until it fits.
        preview_chars: Characters kept from each end of an offloaded result.
        sub_agents_available: Whether to mention delegation in the guidance. Only
            worth suggesting when the agent can actually do it.
    """

    name = "large_tool_response"

    def __init__(
        self,
        *,
        individual_token_threshold: int = DEFAULT_INDIVIDUAL_TOKEN_THRESHOLD,
        total_token_threshold: int = DEFAULT_TOTAL_TOKEN_THRESHOLD,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
        sub_agents_available: bool = False,
    ) -> None:
        if individual_token_threshold > total_token_threshold:
            raise ValueError(
                "individual_token_threshold must be <= total_token_threshold "
                f"(got {individual_token_threshold} > {total_token_threshold})"
            )
        self._individual = individual_token_threshold
        self._total = total_token_threshold
        self._preview_chars = preview_chars
        self._sub_agents = sub_agents_available
        self._tools = LocalToolSet(
            "artifacts",
            [
                Tool(_read_artifact, name="read_artifact", read_only=True),
                Tool(_search_artifact, name="search_artifact", read_only=True),
            ],
            description="Read stored artifacts: large tool results and uploaded files.",
            kind="builtin",
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools]

    # -- offloading --------------------------------------------------------- #

    def _guidance(self, stored: bool) -> str:
        steps = ["Narrow the call — most tools accept a filter, a limit, or a field selection."]
        if stored:
            steps.append(
                "Read the stored artifact with read_artifact, or find what you need with "
                "search_artifact."
            )
        if self._sub_agents:
            steps.append(
                "Delegate the reading to a sub-agent and ask it for just the summary you need."
            )
        return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))

    async def process_tool_results(
        self,
        results: list[Any],
        context: ExecutionContext,
    ) -> list[CapabilityOutput]:
        if not results:
            return []

        sizes = [estimate_tokens(result.message.content) for result in results]
        running_total = sum(sizes)
        if running_total < self._individual:
            return []

        outputs: list[CapabilityOutput] = []
        offloaded: set[int] = set()

        # Pass one: anything individually oversized.
        for index, (result, size) in enumerate(zip(results, sizes, strict=True)):
            if size >= self._individual:
                running_total -= await self._shrink(result, context, outputs)
                running_total += estimate_tokens(result.message.content)
                offloaded.add(index)

        # Pass two: biggest first, until the batch as a whole fits.
        remaining = sorted(
            (index for index in range(len(results)) if index not in offloaded),
            key=lambda index: sizes[index],
            reverse=True,
        )
        for index in remaining:
            if running_total < self._total:
                break
            running_total -= await self._shrink(results[index], context, outputs)
            running_total += estimate_tokens(results[index].message.content)

        return outputs

    async def _shrink(
        self,
        result: Any,
        context: ExecutionContext,
        outputs: list[CapabilityOutput],
    ) -> int:
        """Replace one result's content with something small. Returns its old size."""
        original = result.message.content
        original_tokens = estimate_tokens(original)

        if result.failed:
            # An error message has no long tail worth keeping.
            result.message.content = original[:_FAILURE_TRUNCATION_CHARS]
            return original_tokens

        store = context.artifacts
        preview = _preview(original, self._preview_chars)

        if store is None:
            result.message.content = (
                f"This result was too large to include ({len(original)} characters). No artifact "
                f"store is configured, so it could not be stored.\n\n"
                f"{self._guidance(stored=False)}\n\nPreview:\n{preview}"
            )
            return original_tokens

        tool_name = getattr(result.tool_call.function, "name", "tool")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_name)[:60]

        try:
            artifact = await store.write(
                name=f"{safe_name}.txt",
                content=original.encode("utf-8"),
                mime_type="application/json" if result.outcome.is_structured else "text/plain",
                source=tool_name,
                metadata={"thread_id": context.thread_id, "tool_call_id": result.tool_call.id},
            )
        except Exception as exc:
            # Storage failing must not fail the turn: fall back to a preview.
            result.message.content = (
                f"This result was too large to include ({len(original)} characters), and storing "
                f"it failed ({type(exc).__name__}).\n\n{self._guidance(stored=False)}\n\n"
                f"Preview:\n{preview}"
            )
            return original_tokens

        result.artifact_id = artifact.id
        result.message.content = (
            f"This result was too large to include ({len(original)} characters, "
            f"~{original_tokens} tokens) and has been stored as artifact "
            f"`{artifact.id}`.\n\n{self._guidance(stored=True)}\n\nPreview:\n{preview}"
        )
        outputs.append(
            EmitEvent(
                event=ArtifactCreated(
                    thread_id=context.thread_id,
                    artifact_id=artifact.id,
                    name=artifact.name,
                    size_bytes=artifact.size_bytes,
                    mime_type=artifact.mime_type,
                    source_tool=tool_name,
                )
            )
        )
        return original_tokens
