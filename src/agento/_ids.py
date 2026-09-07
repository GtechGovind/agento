"""Identifier generation.

Event ids are **monotonic ULIDs**, and that choice is load-bearing rather than
cosmetic: the durable ordering of events inside a turn is defined as the
lexicographic order of their ids. A ULID's first 48 bits are a millisecond
timestamp, so ids sort chronologically as plain strings, and the monotonic
variant guarantees that two ids minted in the same millisecond still sort in
creation order. That is what lets a session store paginate an event log with a
simple ``WHERE id > :cursor ORDER BY id`` and never lose or reorder a row.

If ``python-ulid`` is installed we use it. If not, the fallback below implements
the same specification in about forty lines of stdlib, so agento's ordering
guarantees hold with zero dependencies.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

__all__ = ["new_event_id", "new_id", "new_thread_id"]

# Crockford's base32 alphabet: no I, L, O or U, so ids stay unambiguous when a
# human reads one out of a log.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_lock = threading.Lock()
_last_ms = -1
_last_randomness = 0

# 80 bits of randomness, per the ULID spec.
_RANDOM_BITS = 80
_MAX_RANDOMNESS = (1 << _RANDOM_BITS) - 1


def _encode(value: int, length: int) -> str:
    """Encode ``value`` as ``length`` Crockford base32 characters, zero-padded."""
    out = [""] * length
    for i in range(length - 1, -1, -1):
        out[i] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(out)


def _fallback_monotonic_ulid() -> str:
    """Generate a monotonic ULID using only the standard library.

    Within a single millisecond the randomness component is incremented rather
    than redrawn, which is exactly what the ULID spec prescribes for monotonicity
    and what makes ids minted in a tight loop still sort correctly.
    """
    global _last_ms, _last_randomness

    with _lock:
        now_ms = int(time.time() * 1000)
        if now_ms > _last_ms:
            _last_ms = now_ms
            _last_randomness = int.from_bytes(os.urandom(10), "big")
        else:
            # Same millisecond (or a clock that went backwards): keep the previous
            # timestamp and step the randomness so ordering is preserved.
            _last_randomness += 1
            if _last_randomness > _MAX_RANDOMNESS:
                # Overflow is astronomically unlikely; roll into the next
                # millisecond rather than emit a non-monotonic id.
                _last_ms += 1
                _last_randomness = int.from_bytes(os.urandom(10), "big")
        timestamp_ms = _last_ms
        randomness = _last_randomness

    return _encode(timestamp_ms, 10) + _encode(randomness, 16)


try:  # pragma: no cover - exercised only when the optional dep is installed
    from ulid import ULID as _ULID

    def _monotonic_ulid() -> str:
        return str(_ULID())

except Exception:  # pragma: no cover - the common path in a minimal install
    _monotonic_ulid = _fallback_monotonic_ulid


def new_event_id() -> str:
    """A new monotonic event id.

    Lowercased so ids read comfortably in URLs and JSON, and so that string
    comparison in a case-sensitive database column still yields ULID order
    (the alphabet is uniform in case, so lowercasing preserves ordering).
    """
    return _monotonic_ulid().lower()


def new_id() -> str:
    """A new sortable identifier for sessions, turns and similar records."""
    return _monotonic_ulid().lower()


def new_thread_id() -> str:
    """A new sub-agent thread id.

    Thread ids are not ordering keys, only map keys, so a UUID4 is appropriate
    and makes them visually distinct from event ids in logs.
    """
    return str(uuid.uuid4())
