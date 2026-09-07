"""Tracing.

agento emits a span for every meaningful unit of work: the run as a whole, each
sub-agent, each model call, each tool execution, each MCP round trip. By default
those spans go nowhere — :class:`NoopTracer` is a handful of no-op methods that
an optimizer will erase, so tracing costs nothing until you ask for it.

To send spans somewhere, pass any object satisfying :class:`Tracer` to
:class:`~agento.session.agento.Agento`. :class:`OTelTracer` is included and
implements the protocol against ``opentelemetry-api``::

    from agento.tracing import OTelTracer
    app = agento.Agento(llm=..., tracer=OTelTracer())

Writing your own is small — it is two methods — which is the point of keeping
this a protocol rather than a dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

__all__ = ["NoopSpan", "NoopTracer", "OTelTracer", "Span", "Tracer"]


@runtime_checkable
class Span(Protocol):
    """One unit of traced work.

    Implementations must tolerate being finished more than once, and must never
    raise: a tracing failure should not be able to fail an agent run.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key/value to this span."""
        ...

    def set_output(self, output: str) -> None:
        """Record the result of the work this span covers."""
        ...

    def set_error(self, error: BaseException | str) -> None:
        """Mark this span failed."""
        ...

    def end(self) -> None:
        """Finish the span. Idempotent."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Creates spans."""

    def start_span(self, name: str, **attributes: Any) -> Span:
        """Begin a span. The caller is responsible for calling ``end()``.

        Prefer :meth:`span` unless you need to hold a span across an await
        boundary that a ``with`` block cannot express.
        """
        ...

    def span(self, name: str, **attributes: Any) -> Any:
        """Context manager form: begins a span and ends it on exit.

        Records an error automatically if the block raises, then re-raises.
        """
        ...


class NoopSpan:
    """A span that discards everything. The default."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_output(self, output: str) -> None:
        return None

    def set_error(self, error: BaseException | str) -> None:
        return None

    def end(self) -> None:
        return None

    def __enter__(self) -> NoopSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


_NOOP_SPAN = NoopSpan()


class NoopTracer:
    """Tracer that produces no output. Used unless a tracer is configured."""

    __slots__ = ()

    def start_span(self, name: str, **attributes: Any) -> NoopSpan:
        return _NOOP_SPAN

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[NoopSpan]:
        yield _NOOP_SPAN


NOOP_TRACER = NoopTracer()
"""Shared no-op tracer. Cheap to pass around; holds no state."""


class _OTelSpan:
    """Adapts an OpenTelemetry span to the :class:`Span` protocol."""

    __slots__ = ("_span", "_ended")

    def __init__(self, span: Any) -> None:
        self._span = span
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            self._span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
        except Exception:
            pass

    def set_output(self, output: str) -> None:
        self.set_attribute("agento.output", output)

    def set_error(self, error: BaseException | str) -> None:
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            if isinstance(error, BaseException):
                self._span.record_exception(error)
        except Exception:
            pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._span.end()
        except Exception:
            pass


class OTelTracer:
    """Tracer backed by ``opentelemetry-api``.

    Requires the ``otel`` extra. Spans are created on the tracer named
    ``"agento"``; wire up an exporter through the usual OpenTelemetry SDK
    configuration and agento's spans appear alongside the rest of your traces.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Any | None = None) -> None:
        if tracer is None:
            from opentelemetry import trace

            tracer = trace.get_tracer("agento")
        self._tracer = tracer

    def start_span(self, name: str, **attributes: Any) -> _OTelSpan:
        span = self._tracer.start_span(name)
        wrapped = _OTelSpan(span)
        for key, value in attributes.items():
            wrapped.set_attribute(key, value)
        return wrapped

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_OTelSpan]:
        span = self.start_span(name, **attributes)
        try:
            yield span
        except BaseException as exc:
            span.set_error(exc)
            raise
        finally:
            span.end()
