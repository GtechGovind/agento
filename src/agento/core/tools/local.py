"""Python functions as agent tools.

This is the ergonomic front door of agento. Decorate a function, hand it to an
agent, done::

    from agento import tool

    @tool
    async def get_weather(city: str, units: str = "celsius") -> str:
        \"\"\"Look up the current weather for a city.

        Args:
            city: City name, e.g. "Delhi".
            units: "celsius" or "fahrenheit".
        \"\"\"
        return await weather_api.current(city, units)

    agent = agento.Agent(model="openai/gpt-4o", tools=[get_weather])

Everything the model needs is derived from the function itself:

* **The JSON Schema** comes from the type hints, through pydantic. Anything
  pydantic can validate works as a parameter type — ``str``, ``int``,
  ``list[str]``, ``Literal["a", "b"]``, an ``Enum``, a nested ``BaseModel``.
* **The description** is the docstring's summary, and per-argument descriptions
  come from its ``Args:`` section. Those descriptions matter more than they look
  like they should: they are the model's only guidance on what to pass.
* **Arguments are validated** before your function runs, so a hallucinated
  argument becomes a clear error message the model can correct, not a
  ``TypeError`` in your code.

Marking behaviour turns on the safety features::

    @tool(destructive=True)
    async def delete_project(project_id: str) -> str: ...

Now the default approval policy (``["@write", "@destructive"]``) pauses the run
and asks a human before it ever executes. ``read_only=True`` is the other side of
that: it lets an agent be configured with ``enable_tools=["@read-only"]``.

Sync functions work too — they run on a worker thread so they cannot block the
event loop.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ..messages import ApprovalDecision, InternalToolInfo
from .base import (
    ApprovalRequiredOutcome,
    AuthRequiredOutcome,
    ClientSideRequiredOutcome,
    CreateSubAgentOutcome,
    ToolAnnotations,
    ToolListing,
    ToolListOutcome,
    ToolOutcome,
    ToolSchema,
    ToolSuccess,
    error_result,
)
from .context import ToolContext, current_tool_context

__all__ = ["LocalToolSet", "Tool", "function_tools", "tool"]


# --------------------------------------------------------------------------- #
# Docstring parsing                                                            #
# --------------------------------------------------------------------------- #

_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Example|Examples|Note|Notes)\s*:\s*$",
    re.IGNORECASE,
)
_ARG_RE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into a description and per-argument descriptions.

    Uses ``docstring_parser`` when it is installed (it handles Google, NumPy and
    Sphinx styles); otherwise falls back to the small Google-style parser below,
    which covers the shape almost every Python codebase actually writes.

    Returns:
        ``(description, {argument_name: description})``
    """
    if not doc:
        return "", {}

    try:  # pragma: no cover - optional dependency
        from docstring_parser import parse as _parse  # type: ignore[import-not-found]

        parsed = _parse(doc)
        description = "\n\n".join(
            part for part in (parsed.short_description, parsed.long_description) if part
        ).strip()
        parsed_params = {p.arg_name: (p.description or "").strip() for p in parsed.params if p.arg_name}
        return description, parsed_params
    except Exception:
        pass

    lines = inspect.cleandoc(doc).splitlines()
    description_lines: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current_arg: str | None = None

    for line in lines:
        match = _SECTION_RE.match(line)
        if match:
            section = match.group(1).lower()
            current_arg = None
            continue

        if section in {"args", "arguments", "parameters"}:
            arg_match = _ARG_RE.match(line)
            if arg_match and line[:1] in " \t":
                current_arg = arg_match.group(1).lstrip("*")
                params[current_arg] = arg_match.group(3).strip()
            elif current_arg and line.strip():
                # Continuation line of the previous argument's description.
                params[current_arg] = f"{params[current_arg]} {line.strip()}".strip()
            continue

        if section is None:
            description_lines.append(line)

    return "\n".join(description_lines).strip(), params


# --------------------------------------------------------------------------- #
# Schema generation                                                            #
# --------------------------------------------------------------------------- #

_SKIP_PARAM_NAMES = {"self", "cls"}


def _build_arguments_model(
    func: Callable[..., Any],
    param_docs: dict[str, str],
) -> tuple[type[BaseModel] | None, str | None, list[str]]:
    """Build a pydantic model describing the function's arguments.

    Returns:
        ``(model, context_param_name, variadic_names)``. ``model`` is ``None``
        when the function takes no model-facing arguments.
    """
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        # Unresolvable forward references shouldn't stop a tool from working;
        # untyped parameters simply become free-form.
        hints = {}

    fields: dict[str, tuple[Any, Any]] = {}
    context_param: str | None = None
    variadic: list[str] = []

    for name, param in signature.parameters.items():
        if name in _SKIP_PARAM_NAMES:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            variadic.append(name)
            continue

        annotation = hints.get(name, param.annotation)

        # A ToolContext parameter is injected by the runtime, never by the model.
        if annotation is ToolContext or (
            isinstance(annotation, type) and issubclass(annotation, ToolContext)
        ):
            context_param = name
            continue

        if annotation is inspect.Parameter.empty:
            annotation = Any

        default = ... if param.default is inspect.Parameter.empty else param.default
        description = param_docs.get(name)
        field = Field(default, description=description) if description else Field(default)
        fields[name] = (annotation, field)

    if not fields:
        return None, context_param, variadic

    model = create_model(  # type: ignore[call-overload]
        f"{_pascal(func.__name__)}Arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model, context_param, variadic


def _pascal(name: str) -> str:
    return "".join(part.title() for part in re.split(r"[_\-\s]+", name) if part) or "Tool"


def _json_schema(model: type[BaseModel] | None) -> dict[str, Any]:
    """Render a pydantic model as the JSON Schema the model will see.

    ``mode="serialization"`` is deliberately *not* used: we want the *input*
    view, where a field with a default is optional. Definitions are inlined
    because not every provider resolves ``$ref``.
    """
    if model is None:
        return {"type": "object", "properties": {}}
    schema = model.model_json_schema(ref_template="#/$defs/{model}")
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


# --------------------------------------------------------------------------- #
# Tool                                                                         #
# --------------------------------------------------------------------------- #


class Tool:
    """One callable tool: a schema plus the function behind it.

    Created by the :func:`tool` decorator rather than directly. The instance
    stays callable — ``await get_weather(city="Delhi")`` still works — so
    decorating a function does not take it away from the rest of your code.

    Attributes:
        schema: What the model is told about this tool.
        requires_approval: Explicit override. ``None`` means "let the agent's
            approval policy decide from the annotations", which is the norm.
    """

    __slots__ = (
        "schema",
        "requires_approval",
        "_func",
        "_arguments_model",
        "_context_param",
        "_accepts_kwargs",
        "_is_async",
    )

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        requires_approval: bool | None = None,
    ) -> None:
        if annotations is None and (
            read_only is not None or destructive is not None or idempotent is not None
        ):
            annotations = ToolAnnotations(
                read_only=read_only, destructive=destructive, idempotent=idempotent
            )
        doc_description, param_docs = _parse_docstring(func.__doc__)
        arguments_model, context_param, variadic = _build_arguments_model(func, param_docs)

        self._func = func
        self._arguments_model = arguments_model
        self._context_param = context_param
        self._accepts_kwargs = bool(variadic)
        self._is_async = inspect.iscoroutinefunction(func)
        self.requires_approval = requires_approval
        self.schema = ToolSchema(
            name=name or func.__name__,
            description=(description or doc_description or "").strip(),
            input_schema=input_schema or _json_schema(arguments_model),
            output_schema=output_schema,
            annotations=annotations,
        )

    @property
    def name(self) -> str:
        return self.schema.name

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Tool({self.schema.name!r})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly, bypassing validation."""
        return self._func(*args, **kwargs)

    async def execute(self, arguments: dict[str, Any]) -> ToolOutcome:
        """Validate arguments, run the function, and normalize the result.

        Every failure mode becomes a :class:`ToolSuccess` with ``is_error=True``
        rather than an exception, because the model is usually the one who can
        fix the problem — a validation message like "city: Field required" leads
        to a corrected retry, whereas a raised exception just kills the turn.
        """
        if self._arguments_model is not None:
            try:
                validated = self._arguments_model(**arguments)
            except ValidationError as exc:
                return error_result(
                    json.dumps(
                        {
                            "error": "Invalid arguments",
                            "details": [
                                {
                                    "field": ".".join(str(p) for p in err["loc"]),
                                    "problem": err["msg"],
                                }
                                for err in exc.errors()
                            ],
                        }
                    )
                )
            call_kwargs: dict[str, Any] = {
                key: getattr(validated, key) for key in self._arguments_model.model_fields
            }
        else:
            call_kwargs = dict(arguments) if self._accepts_kwargs else {}

        if self._context_param is not None:
            call_kwargs[self._context_param] = current_tool_context() or ToolContext()

        try:
            if self._is_async:
                result = await self._func(**call_kwargs)
            else:
                # Sync tools run off the event loop so a blocking call in one
                # tool cannot stall streaming or a parallel tool call.
                result = await asyncio.to_thread(self._func, **call_kwargs)
        except Exception as exc:
            return error_result(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

        return _normalize_result(result)


# Outcomes a tool may return directly instead of a value. Returning one of these
# is how a tool asks the loop to do something other than record a result — spawn
# a sub-agent, defer to the host, require authorization.
_PASSTHROUGH_OUTCOMES = (
    ToolSuccess,
    CreateSubAgentOutcome,
    ClientSideRequiredOutcome,
    ApprovalRequiredOutcome,
    AuthRequiredOutcome,
)


def _normalize_result(result: Any) -> ToolOutcome:
    """Turn whatever a tool returned into a tool outcome."""
    if isinstance(result, _PASSTHROUGH_OUTCOMES):
        return result
    if result is None:
        return ToolSuccess(content="")
    if isinstance(result, str):
        return ToolSuccess(content=result)
    if isinstance(result, BaseModel):
        return ToolSuccess(content=result.model_dump_json(), is_structured=True)
    if isinstance(result, (dict, list, tuple)):
        return ToolSuccess(content=json.dumps(result, default=str), is_structured=True)
    return ToolSuccess(content=str(result))


@overload
def tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    requires_approval: bool | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Tool: ...


@overload
def tool(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    requires_approval: bool | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    requires_approval: bool | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Any:
    """Turn a function into an agent tool.

    Works bare or with arguments::

        @tool
        async def ping() -> str: ...

        @tool(destructive=True, name="drop_table")
        async def drop(table: str) -> str: ...

    Args:
        func: The function, when used bare.
        name: Override the tool name. Defaults to the function's name.
        description: Override the description. Defaults to the docstring summary.
        read_only: Mark the tool as having no side effects. Lets it match
            ``@read-only`` selectors.
        destructive: Mark the tool as destructive. Under the default approval
            policy this means the run pauses for a human before it executes.
        idempotent: Mark repeated calls as safe. Informational.
        requires_approval: Force approval on or off for this tool, ignoring the
            agent's policy. Leave as ``None`` unless you mean to override.
        input_schema: Supply the JSON Schema yourself instead of deriving it.
        output_schema: Describe the return shape. Optional but useful — an agent
            can inspect it before deciding how to use the result.

    Returns:
        A :class:`Tool`, which is still callable as the original function.
    """
    annotations: ToolAnnotations | None = None
    if read_only is not None or destructive is not None or idempotent is not None:
        annotations = ToolAnnotations(
            read_only=read_only, destructive=destructive, idempotent=idempotent
        )

    def decorate(target: Callable[..., Any]) -> Tool:
        return Tool(
            target,
            name=name,
            description=description,
            annotations=annotations,
            input_schema=input_schema,
            output_schema=output_schema,
            requires_approval=requires_approval,
        )

    if func is not None:
        return decorate(func)
    return decorate


# --------------------------------------------------------------------------- #
# LocalToolSet                                                                 #
# --------------------------------------------------------------------------- #


class LocalToolSet:
    """A :class:`~agento.core.tools.base.ToolSet` over in-process functions.

    Everything runs in your process, with no network, no policy filtering and no
    approval gate of its own — approval is decided by the agent's policy from the
    tool's annotations, or by ``requires_approval`` on the tool itself.

    Args:
        name: Set name, shown to the model on each tool description.
        tools: :class:`Tool` instances, or plain functions (wrapped for you).
        description: One line about the set, shown when its tools are deferred.
        kind: Reported tool kind. ``"local"`` for host functions; agento's own
            built-ins pass ``"builtin"`` so a UI can tell them apart.
        preload: Whether these tools go into the system prompt up front. Local
            tool sets default to ``True``, because a handful of well-described
            local functions is exactly the case where preloading is right.
    """

    def __init__(
        self,
        name: str,
        tools: Sequence[Tool | Callable[..., Any]],
        *,
        description: str = "",
        kind: str = "local",
        preload: bool = True,
    ) -> None:
        self._name = name
        self._description = description
        self._kind = kind
        self._preload = preload
        self._tools: dict[str, Tool] = {}
        for entry in tools:
            resolved = entry if isinstance(entry, Tool) else Tool(entry)
            self._tools[resolved.name] = resolved

    # -- ToolSet surface ---------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def preload(self) -> bool:
        return self._preload

    @property
    def has_preloaded_tools(self) -> bool:
        return self._preload

    @property
    def tools(self) -> list[Tool]:
        """The tools in this set."""
        return list(self._tools.values())

    def allowed_tool_names(self) -> list[str] | None:
        return list(self._tools)

    def add(self, entry: Tool | Callable[..., Any]) -> None:
        """Add a tool after construction."""
        resolved = entry if isinstance(entry, Tool) else Tool(entry)
        self._tools[resolved.name] = resolved

    async def list_tools(self) -> ToolListOutcome:
        return ToolListing(
            tools=[
                schema.model_copy(update={"preload": self._preload})
                for schema in (t.schema for t in self._tools.values())
            ]
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        approval: ApprovalDecision | None = None,
    ) -> ToolOutcome:
        target = self._tools.get(name)
        if target is None:
            return error_result(json.dumps({"error": f"Unknown tool: {name}"}))
        if approval == "deny":
            return error_result(json.dumps({"error": "User denied this tool call."}))
        return await target.execute(arguments)

    async def tool_info(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        resolve_underlying: bool = False,
    ) -> InternalToolInfo:
        target = self._tools.get(name)
        return InternalToolInfo(
            kind=self._kind,  # type: ignore[arg-type]
            source_id=self._name,
            source_name=self._name,
            original_tool_name=name,
            requires_approval=bool(target.requires_approval) if target else False,
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"LocalToolSet({self._name!r}, {len(self._tools)} tools)"


def function_tools(
    *functions: Tool | Callable[..., Any],
    name: str = "tools",
    description: str = "",
) -> LocalToolSet:
    """Bundle loose functions into one :class:`LocalToolSet`.

    ``Agent(tools=[...])`` does this for you; call it directly when you want to
    name the set or share it between agents.
    """
    return LocalToolSet(name, list(functions), description=description)
