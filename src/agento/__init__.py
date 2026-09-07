"""agento — an embeddable agent harness for Python.

agento runs the agent loop: model calls, tool execution, human approvals, context
management, sub-agents, and durable session state. It is a library, not a server —
your application calls it directly and gets an async stream of events back.

Start here::

    import agento

    app = agento.Agento(llm=agento.LiteLLMProvider())
    agent = agento.Agent(model="openai/gpt-4o", instructions="Be concise.")

    print(await app.run(agent, "What is an agent harness?"))

Add a tool::

    @agento.tool
    async def get_weather(city: str) -> str:
        \"\"\"Look up the current weather.

        Args:
            city: City name, e.g. "Delhi".
        \"\"\"
        return await weather.current(city)

    agent = agento.Agent(model="openai/gpt-4o", tools=[get_weather])

Hold a conversation and watch it happen::

    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn("what's the weather in Delhi?")

    async for event in turn.stream():
        if isinstance(event, agento.ModelMessageDelta) and event.content:
            print(event.content, end="", flush=True)

The map of the package:

===============================  ============================================
:class:`Agento`                  Shared configuration; construct once.
:class:`Agent`                   A declarative agent definition.
:func:`tool`                     Turn a function into a tool.
``app.sessions``                 Create and load conversations.
``session.create_turn()``        Prepare a turn.
``turn.stream()``                Run it, and watch.
===============================  ============================================

Extending it: :class:`~agento.core.capabilities.base.Capability` hooks the loop,
:class:`~agento.core.llm.base.LLM` is the model interface,
:class:`~agento.core.tools.base.ToolSet` the tool interface,
:class:`~agento.session.store.base.SessionStore` the persistence interface, and
:class:`~agento.artifacts.base.ArtifactStore` and
:class:`~agento.skills.base.SkillSource` cover large content and procedures.
Everything agento ships is written against those same interfaces.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# -- errors ----------------------------------------------------------------- #
from .artifacts.base import Artifact, ArtifactStore
from .artifacts.local import LocalArtifactStore
from .artifacts.memory import MemoryArtifactStore

# -- capabilities ----------------------------------------------------------- #
from .core.capabilities.base import (
    AppendContext,
    Capability,
    ContextUsage,
    EmitEvent,
    ExecutionContext,
    ReplaceContext,
    SetState,
)
from .core.capabilities.builtins import (
    AskUserQuestion,
    ContextCompaction,
    CurrentDateTime,
    DeferredTools,
    GenerativeUI,
    LargeToolResponse,
    Skills,
    SubAgents,
)

# -- events (the public contract) ------------------------------------------- #
from .core.events import (
    ActionRequired,
    AgentInfo,
    ApprovalRequired,
    ArtifactCreated,
    ClientToolRequired,
    ContextCompacted,
    Event,
    McpAuthRequired,
    McpInitialized,
    ModelMessage,
    ModelMessageDelta,
    StreamEvent,
    ThreadCreated,
    ThreadDone,
    ToolApproval,
    ToolReply,
    ToolResult,
    TurnCreated,
    TurnDone,
    TurnInput,
    TurnMetrics,
    TurnState,
    TurnStateCancelled,
    TurnStateDone,
    TurnStateError,
    TurnStateRunning,
    UserMessage,
)

# -- messages --------------------------------------------------------------- #
from .core.llm.base import LLM, BaseLLM, LLMRequest, LLMResponse, ModelProperties, StreamChunk

# -- models ----------------------------------------------------------------- #
from .core.llm.scripted import ScriptedLLM, ScriptedResponse, say
from .core.messages import FilePart, ImagePart, TextPart, ToolInfo, Usage

# -- runtime (for advanced use) --------------------------------------------- #
from .core.runtime.agent_thread import AgentDefinition, AgentThread
from .core.runtime.orchestrator import Orchestrator

# -- tools ------------------------------------------------------------------ #
from .core.tools.base import ToolSchema, ToolSet, ToolSource, ToolSuccess, error_result, text_result
from .core.tools.client_side import ClientSideToolSet
from .core.tools.context import ToolContext, current_tool_context
from .core.tools.local import LocalToolSet, Tool, function_tools, tool
from .core.tools.policy import PolicyToolSet, ToolSelectors
from .errors import (
    AgentoError,
    ConfigurationError,
    InvalidSendInputError,
    McpConnectionError,
    SessionNotFoundError,
    TurnNotFoundError,
)

# -- session ---------------------------------------------------------------- #
from .session.agent import (
    Agent,
    CompactionConfig,
    GenerativeUIConfig,
    LargeToolResponseConfig,
    MCPServerConfig,
    MCPServerRef,
    RuntimeConfig,
    SubAgentConfig,
)
from .session.agento import Agento
from .session.resolver import ResourceResolver
from .session.session_handle import SessionHandle
from .session.sessions import Sessions
from .session.store.base import (
    Page,
    SessionRecord,
    SessionStore,
    TurnRecord,
    TurnSnapshot,
)
from .session.store.memory import MemorySessionStore
from .session.turn_handle import TurnHandle

# -- skills ----------------------------------------------------------------- #
from .skills.base import Skill, SkillSource
from .skills.filesystem import FileSkillSource

# -- tracing ---------------------------------------------------------------- #
from .tracing import NoopTracer, Span, Tracer

__all__ = [
    # runtime facade
    "Agent",
    "Agento",
    "Sessions",
    "SessionHandle",
    "TurnHandle",
    "ResourceResolver",
    # agent configuration
    "CompactionConfig",
    "GenerativeUIConfig",
    "LargeToolResponseConfig",
    "MCPServerConfig",
    "MCPServerRef",
    "RuntimeConfig",
    "SubAgentConfig",
    # tools
    "ClientSideToolSet",
    "LocalToolSet",
    "PolicyToolSet",
    "Tool",
    "ToolContext",
    "ToolSchema",
    "ToolSelectors",
    "ToolSet",
    "ToolSource",
    "ToolSuccess",
    "current_tool_context",
    "error_result",
    "function_tools",
    "text_result",
    "tool",
    # models
    "BaseLLM",
    "LLM",
    "LLMRequest",
    "LLMResponse",
    "LiteLLMClient",
    "LiteLLMProvider",
    "ModelProperties",
    "OpenAIClient",
    "OpenAIProvider",
    "ScriptedLLM",
    "ScriptedResponse",
    "StreamChunk",
    "say",
    # events and inputs
    "ActionRequired",
    "AgentInfo",
    "ApprovalRequired",
    "ArtifactCreated",
    "ClientToolRequired",
    "ContextCompacted",
    "Event",
    "FilePart",
    "ImagePart",
    "McpAuthRequired",
    "McpInitialized",
    "ModelMessage",
    "ModelMessageDelta",
    "StreamEvent",
    "TextPart",
    "ThreadCreated",
    "ThreadDone",
    "ToolApproval",
    "ToolInfo",
    "ToolReply",
    "ToolResult",
    "TurnCreated",
    "TurnDone",
    "TurnInput",
    "TurnMetrics",
    "TurnState",
    "TurnStateCancelled",
    "TurnStateDone",
    "TurnStateError",
    "TurnStateRunning",
    "Usage",
    "UserMessage",
    # capabilities
    "AppendContext",
    "AskUserQuestion",
    "Capability",
    "ContextCompaction",
    "ContextUsage",
    "CurrentDateTime",
    "DeferredTools",
    "EmitEvent",
    "ExecutionContext",
    "GenerativeUI",
    "LargeToolResponse",
    "ReplaceContext",
    "SetState",
    "Skills",
    "SubAgents",
    # storage
    "MemorySessionStore",
    "Page",
    "SQLSessionStore",
    "SessionRecord",
    "SessionStore",
    "TurnRecord",
    "TurnSnapshot",
    # artifacts and skills
    "Artifact",
    "ArtifactStore",
    "FileSkillSource",
    "LocalArtifactStore",
    "MemoryArtifactStore",
    "Skill",
    "SkillSource",
    # advanced runtime
    "AgentDefinition",
    "AgentThread",
    "Orchestrator",
    # tracing and errors
    "AgentoError",
    "ConfigurationError",
    "InvalidSendInputError",
    "McpConnectionError",
    "NoopTracer",
    "SessionNotFoundError",
    "Span",
    "Tracer",
    "TurnNotFoundError",
    "__version__",
]

# Adapters that need an optional dependency are imported on first use, so a
# minimal install stays minimal and a missing package produces a clear message
# naming the extra to install rather than an ImportError at import time.
_LAZY = {
    "LiteLLMClient": "agento.core.llm.litellm_client",
    "LiteLLMProvider": "agento.core.llm.litellm_client",
    "OpenAIClient": "agento.core.llm.openai_client",
    "OpenAIProvider": "agento.core.llm.openai_client",
    "RemoteMCP": "agento.core.tools.remote_mcp",
    "SQLSessionStore": "agento.session.store.sql",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*__all__, *_LAZY})
