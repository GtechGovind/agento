"""The capabilities agento ships with.

Each is an ordinary :class:`~agento.core.capabilities.base.Capability` — none of
them is privileged, and each is a working example of how to write your own.

===================================  ==========================================
:class:`CurrentDateTime`             Tells the agent today's date.
:class:`AskUserQuestion`             Structured clarification, answered by the host.
:class:`SubAgents`                   Delegation to focused child agents.
:class:`DeferredTools`               Discover tools instead of preloading them.
:class:`ContextCompaction`           Summarize and replace a long conversation.
:class:`LargeToolResponse`           Offload oversized results to the artifact store.
:class:`Skills`                      Procedures loaded on demand.
:class:`GenerativeUI`                The openui language for rich output.
===================================  ==========================================

The session layer assembles these from an :class:`~agento.session.agent.Agent`'s
configuration; construct them directly only when driving
:class:`~agento.core.runtime.agent_thread.AgentThread` yourself.
"""

from .ask_user_question import AskUserQuestion
from .compaction import ContextCompaction
from .current_datetime import CurrentDateTime
from .deferred_tools import DeferredTools
from .generative_ui import GenerativeUI, render_openui_specification
from .large_tool_response import LargeToolResponse
from .skills import Skills
from .subagents import SubAgents

__all__ = [
    "AskUserQuestion",
    "ContextCompaction",
    "CurrentDateTime",
    "DeferredTools",
    "GenerativeUI",
    "LargeToolResponse",
    "Skills",
    "SubAgents",
    "render_openui_specification",
]
