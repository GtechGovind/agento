"""Observable contracts for journal queries and multi-thread coordination."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace as Obj

import pytest
from helpers import build_app, say
from test_subagents import _agent, _assistant_turns, _is_sub

import agento
from agento.core.events import (
    McpAuthRequired,
    McpServerAuth,
    ModelMessage,
    ThreadDone,
    ThreadParent,
    ToolResult,
)
from agento.core.messages import (
    ApprovalRecord,
    EnrichedToolCall,
    InternalToolInfo,
    LLMAssistantMessage,
    LLMToolMessage,
    LLMUserMessage,
    TextPart,
)
from agento.core.runtime import context_utils as journal
from agento.core.runtime.agent_thread import AgentDefinition, AgentThread, ThreadSnapshot
from agento.core.runtime.internal_events import AppendContext, SubAgentCompletion, ThreadFinished
from agento.core.runtime.metrics import ThreadMetrics
from agento.core.runtime.orchestrator import Orchestrator, _merge
from agento.errors import InvalidSendInputError


def call(identifier, **flags):
    return EnrichedToolCall(
        id=identifier, function={"name": "operation", "arguments": '{"value":1}'},
        tool_info=InternalToolInfo(kind="local", original_tool_name="operation", **flags),
    )


def test_journal_queries_use_latest_assistant_and_latest_decision():
    first = LLMAssistantMessage(tool_calls=[call("shared"), call("old")])
    latest = LLMAssistantMessage(tool_calls=[call("shared", requires_approval=True), call("host", is_client_side=True)])
    context = [first, LLMToolMessage(tool_call_id="shared", content="previous result"), latest,
               ApprovalRecord(tool_call_id="shared", decision="deny"),
               LLMUserMessage(content="interleaved note"),
               ApprovalRecord(tool_call_id="shared", decision="allow")]
    assert journal.last_assistant_message(context) is latest
    assert journal.open_tool_call_ids(context) == {"shared", "host"}
    assert journal.scan_approvals(context) == {"shared": "allow"}
    assert journal.pending_approval_calls(context) == []
    assert [item.id for item in journal.pending_client_side_calls(context)] == ["host"]
    assert journal.closable_open_tool_call_ids(context) == set()
    context.append(LLMToolMessage(tool_call_id="host", content="host result"))
    assert journal.closable_open_tool_call_ids(context) == {"shared"}
    context.append(LLMAssistantMessage(content="completed"))
    assert journal.open_tool_call_ids(context) == set()
    assert journal.scan_approvals(context) == {"shared": "allow"}


def test_pending_approval_blocks_repair_but_a_child_call_has_its_own_join():
    context = [LLMAssistantMessage(tool_calls=[
        call("local"), call("approval", requires_approval=True), call("child", creates_subagent=True),
    ])]
    assert [item.id for item in journal.pending_approval_calls(context)] == ["approval"]
    assert not journal.closable_open_tool_call_ids(context)
    context.append(ApprovalRecord(tool_call_id="approval", decision="allow"))
    assert journal.closable_open_tool_call_ids(context) == {"local", "approval"}
    assert journal.open_tool_call_ids(context) == {"local", "approval", "child"}
    assert journal.last_assistant_message([]) is None


def test_internal_notes_and_token_fields_keep_wire_conventions(monkeypatch):
    note = journal.internal_message("runtime fact")
    assert note.content == "<agento-internal>runtime fact</agento-internal>"
    assert journal.is_internal_message(note)
    assert not journal.is_internal_message(LLMUserMessage(content=[TextPart(text="ordinary")]))
    decision = ApprovalRecord(tool_call_id="identifier", decision="allow")
    assert not journal.is_llm_message(decision)
    assert journal.is_llm_message(note)
    fields = []

    def count(value):
        fields.append(value)
        return len(value)

    monkeypatch.setattr(journal, "estimate_tokens", count)
    messages = iter([
        LLMUserMessage(content=[TextPart(text="question")]),
        LLMAssistantMessage(content="answer", tool_calls=[call("call-id")]),
        LLMToolMessage(tool_call_id="call-id", content="result"), decision,
    ])
    usage = journal.estimate_context_usage(messages)
    assert fields == ["question", "answer", "operation", '{"value":1}', "result", "call-id", "identifier"]
    assert usage.prompt_tokens == sum(map(len, fields))
    assert usage.completion_tokens == 0


async def test_merge_preserves_generator_task_context_and_backpressure():
    local = ContextVar("merge_contract", default="outside")
    phases = []
    closed = []

    async def source(name):
        owner = asyncio.current_task()
        token = local.set(name)
        try:
            phases.append((name, 1))
            yield (name, 1)
            assert asyncio.current_task() is owner
            assert local.get() == name
            phases.append((name, 2))
            yield (name, 2)
        finally:
            assert asyncio.current_task() is owner
            local.reset(token)
            closed.append(name)

    merged = _merge([source("one"), source("two")])
    first = await anext(merged)
    await asyncio.sleep(0)
    assert first[1] == 1
    assert sorted(phases) == [("one", 1), ("two", 1)]
    remaining = [item async for item in merged]
    assert sorted([first, *remaining]) == [("one", 1), ("one", 2), ("two", 1), ("two", 2)]
    assert sorted(closed) == ["one", "two"]
    assert local.get() == "outside"


async def test_merge_error_closes_other_blocked_sources():
    started = asyncio.Event()
    closed = []

    async def broken():
        await started.wait()
        raise ValueError("source failed")
        yield  # pragma: no cover

    async def blocked():
        try:
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            closed.append("blocked")

    with pytest.raises(ValueError, match="source failed"):
        _ = [item async for item in _merge([broken(), blocked()])]
    assert closed == ["blocked"]


async def test_merge_propagates_source_cancellation_without_hanging():
    async def cancelled():
        raise asyncio.CancelledError
        yield  # pragma: no cover

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(anext(_merge([cancelled()])), timeout=1)
    assert [item async for item in _merge([])] == []


class JournalThread:
    def __init__(self, identifier, *, parent=None, events=(), reject=False):
        self.thread_id = identifier
        self.parent = parent
        self.context = []
        self.metrics = ThreadMetrics(total_tokens=7)
        self.events = events
        self.reject = reject
        self.received = []
        self.started = False

    def validate_input(self, batch):
        if self.reject and batch:
            raise InvalidSendInputError("invalid test batch")

    async def send(self, batch):
        self.received.extend(batch)
        if batch:
            yield self.thread_id

    async def execute(self, cancel):
        self.started = True
        for event in self.events:
            yield event

    def snapshot(self):
        return tuple(self.context)

    def deliver_tool_message(self, message):
        self.context.append(message)
        return AppendContext(thread_id=self.thread_id, messages=[message])


async def test_send_validates_all_routes_before_any_thread_mutation():
    parent = ThreadParent(thread_id="main", tool_call_id="child-call")
    root = JournalThread("main")
    child = JournalThread("child", parent=parent, reject=True)
    coordinator = Orchestrator({"main": root, "child": child})
    batch = [Obj(thread_id="main"), Obj(thread_id="child")]
    with pytest.raises(InvalidSendInputError, match="invalid test batch"):
        _ = [event async for event in coordinator.send(batch)]
    assert root.received == child.received == []
    with pytest.raises(InvalidSendInputError, match="Unknown thread_id"):
        _ = [event async for event in coordinator.send([Obj(thread_id="missing")])]
    with pytest.raises(InvalidSendInputError, match="sub-agents"):
        _ = [event async for event in coordinator.send([Obj(type="user.message")])]
    child.reject = False
    assert [event async for event in coordinator.send(batch)] == ["main", "child"]
    assert root.received == batch[:1]
    assert child.received == batch[1:]


async def test_paused_fanout_finishes_only_current_batch_and_merges_auth():
    active = 0
    peak = 0
    started = []
    ready = asyncio.Event()
    release = asyncio.Event()
    root = JournalThread("main")
    auth = McpServerAuth(id="shared", name="First", auth_url="https://example.invalid/authorize")

    class WaitingThread(JournalThread):
        async def execute(self, cancel):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.append(self.thread_id)
            if active == 5:
                ready.set()
            try:
                await release.wait()
                yield McpAuthRequired(thread_id=self.thread_id, mcp_servers=[auth])
                yield ("drained", self.thread_id)
            finally:
                active -= 1

    threads = {"main": root}
    for index in range(8):
        name = str(index)
        threads[name] = WaitingThread(name, parent=ThreadParent(thread_id="main", tool_call_id=name))
    coordinator = Orchestrator(threads)

    async def collect():
        return [event async for event in coordinator.execute()]

    work = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
        assert len(started) == peak == 5
        release.set()
        events = await asyncio.wait_for(work, timeout=1)
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
    assert active == 0 and len(started) == 5 and not root.started
    assert len([event for event in events if isinstance(event, tuple)]) == 5
    assert isinstance(events[-1], McpAuthRequired)
    assert events[-1].thread_id is None
    assert events[-1].mcp_servers == [auth]
    assert coordinator.outcome.required_actions == [events[-1]]


async def test_join_checkpoint_precedes_public_result_and_retirement_is_idempotent():
    relation = ThreadParent(thread_id="main", tool_call_id="join")
    root = JournalThread("main")
    root.context = [LLMAssistantMessage(tool_calls=[call("join", creates_subagent=True)])]
    child = JournalThread("child", parent=relation)
    coordinator = Orchestrator({"main": root, "child": child})
    completion = ThreadFinished(
        thread_id="child", parent=relation, output=ModelMessage(thread_id="child", content="summary"),
        send_to_parent=LLMToolMessage(tool_call_id="join", content="summary"),
    )
    stream = coordinator._finish(completion)
    checkpoint = await anext(stream)
    assert isinstance(checkpoint, AppendContext)
    assert root.context[-1].content == "summary"
    assert "child" not in coordinator.snapshots()
    public = await anext(stream)
    assert isinstance(public, ToolResult) and checkpoint.events[0] == public
    done = await anext(stream)
    assert isinstance(done, ThreadDone)
    assert checkpoint.events == [public, done]
    assert list(coordinator.snapshots()) == ["main"]
    assert coordinator.metrics().total_tokens == 14
    assert [event async for event in stream] == []
    replay = [event async for event in coordinator._finish(completion)]
    assert len(replay) == 1 and isinstance(replay[0], ThreadDone)
    assert coordinator.metrics().total_tokens == 14
    detached_metrics = coordinator.metrics()
    detached_metrics.total_tokens = 999
    assert coordinator.metrics().total_tokens == 14


@pytest.mark.parametrize("status", ["done", "error"])
async def test_root_outcome_has_no_child_done_event(status):
    output = ModelMessage(thread_id="main", content="result")
    root = JournalThread("main", events=[ThreadFinished(thread_id="main", status=status, output=output, error="failure")])
    coordinator = Orchestrator({"main": root})
    assert [event async for event in coordinator.execute()] == []
    assert coordinator.outcome.output == (output if status == "done" else None)
    assert coordinator.outcome.error == ("failure" if status == "error" else None)


async def test_already_cancelled_execution_starts_no_threads():
    root = JournalThread("main")
    cancel = asyncio.Event()
    cancel.set()
    assert [event async for event in Orchestrator({"main": root}).execute(cancel)] == []
    assert not root.started


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("child_status", ["done", "error"])
async def test_child_is_not_reexecuted_after_join_stream_closes(backend, child_status, tmp_path):
    store = agento.MemorySessionStore()
    if backend == "sqlite":
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        store = agento.SQLSessionStore(f"sqlite+aiosqlite:///{tmp_path}/joined-child.db")
        await store.create_tables()
    child_calls = []

    def script(request):
        if _is_sub(request):
            child_calls.append("executed")
            return say(error="child provider failed") if child_status == "error" else say("child completed")
        if _assistant_turns(request) == 0:
            return say(tool_calls=[("create_sub_agent", {"name": "worker", "input": "work"})])
        return say("parent handled the failure")

    try:
        app, _ = build_app([script] * 10, store=store)
        session = await app.sessions.create(agent=_agent())
        turn = await session.create_turn("start")
        stream = turn.stream()
        async for event in stream:
            if isinstance(event, ToolResult) and event.thread_id == "main":
                assert event.content == ("child provider failed" if child_status == "error" else "child completed")
                record = await store.get_turn(session.id, turn.id)
                assert list(record.snapshot.threads) == ["main"]
                saved = (await turn.list_events()).items
                child_done = [item for item in saved if isinstance(item, ThreadDone)]
                assert len(child_done) == 1
                assert child_done[0].state.status == child_status
                assert saved.index(event) < saved.index(child_done[0])
                break
        await stream.aclose()
        assert child_calls == ["executed"]
        resumed = await session.create_turn()
        await resumed.drain()
        assert child_calls == ["executed"]
        assert resumed.state.status == "done"
        assert resumed.state.output.content == "parent handled the failure"
        assert list(resumed.record.snapshot.threads) == ["main"]
    finally:
        if backend == "sqlite":
            await store.dispose()


@pytest.mark.parametrize("historic_status", [None, "done", "error"])
async def test_legacy_answered_child_snapshot_retires_without_invoking_child(historic_status):
    relation = ThreadParent(thread_id="main", tool_call_id="historic-join")
    reply = LLMToolMessage(tool_call_id="historic-join", content="historic reply")
    completion = None
    if historic_status is not None:
        completion = SubAgentCompletion(
            status=historic_status,
            output=ModelMessage(thread_id="child", content="retained output"),
            error="retained error" if historic_status == "error" else None,
            send_to_parent=reply,
        )
    stored_child = ThreadSnapshot(
        thread_id="child", parent=relation,
        context=[LLMUserMessage(content="historic task")], completion=completion,
    )
    restored = ThreadSnapshot.model_validate_json(stored_child.model_dump_json())
    child_llm = agento.ScriptedLLM([say(error="child must not be invoked")])
    child = AgentThread(
        AgentDefinition(llm=child_llm), thread_id=restored.thread_id,
        context=restored.context, parent=restored.parent, completion=restored.completion,
    )
    parent = AgentThread(
        AgentDefinition(llm=agento.ScriptedLLM([say("parent resumed")])),
        context=[LLMAssistantMessage(tool_calls=[call("historic-join", creates_subagent=True)]), reply],
    )
    parent.metrics.total_tokens = 5
    child.metrics.total_tokens = 7
    coordinator = Orchestrator({"main": parent, "child": child})
    stream = coordinator.execute()
    first = await anext(stream)
    assert list(coordinator.snapshots()) == ["main"]
    assert coordinator.metrics().total_tokens == 12
    if historic_status is None:
        assert isinstance(first, AppendContext)
        assert first.messages == first.events == []
    else:
        assert isinstance(first, ThreadDone)
        assert first.state.status == historic_status
        assert first.state.output.content == "retained output"
        if historic_status == "error":
            assert first.state.error == "retained error"
    remaining = [event async for event in stream]
    assert child_llm.requests == []
    assert coordinator.outcome.output.content == "parent resumed"
    assert not any(isinstance(event, (ToolResult, ThreadDone)) for event in remaining)
    assert [message for message in parent.context if isinstance(message, LLMToolMessage)] == [reply]
