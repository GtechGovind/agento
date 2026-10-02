"""Observable contracts for the conversation runner and capability boundaries."""

from __future__ import annotations

import asyncio

import pytest
from helpers import build_agent, build_app

import agento
from agento.core.capabilities.base import AppendContext as AddMessages
from agento.core.capabilities.base import ContextUsage, EmitEvent
from agento.core.capabilities.base import ReplaceContext as ReplaceMessages
from agento.core.capabilities.base import SetState as SaveValue
from agento.core.events import McpServerAuth, ThreadParent
from agento.core.llm.base import StreamChunk
from agento.core.messages import LLMAssistantMessage, LLMUserMessage
from agento.core.runtime.agent_thread import AgentDefinition, AgentThread
from agento.core.runtime.internal_events import AppendContext, ReplaceContext, SetState, ThreadFinished
from agento.core.tools.base import AuthRequiredOutcome
from agento.core.tools.local import LocalToolSet
from agento.errors import CapabilityStateError
from agento.session.store.base import SessionRecord, TurnRecord, TurnSnapshot


def conversation(script=(), *, capabilities=(), tools=(), **kwargs):
    llm = agento.ScriptedLLM(list(script))
    definition = AgentDefinition(llm=llm, tool_sets=[LocalToolSet("host", tools)])
    return AgentThread(definition, capabilities=capabilities, **kwargs), llm


async def drain(stream):
    return [item async for item in stream]


async def test_thread_rejects_overlapping_consumers_and_releases_ownership_on_close():
    thread, _ = conversation([agento.say("answer")])
    active = thread.execute()
    assert isinstance(await anext(active), agento.ModelMessage)
    with pytest.raises(RuntimeError, match="already running"):
        await anext(thread.execute())
    with pytest.raises(RuntimeError, match="already running"):
        await anext(thread.send([agento.UserMessage(content="overlap")]))
    await active.aclose()
    completed = await drain(thread.execute())
    assert completed[-1].status == "done"
    assert completed[-1].output.content == "answer"


@pytest.mark.parametrize("stop", ["close", "task_cancel", "cooperative"])
async def test_interrupted_provider_is_closed_and_partial_output_is_not_checkpointed(stop):
    closed = []
    waiting = asyncio.Event()

    class Provider:
        async def stream(self, request):
            try:
                yield StreamChunk(content="unfinished")
                if stop == "task_cancel":
                    waiting.set()
                    await asyncio.Future()
                yield StreamChunk(finish_reason="stop")
            finally:
                closed.append(True)

    thread = AgentThread(AgentDefinition(llm=Provider()))
    cancellation = asyncio.Event()
    stream = thread.execute(cancellation)
    assert isinstance(await anext(stream), agento.ModelMessage)
    assert isinstance(await anext(stream), agento.ModelMessageDelta)
    if stop == "close":
        await stream.aclose()
    elif stop == "cooperative":
        cancellation.set()
        remaining = await drain(stream)
        assert not any(isinstance(item, AppendContext) for item in remaining)
    else:
        consumer = asyncio.create_task(anext(stream))
        await asyncio.wait_for(waiting.wait(), 1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    assert closed == [True]
    assert thread.snapshot().context == []
    # Even task cancellation relinquishes the conversation's ownership.
    checkpoints = await drain(thread.send([agento.UserMessage(content="resume")]))
    assert checkpoints[-1].messages[-1].content == "resume"


async def test_committed_approval_request_remains_visible_after_cooperative_cancellation():
    executed = []

    @agento.tool(requires_approval=True)
    async def change_record() -> str:
        executed.append(True)
        return "changed"

    thread, _ = conversation([agento.say(tool_calls=["change_record"])], tools=[change_record])
    cancellation = asyncio.Event()
    seen = []
    async for event in thread.execute(cancellation):
        seen.append(event)
        if isinstance(event, AppendContext) and event.events:
            cancellation.set()
    message = next(event for event in seen if isinstance(event, agento.ModelMessage) and event.tool_calls)
    approval = next(event for event in seen if isinstance(event, agento.ApprovalRequired))
    assert approval.tool_calls[0].source_event_id == message.id
    assert executed == []
    assert thread.is_awaiting_user_input()


async def test_capability_hooks_checkpoint_state_and_leave_ephemeral_edits_out_of_history():
    observed = []

    class Changes(agento.Capability):
        state_key = "counter"

        def load_state(self, value):
            observed.append(("loaded", value))

        async def pre_send(self, context):
            observed.append(("pre_send", len(context.context)))
            yield AddMessages(messages=[LLMUserMessage(content="prepared")])

        async def pre_llm(self, context):
            yield SaveValue(key=self.state_key, value=8)
            yield ReplaceMessages(messages=[LLMUserMessage(content="revised")], usage=ContextUsage(prompt_tokens=2))

        def prepare_request(self, messages):
            return [*messages, {"role": "user", "content": "temporary"}]

    class Observer(agento.Capability):
        async def pre_llm(self, context):
            observed.append(("history", context.context[-1].content))
            if False:
                yield

    thread, llm = conversation([agento.say("done")], capabilities=[Changes(), Observer()],
                              capability_state={"counter": 7, "orphan": 3})
    await drain(thread.send([agento.UserMessage(content="start")]))
    checkpoint_types = []
    async for item in thread.execute():
        if isinstance(item, SetState):
            assert thread.snapshot().capability_state == {"counter": 8}
        if isinstance(item, ReplaceContext):
            assert thread.context[-1].content == "revised"
        checkpoint_types.append(type(item))
    assert observed == [("loaded", 7), ("pre_send", 0), ("history", "revised")]
    assert SetState in checkpoint_types and ReplaceContext in checkpoint_types
    assert llm.requests[0].messages[-1]["content"] == "temporary"
    assert all(message.content != "temporary" for message in thread.context)


def test_duplicate_capability_state_owners_fail_before_loading_state():
    class Owner(agento.Capability):
        state_key = "same"

        def load_state(self, value):
            raise AssertionError("ambiguous state must not be delivered")

    with pytest.raises(CapabilityStateError, match="Duplicate"):
        conversation(capabilities=[Owner(), Owner()], capability_state={"same": 3})


async def test_undeclared_capability_write_is_a_terminal_error_without_state_mutation():
    class Invalid(agento.Capability):
        async def pre_llm(self, context):
            yield SaveValue(key="unclaimed", value=9)

    thread, llm = conversation([agento.say("unused")], capabilities=[Invalid()])
    events = await drain(thread.execute())
    assert isinstance(events[-1], ThreadFinished)
    assert events[-1].status == "error" and "Undeclared" in events[-1].error
    assert thread.snapshot().capability_state == {}
    assert llm.requests == []


@pytest.mark.parametrize("hook", ["pre_llm", "process_tool_results"])
async def test_capability_cannot_overwrite_another_capability_state(hook):
    restored = []

    class First(agento.Capability):
        state_key = "first"

        async def pre_llm(self, context):
            if hook == "pre_llm":
                yield SaveValue(key="second", value="corrupted")

        async def process_tool_results(self, results, context):
            return [SaveValue(key="second", value="corrupted")] if hook == "process_tool_results" else []

    class Second(agento.Capability):
        state_key = "second"

        def load_state(self, value):
            restored.append(value)

    @agento.tool(read_only=True)
    async def inspect() -> str:
        return "observed"

    original = {"first": 1, "second": "retained"}
    thread, _ = conversation([agento.say(tool_calls=["inspect"])], tools=[inspect],
                             capabilities=[First(), Second()], capability_state=original)
    events = await drain(thread.execute())
    assert events[-1].status == "error"
    assert "'first' cannot update 'second'" in events[-1].error
    assert thread.snapshot().capability_state == original
    assert restored == ["retained"]
    assert not any(isinstance(event, SetState) for event in events)


@pytest.mark.parametrize("invalid_rewrite", [False, True])
async def test_tool_rewrites_publish_checkpoint_before_events_and_reject_removed_results(invalid_rewrite):
    side_effects = []
    notice = agento.ArtifactCreated(artifact_id="a", name="result.txt", mime_type="text/plain", size_bytes=3)

    @agento.tool(requires_approval=False)
    async def perform() -> str:
        side_effects.append("executed")
        return "raw result"

    class Rewrite(agento.Capability):
        async def process_tool_results(self, results, context):
            if invalid_rewrite:
                results.clear()
            else:
                results[0].message.content = "rewritten result"
                results[0].artifact_id = "a"
            return [EmitEvent(event=notice)]

    thread, llm = conversation([agento.say(tool_calls=["perform"]), agento.say("done")],
                              capabilities=[Rewrite()], tools=[perform])
    events = await drain(thread.execute())
    assert side_effects == ["executed"]
    if invalid_rewrite:
        assert events[-1].status == "error"
        assert not any(isinstance(event, agento.ToolResult) for event in events)
        assert len(llm.requests) == 1
    else:
        result = next(event for event in events if isinstance(event, agento.ToolResult))
        checkpoint = next(event for event in events if isinstance(event, AppendContext) and result in event.events)
        assert events.index(checkpoint) < events.index(notice) < events.index(result)
        assert result.artifact_id == "a" and result.content == "rewritten result"
        assert llm.requests[1].messages[-1]["content"] == "rewritten result"


async def test_child_completion_is_replayable_before_its_public_message_is_consumed():
    parent = ThreadParent(thread_id="parent", tool_call_id="spawn")
    thread, _ = conversation([agento.say("child answer")], thread_id="child", parent=parent)
    stream = thread.execute()
    async for event in stream:
        if isinstance(event, AppendContext) and event.completion is not None:
            saved = thread.snapshot()
            break
    await stream.aclose()
    replay, llm = conversation(thread_id="child", parent=parent, context=saved.context,
                               completion=saved.completion)
    events = await drain(replay.execute())
    assert len(events) == 1 and isinstance(events[0], ThreadFinished)
    assert events[0].send_to_parent.content == "child answer"
    assert llm.requests == []


async def test_initial_auth_requirement_prevents_a_model_call():
    requirement = McpServerAuth(id="private", name="private", auth_url="https://example.invalid/auth")

    class PrivateTools(LocalToolSet):
        async def list_tools(self):
            return AuthRequiredOutcome(servers=[requirement])

    llm = agento.ScriptedLLM([agento.say("unused")])
    thread = AgentThread(AgentDefinition(llm=llm, tool_sets=[PrivateTools("private", [])]))
    events = await drain(thread.execute())
    assert len(events) == 1 and isinstance(events[0], agento.McpAuthRequired)
    assert events[0].mcp_servers == [requirement]
    assert llm.requests == []


def test_validation_does_not_modify_context_after_invalid_or_empty_user_input():
    thread, _ = conversation(context=[LLMAssistantMessage(content="previous")])
    before = thread.snapshot()
    for content in ("", " \n", []):
        with pytest.raises(agento.InvalidSendInputError, match="empty"):
            thread.validate_input([agento.UserMessage(content=content)])
    with pytest.raises(agento.InvalidSendInputError, match="no open tool"):
        thread.validate_input([agento.ToolReply(thread_id="main", tool_call_id="missing", content="result")])
    assert thread.snapshot() == before


@pytest.mark.parametrize("hook", ["pre_send", "pre_llm", "post_tool_call", "process_tool_results"])
async def test_appended_hook_event_is_persisted_before_live_delivery_exactly_once(hook):
    notice = agento.ArtifactCreated(artifact_id="notice", name="note.txt", mime_type="text/plain", size_bytes=1)

    class AttachedNotice(agento.Capability):
        issued = False

        def commands(self, phase):
            if phase != hook or self.issued:
                return []
            self.issued = True
            return [AddMessages(events=[notice])]

        async def pre_send(self, context):
            for command in self.commands("pre_send"):
                yield command

        async def pre_llm(self, context):
            for command in self.commands("pre_llm"):
                yield command

        async def post_tool_call(self, context):
            for command in self.commands("post_tool_call"):
                yield command

        async def process_tool_results(self, results, context):
            return self.commands("process_tool_results")

    @agento.tool(read_only=True)
    async def inspect() -> str:
        return "observed"

    app, _ = build_app([agento.say(tool_calls=["inspect"]), agento.say("done")],
                      capabilities=[AttachedNotice()])
    session = await app.sessions.create(agent=build_agent(tools=[inspect]))
    turn = await session.create_turn("start")
    delivered = []
    async for event in turn.stream():
        delivered.append(event.id)
        if event.id == notice.id:
            assert [saved.id for saved in (await turn.list_events()).items].count(notice.id) == 1
    assert delivered.count(notice.id) == 1
    assert [saved.id for saved in (await turn.list_events()).items].count(notice.id) == 1


@pytest.mark.parametrize("hook", ["pre_llm", "post_tool_call"])
async def test_closing_after_attached_event_checkpoint_leaves_event_recoverable(hook):
    notice = agento.ArtifactCreated(artifact_id="notice", name="note.txt", mime_type="text/plain", size_bytes=1)

    class AttachedNotice(agento.Capability):
        async def pre_llm(self, context):
            if hook == "pre_llm":
                yield AddMessages(messages=[LLMUserMessage(content="checkpointed note")], events=[notice])

        async def post_tool_call(self, context):
            if hook == "post_tool_call":
                yield AddMessages(messages=[LLMUserMessage(content="checkpointed note")], events=[notice])

    @agento.tool(read_only=True)
    async def inspect() -> str:
        return "observed"

    thread, _ = conversation([agento.say(tool_calls=["inspect"]), agento.say("unused")],
                             tools=[inspect], capabilities=[AttachedNotice()])
    store = agento.MemorySessionStore()
    await store.create_session(SessionRecord(session_id="s"))
    await store.create_turn(TurnRecord(session_id="s", turn_id="t"))
    stream = thread.execute()
    delivered = []
    async for event in stream:
        if isinstance(event, AppendContext) and notice in event.events:
            # A checkpoint consumer can stop before requesting the public event.
            await store.update_turn("s", "t", events=event.events,
                                    snapshot=TurnSnapshot(threads={"main": thread.snapshot().model_dump()}))
            break
        delivered.append(event)
    else:
        pytest.fail("Hook did not produce its checkpoint")
    await stream.aclose()
    assert notice not in delivered
    stored = await store.list_turn_events("s", "t")
    assert [event.id for event in stored.items] == [notice.id]
    record = await store.get_turn("s", "t")
    assert record.snapshot.threads["main"]["context"][-1]["content"] == "checkpointed note"
