"""Regression tests at publication, cancellation and transaction boundaries."""
from __future__ import annotations

import asyncio
import base64
import os
import uuid

import pytest
from helpers import build_agent, build_app, say

import agento
from agento.core.capabilities.base import SetState
from agento.core.runtime.agent_thread import ThreadSnapshot
from agento.core.runtime.context_utils import open_tool_call_ids
from agento.errors import (
    ConfigurationError,
    PreviousTurnRunningError,
    SessionStoreConflictError,
    TurnNotRunningError,
)
from agento.session.store.base import SessionRecord, TurnRecord, TurnSnapshot


@pytest.fixture(params=['memory', 'sqlite'] + (['postgres'] if os.getenv('AGENTO_TEST_POSTGRES_URL') else []))
async def store(request, tmp_path):
    if request.param == 'memory':
        yield agento.MemorySessionStore()
    else:
        pytest.importorskip('sqlalchemy')
        if request.param == 'sqlite':
            pytest.importorskip('aiosqlite')
            value = agento.SQLSessionStore(f'sqlite+aiosqlite:///{tmp_path}/state.db')
        else:
            pytest.importorskip('asyncpg')
            value = agento.SQLSessionStore(os.environ['AGENTO_TEST_POSTGRES_URL'],
                                          table_prefix=f'test_{uuid.uuid4().hex[:12]}_')
        await value.create_tables()
        try:
            yield value
        finally:
            if request.param == "postgres":
                await value.drop_tables()
            await value.dispose()


async def test_every_complete_event_is_durable_at_yield(store):
    @agento.tool(read_only=True)
    async def lookup() -> str:
        return 'found'

    app, _ = build_app([say(tool_calls=[('lookup', {})]), say('answer')], store=store)
    session = await app.sessions.create(agent=build_agent(tools=[lookup]))
    turn = await session.create_turn('go')
    async for event in turn.stream():
        if isinstance(event, agento.ModelMessage) and not (event.content or event.tool_calls):
            continue  # transient opening placeholder
        if isinstance(event, agento.ModelMessageDelta):
            continue
        assert event.id in {e.id for e in (await turn.list_events()).items}
        record = await store.get_turn(session.id, turn.id)
        if isinstance(event, agento.ModelMessage):
            assert ThreadSnapshot.model_validate(record.snapshot.threads['main']).context[-1].role == 'assistant'
        if isinstance(event, agento.ToolResult):
            assert not open_tool_call_ids(ThreadSnapshot.model_validate(record.snapshot.threads['main']).context)
    assert turn.state.status == 'done'


@pytest.mark.parametrize('boundary', ['created', 'delta', 'message', 'result'])
async def test_closing_stream_finalizes_without_replaying_completed_tool(store, boundary):
    effects = []

    @agento.tool(requires_approval=False)
    async def action() -> str:
        effects.append('done')
        return 'action completed'

    app, llm = build_app([say(tool_calls=[('action', {})]), say('continued')], store=store)
    session = await app.sessions.create(agent=build_agent(tools=[action]))
    turn = await session.create_turn('go')
    stream = turn.stream()
    async for event in stream:
        if (boundary == 'created' and isinstance(event, agento.TurnCreated)
            or boundary == 'delta' and isinstance(event, agento.ModelMessageDelta)
            or boundary == 'message' and isinstance(event, agento.ModelMessage) and event.tool_calls
            or boundary == 'result' and isinstance(event, agento.ToolResult)):
            break
    await stream.aclose()
    assert turn.state.status == 'cancelled'
    assert (await store.get_turn(session.id, turn.id)).state.status == 'cancelled'
    if boundary == 'result':
        resumed = await session.create_turn('continue')
        await resumed.drain()
        assert effects == ['done']
        tool_messages = [m['content'] for m in llm.requests[-1].messages if m['role'] == 'tool']
        assert tool_messages == ['action completed']


async def test_state_and_compaction_checkpoint_contains_new_state(store):
    class Counter(agento.Capability):
        state_key = 'counter'

        async def pre_llm(self, context):
            yield SetState(key='counter', value=7)

    app, _ = build_app([say('first'), say('summary'), say('second')], store=store, capabilities=[Counter()])
    agent = build_agent()
    agent.config.compaction = agento.CompactionConfig(threshold_tokens=1)
    session = await app.sessions.create(agent=agent)
    await session.run('first question')
    turn = await session.create_turn('second question')
    compacted = []
    async for event in turn.stream():
        record = await store.get_turn(session.id, turn.id)
        if isinstance(event, agento.ContextCompacted):
            compacted.append(event.id)
            assert any('summary' in str(m.content) for m in ThreadSnapshot.model_validate(record.snapshot.threads['main']).context)
        if isinstance(event, agento.ModelMessageDelta):
            assert ThreadSnapshot.model_validate(record.snapshot.threads['main']).capability_state['counter'] == 7
    assert turn.state.status == 'done'
    assert len(compacted) == 1
    ids = [e.id for e in (await turn.list_events()).items]
    assert ids.count(compacted[0]) == 1


async def test_terminal_writes_do_not_change_outcome_or_double_count(store):
    await store.create_session(SessionRecord(session_id='s'))
    await store.create_turn(TurnRecord(session_id='s', turn_id='t'))
    state = agento.TurnStateDone(metrics=agento.TurnMetrics(total_tokens=10))
    await store.update_turn('s', 't', state=state)
    for patch in ({'state': state}, {'custom': {'late': True}}, {'snapshot': TurnSnapshot()}):
        with pytest.raises(TurnNotRunningError):
            await store.update_turn('s', 't', **patch)
    assert (await store.get_session('s')).metrics.total_tokens == 10
    assert (await store.get_turn('s', 't')).custom == {}


async def test_snapshot_and_event_roll_back_together(store):
    await store.create_session(SessionRecord(session_id='s'))
    await store.create_turn(TurnRecord(session_id='s', turn_id='t'))
    event = agento.ToolResult(tool_call_id='x', content='ok')
    await store.append_events('s', 't', [event])
    error_type = agento.errors.SessionStoreInvariantError
    if isinstance(store, agento.SQLSessionStore):
        from sqlalchemy.exc import IntegrityError

        error_type = IntegrityError
    with pytest.raises(error_type):
        await store.update_turn('s', 't', state=agento.TurnStateDone(
            metrics=agento.TurnMetrics(total_tokens=17)), custom={'bad': True}, events=[event])
    turn = await store.get_turn('s', 't')
    assert turn.state.status == 'running'
    assert turn.custom == {}
    assert (await store.get_session('s')).metrics.total_tokens == 0
    assert len((await store.list_turn_events('s', 't')).items) == 1


async def test_concurrent_terminal_writes_have_one_winner(store):
    await store.create_session(SessionRecord(session_id='s'))
    await store.create_turn(TurnRecord(session_id='s', turn_id='t'))
    outcomes = await asyncio.gather(*[
        store.update_turn('s', 't', state=agento.TurnStateDone(
            metrics=agento.TurnMetrics(total_tokens=11)), events=[agento.TurnDone(state=agento.TurnStateDone())])
        for _ in range(4)
    ], return_exceptions=True)
    assert sum(value is None for value in outcomes) == 1
    assert all(value is None or isinstance(value, TurnNotRunningError) for value in outcomes)
    assert (await store.get_session('s')).metrics.total_tokens == 11
    assert len((await store.list_turn_events('s', 't')).items) == 1


async def test_stale_handles_refresh_and_active_turns_are_not_cancelled(store):
    app, llm = build_app([say('remembered'), say('yes')], store=store)
    session = await app.sessions.create(agent=build_agent())
    stale = await app.sessions.get(session.id)
    first = await session.create_turn('remember alpha')
    with pytest.raises(PreviousTurnRunningError):
        await stale.create_turn('interrupt')
    with pytest.raises(PreviousTurnRunningError):
        await stale.create_turn('fresh', previous_turn_id='none')
    assert (await store.get_turn(session.id, first.id)).state.status == 'running'
    await first.drain()
    second = await stale.create_turn('continue')
    assert second.previous_turn_id == first.id
    await second.drain()
    assert any('alpha' in str(m.get('content')) for m in llm.requests[-1].messages)


async def test_compare_tip_prevents_two_simultaneous_creates(store):
    await store.create_session(SessionRecord(session_id='s'))
    results = await asyncio.gather(*[
        store.create_turn(TurnRecord(session_id='s', turn_id=f't{i}'), expected_tip=(None,))
        for i in range(4)
    ], return_exceptions=True)
    assert sum(value is None for value in results) == 1
    assert all(value is None or isinstance(value, SessionStoreConflictError) for value in results)
    assert (await store.get_session('s')).metrics.total_turns == 1


async def test_external_id_race_returns_one_session(store):
    app, _ = build_app([], store=store)
    values = await asyncio.gather(*[
        app.sessions.get_or_create_by_external_id('shared', agent=build_agent()) for _ in range(4)
    ])
    assert len({session.id for session, _ in values}) == 1
    assert sum(created for _, created in values) == 1


async def test_upload_events_and_live_tools_survive_reload(store):
    @agento.tool(read_only=True)
    async def lookup() -> str:
        return 'found'

    app, _ = build_app([say('received')], store=store, artifacts=agento.MemoryArtifactStore())
    agent = build_agent(tools=[lookup])
    session = await app.sessions.create(agent=agent)
    turn = await session.create_turn(agento.UserMessage(content=[agento.FilePart(
        name='input.csv', data='data:text/csv;base64,' + base64.b64encode(b'a,b\n1,2').decode())]))
    live = [event async for event in turn.stream()]
    stored = (await turn.list_events()).items
    uploads = [e for e in live if isinstance(e, agento.ArtifactCreated)]
    assert len(uploads) == 1
    assert uploads[0].id in {e.id for e in stored}
    fresh, _ = build_app([], store=store)
    loaded = await fresh.sessions.get(session.id)
    with pytest.raises(ConfigurationError):
        await loaded.create_turn('use lookup')
    bound = await fresh.sessions.get(session.id, agent=agent)
    assert bound.agent.tools == agent.tools


async def test_awaitable_stream_api_remains_compatible():
    app, _ = build_app([say('session'), say('app')])
    session = await app.sessions.create(agent=build_agent())
    events = [event async for event in await session.stream('hello')]
    assert events[-1].state.output.content == 'session'
    events = [event async for event in await app.stream(build_agent(), 'hello')]
    assert events[-1].state.output.content == 'app'


async def test_cancelled_approved_action_is_reconciled_before_retry(store):
    started = asyncio.Event()
    effects = []

    @agento.tool(destructive=True)
    async def action() -> str:
        effects.append('effect')
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'

    app, llm = build_app([say(tool_calls=[('action', {})]), say('reconcile')], store=store)
    session = await app.sessions.create(agent=build_agent(tools=[action]))
    first = await session.create_turn('action')
    events = [e async for e in first.stream()]
    pending = next(e for e in events if isinstance(e, agento.ApprovalRequired))
    approved = await session.create_turn(agento.ToolApproval(
        thread_id='main', tool_call_id=pending.tool_calls[0].id, decision='allow'))
    task = asyncio.create_task(approved.drain())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = await session.create_turn('continue')
    await resumed.drain()
    assert effects == ['effect']
    results = [m['content'] for m in llm.requests[-1].messages if m['role'] == 'tool']
    assert len(results) == 1
    assert 'Outcome unknown' in results[0]
    assert 'Reconcile' in results[0]


async def test_caller_mutation_cannot_rewrite_stored_state_or_events(store):
    await store.create_session(SessionRecord(session_id='s'))
    await store.create_turn(TurnRecord(session_id='s', turn_id='t'))
    state = agento.TurnStateDone(output=None)
    event = agento.ToolResult(tool_call_id='call', content='original')
    await store.update_turn('s', 't', state=state, events=[event])
    state.required_actions.append(agento.ApprovalRequired(thread_id='main', tool_calls=[]))
    event.content = 'changed'
    assert not (await store.get_turn('s', 't')).state.required_actions
    feed = await store.list_turn_events('s', 't')
    assert feed.items[0].content == 'original'
    feed.items[0].content = 'changed on read'
    assert (await store.list_turn_events('s', 't')).items[0].content == 'original'
