"""Extension boundaries, graph relationships, and non-provider infrastructure."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as Obj

import pytest
from helpers import build_app, say
from test_subagents import _agent, _assistant_turns, _is_sub

import agento
from agento.core.runtime.agent_thread import ThreadSnapshot
from agento.skills.filesystem import parse_skill_markdown


def test_public_exports_and_core_dependency_direction():
    for name in agento.__all__:
        assert getattr(agento, name) is not None, name
    root = Path(agento.__file__).parent
    for path in (root / 'core').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('agento.session'), path
                if node.level >= 3:
                    assert not (node.module or '').startswith('session'), path
    with pytest.raises(AttributeError):
        _ = agento.not_a_public_api


def test_fallback_ids_preserve_order_when_clock_repeats_or_moves_back(monkeypatch):
    from agento import _ids

    monkeypatch.setattr(_ids, '_last_ms', -1)
    monkeypatch.setattr(_ids, '_last_randomness', 0)
    times = iter([2.0, 2.0, 1.0, 3.0])
    monkeypatch.setattr(_ids.time, 'time', lambda: next(times))
    ids = [_ids._fallback_monotonic_ulid() for _ in range(4)]
    assert ids == sorted(set(ids))
    assert all(len(value) == 26 for value in ids)


async def test_skill_resource_containment_and_cache_refresh(tmp_path):
    skill = tmp_path / 'review'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('---\nname: review\ndescription: Review changes\n---\nFirst body')
    (skill / 'notes.md').write_text('resource')
    source = agento.FileSkillSource(tmp_path, max_resource_bytes=8)
    assert [item.name for item in await source.list_skills()] == ['review']
    assert await source.list_skills(['other']) == []
    assert await source.read_skill('review') == 'First body'
    assert await source.read_skill('review') == 'First body'
    (skill / 'SKILL.md').write_text('Second body with different length')
    assert 'Second body' in await source.read_skill('review')
    assert await source.read_resource('review', 'notes.md') == 'resource'
    (skill / 'large.md').write_text('too large for limit')
    with pytest.raises(ValueError, match='larger'):
        await source.read_resource('review', 'large.md')
    with pytest.raises(ValueError, match='escapes'):
        await source.read_resource('review', '../outside.md')
    for name, resource in [('missing', 'notes.md'), ('review', 'missing.md')]:
        with pytest.raises(KeyError):
            await source.read_resource(name, resource)
    with pytest.raises(KeyError):
        await source.read_skill('missing')
    (tmp_path / 'outside.md').write_text('secret')
    (skill / 'link.md').symlink_to(tmp_path / 'outside.md')
    with pytest.raises(ValueError, match='escapes'):
        await source.read_resource('review', 'link.md')
    assert (await agento.FileSkillSource(skill).list_skills())[0].name == 'review'
    assert await agento.FileSkillSource(tmp_path / 'missing').list_skills() == []
    front, body = parse_skill_markdown('# Header\n\nUseful description.', fallback_name='fallback')
    assert front['description'] == 'Useful description.'
    assert 'Header' in body


def test_tracing_closes_spans_and_records_errors():
    pytest.importorskip('opentelemetry')
    recorded = []

    class Span:
        def set_attribute(self, key, value):
            recorded.append(('attribute', key, value))

        def set_status(self, status):
            recorded.append(('status', status))

        def record_exception(self, error):
            recorded.append(('exception', str(error)))

        def end(self):
            recorded.append(('end',))

    from agento.tracing import OTelTracer

    tracer = OTelTracer(Obj(start_span=lambda name: Span()))
    with pytest.raises(RuntimeError, match='boom'), tracer.span('work', count=1) as span:
        span.set_output('redacted')
        raise RuntimeError('boom')
    span.end()  # idempotent end
    assert recorded.count(('end',)) == 1
    assert ('exception', 'boom') in recorded
    assert ('attribute', 'agento.output', 'redacted') in recorded
    with agento.NoopTracer().span('ignored') as noop:
        noop.set_output('discarded')
        noop.set_attribute('key', 'value')
        noop.set_error('discarded')
        noop.end()


@pytest.mark.parametrize('boundary', ['completion', 'join', 'retired'])
async def test_child_resume_never_repeats_completed_model_work(boundary):
    calls = []

    def script(request):
        if _is_sub(request):
            calls.append('child model')
            return say('child summary')
        if _assistant_turns(request) == 0:
            return say(tool_calls=[('create_sub_agent', {'name': 'child', 'input': 'work'})])
        return say('parent answer')

    app, _ = build_app([script] * 10)
    session = await app.sessions.create(agent=_agent())
    turn = await session.create_turn('work')
    stream = turn.stream()
    async for event in stream:
        if (boundary == 'completion' and isinstance(event, agento.ModelMessage)
                and event.thread_id != 'main' and event.content == 'child summary'
            or boundary == 'join' and isinstance(event, agento.ToolResult) and event.thread_id == 'main'
            or boundary == 'retired' and isinstance(event, agento.ThreadDone)):
            if boundary == 'completion':
                record = await app.store.get_turn(session.id, turn.id)
                child = ThreadSnapshot.model_validate(record.snapshot.threads[event.thread_id])
                assert child.completion is not None
            break
    await stream.aclose()
    resumed = await session.create_turn()
    await resumed.drain()
    assert calls == ['child model']
    assert resumed.state.status == 'done'
    assert resumed.state.output.content == 'parent answer'
    assert list(resumed.record.snapshot.threads) == ['main']


async def test_merge_closes_child_generators_on_early_exit():
    from agento.core.runtime.orchestrator import _merge

    closed = []

    async def source(name):
        try:
            while True:
                yield name
                await asyncio.sleep(0)
        finally:
            closed.append(name)

    merged = _merge([source('a'), source('b')])
    await anext(merged)
    await merged.aclose()
    assert sorted(closed) == ['a', 'b']
