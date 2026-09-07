"""Artifact containment, upload validation and provider-independent media conversion."""
from __future__ import annotations

import base64

import pytest

import agento
from agento.core.messages import LLMUserMessage, to_wire_message
from agento.core.runtime.user_input import process_user_message
from agento.errors import InvalidFileInputError


@pytest.mark.parametrize('kind', ['local', 'memory'])
async def test_artifact_lifecycle(kind, tmp_path):
    store = agento.LocalArtifactStore(tmp_path) if kind == 'local' else agento.MemoryArtifactStore()
    first = await store.write(name='a.csv', content=b'012345', source='upload')
    second = await store.write(name='b.txt', content=b'second', source='tool')
    assert (await store.stat(first.id)).name == 'a.csv'
    assert await store.read(first.id, offset=2, length=3) == b'234'
    assert [a.id for a in await store.list(source='upload')] == [first.id]
    assert len(await store.list(limit=1)) == 1
    assert await store.delete(first.id)
    assert not await store.delete(first.id)
    assert await store.stat(first.id) is None
    with pytest.raises(KeyError):
        await store.read(first.id)
    assert await store.read(second.id) == b'second'


@pytest.mark.parametrize('identifier', ['../outside', '/tmp/outside', '..\\outside', '', '.', 'a/b', 'a\x00b'])
async def test_local_artifact_rejects_invalid_ids(identifier, tmp_path):
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'protected')
    store = agento.LocalArtifactStore(tmp_path / 'store')
    for operation in [store.read, store.stat, store.delete]:
        with pytest.raises(ValueError, match='Invalid artifact ID'):
            await operation(identifier)
    assert outside.read_bytes() == b'protected'


async def test_local_artifact_rejects_symlinks_and_corrupt_metadata(tmp_path):
    store = agento.LocalArtifactStore(tmp_path / 'store', max_bytes_per_artifact=8)
    with pytest.raises(ValueError, match='limit'):
        await store.write(name='large', content=b'123456789')
    artifact = await store.write(name='safe', content=b'safe')
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'protected')
    blob = tmp_path / 'store' / f'{artifact.id}.bin'
    blob.unlink()
    blob.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        await store.read(artifact.id)
    with pytest.raises(ValueError, match='symlink'):
        await store.delete(artifact.id)
    meta = tmp_path / 'store' / f'{artifact.id}.json'
    meta.write_text('{broken')
    assert await store.stat(artifact.id) is None
    assert await store.list() == []
    meta.unlink()
    meta.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        await store.stat(artifact.id)
    assert await store.list() == []
    assert outside.read_bytes() == b'protected'


@pytest.mark.parametrize('name,data', [
    ('../bad.csv', 'data:text/csv;base64,YQ=='),
    ('bad.csv', 'not-a-uri'), ('bad.csv', 'data:text/csv;base64,'),
    ('bad.csv', 'data:text/csv;base64,%%%'), ('', 'data:text/csv;base64,YQ=='),
])
async def test_bad_uploads_fail_clearly(name, data):
    with pytest.raises(InvalidFileInputError):
        await process_user_message(agento.UserMessage(content=[agento.FilePart(name=name, data=data)]),
                                   artifacts=agento.MemoryArtifactStore())


async def test_upload_routes_inline_media_and_offloads_csv():
    data = base64.b64encode(b'example').decode()
    message = agento.UserMessage(content=[
        agento.TextPart(text='read these'),
        agento.ImagePart(url='https://example.invalid/image.png'),
        agento.FilePart(name='image.png', data=f'data:image/png;base64,{data}'),
        agento.FilePart(name='paper.pdf', data=f'data:application/pdf;base64,{data}'),
        agento.FilePart(name='table.csv', data=f'data:text/csv;base64,{data}'),
    ])
    store = agento.MemoryArtifactStore()
    processed = await process_user_message(message, artifacts=store)
    assert len(processed.events) == 1
    assert await store.read(processed.events[0].artifact_id) == b'example'
    wire = to_wire_message(processed.messages[0])
    assert [p['type'] for p in wire['content']] == ['image_url', 'image_url', 'file', 'text']
    assert wire['content'][2]['file']['filename'] == 'paper.pdf'
    with pytest.raises(InvalidFileInputError):
        await process_user_message(agento.UserMessage(content=[message.content[-1]]))


def test_wire_image_and_file_match_openai_sdk_schema():
    pytest.importorskip('openai')
    from openai.types.chat import ChatCompletionContentPartImageParam
    from openai.types.chat.chat_completion_content_part_param import File
    from pydantic import TypeAdapter

    message = LLMUserMessage(content=[agento.ImagePart(url='https://example.invalid/a.png'),
        agento.FilePart(name='a.pdf', data='data:application/pdf;base64,YQ==')])
    parts = to_wire_message(message)['content']
    assert TypeAdapter(ChatCompletionContentPartImageParam).validate_python(parts[0]) == parts[0]
    assert TypeAdapter(File).validate_python(parts[1]) == parts[1]
