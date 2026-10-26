"""Isolated M03 tests: M01 is faked, filesystem operations are real."""
from __future__ import annotations

import copy
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import pytest

from researchflow import bundles, documents
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
KINDS = ('run', 'analysis', 'publication', 'compiled', 'context', 'release', 'review')


@pytest.fixture(autouse=True)
def m01_fakes(monkeypatch):
    # Narrow test doubles only; the supplied dependency stub stays untouched.
    def load_document(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            raise WorkflowError('CONTRACT', 'Unreadable JSON') from exc
        if not isinstance(value, dict):
            raise WorkflowError('CONTRACT', 'Expected an object')
        return value

    def canonical_bytes(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')

    def digest_document(value: Any) -> str:
        return hashlib.sha256(canonical_bytes(value)).hexdigest()

    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def atomic_write_json(path: Path, value: Any) -> None:
        temporary = path.with_name('.fake-write.tmp')
        temporary.write_bytes(canonical_bytes(value))
        temporary.replace(path)

    def validate_document(kind: str, value: Mapping[str, Any]) -> None:
        schema = json.loads((ROOT / 'src/researchflow/schemas' / (kind + '.json')).read_text())
        checker = jsonschema.FormatChecker()

        # This runtime lacks jsonschema's optional RFC3339 checker dependency.
        # A narrow fake supplies the fixture date-time semantics explicitly.
        @checker.checks('date-time', raises=ValueError)
        def date_time(candidate):
            if not isinstance(candidate, str):
                return True  # JSON Schema's type validator handles nonstrings.
            return ('T' in candidate.upper()
                    and datetime.fromisoformat(candidate.upper()).tzinfo is not None)

        try:
            jsonschema.Draft202012Validator(
                schema, format_checker=checker).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError('CONTRACT', 'Invalid document', {'path': list(exc.path)}) from exc

    def contained_path(root: Path, relative: str) -> Path:
        result = (root / relative).resolve()
        if not result.is_relative_to(root.resolve()):
            raise WorkflowError('CONTRACT', 'Path escapes root')
        return result

    for fake in (load_document, canonical_bytes, digest_document, sha256_file,
                 atomic_write_json, validate_document, contained_path):
        monkeypatch.setattr(documents, fake.__name__, fake)


def manifest(kind='run', status='succeeded'):
    value = json.loads((ROOT / 'tests/fixtures/schema_examples' / (kind + '_manifest.json')).read_text())
    value['status'] = status
    return value


def commit(tmp_path, value=None):
    value = manifest() if value is None else value
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    (stage / 'payload').mkdir()
    (stage / 'payload/z.bin').write_bytes(b'\x00\xffpayload')
    (stage / 'a.txt').write_text('synthetic payload', encoding='utf-8')
    (stage / 'payload/manifest.json').write_text('{}')
    final = tmp_path / value['bundle_id']
    assert bundles.commit_bundle(stage, final, value) == final
    assert not stage.exists()
    return final


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('status', ['succeeded', 'failed', 'cancelled'])
def test_complete_bundle_all_kinds_and_terminal_statuses(tmp_path, kind, status):
    value = manifest(kind, status)
    original = copy.deepcopy(value)
    final = commit(tmp_path, value)
    result = bundles.verify_bundle(final, kind)
    assert value == original  # Commit must not mutate a caller's nested manifest.
    assert result['status'] == status
    paths = [entry['path'] for entry in result['files']]
    assert paths == ['a.txt', 'payload/manifest.json', 'payload/z.bin']
    for entry in result['files']:
        payload = (final / entry['path']).read_bytes()
        assert entry['sha256'] == hashlib.sha256(payload).hexdigest()
        assert entry['size_bytes'] == len(payload)


@pytest.mark.parametrize('change', ['same_size', 'different_size', 'removed', 'extra', 'size_field', 'hash_field'])
def test_detects_payload_and_inventory_changes_without_repair(tmp_path, change):
    final = commit(tmp_path)
    target = final / 'payload/z.bin'
    if change == 'same_size':
        target.write_bytes(b'x' * target.stat().st_size)
    elif change == 'different_size':
        target.write_bytes(b'longer replacement content')
    elif change == 'removed':
        target.unlink()
    elif change == 'extra':
        (final / 'extra').write_bytes(b'')
    else:
        value = json.loads((final / 'manifest.json').read_text())
        entry = value['files'][0]
        entry['size_bytes' if change == 'size_field' else 'sha256'] = (
            999 if change == 'size_field' else 'f' * 64)
        (final / 'manifest.json').write_text(json.dumps(value))
    before = {p.relative_to(final): p.read_bytes() for p in final.rglob('*') if p.is_file()}
    with pytest.raises(WorkflowError) as caught:
        bundles.verify_bundle(final)
    assert caught.value.code == 'CONTRACT'
    assert before == {p.relative_to(final): p.read_bytes() for p in final.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad_id', ['', '.', '..', '../escape', '/absolute', 'a/b', 'a\\b', 'a b',
                                        'x\n', 'x' * 129, '.hidden', 'CON', 'nul.txt', 'trailing.'])
def test_rejects_unsafe_ids(tmp_path, bad_id):
    with pytest.raises(WorkflowError) as caught:
        bundles.begin_bundle(tmp_path, bad_id)
    assert caught.value.code == 'IDENTITY'
    assert not list(tmp_path.iterdir())


def test_rejects_existing_stage_and_final_identities(tmp_path):
    stage = bundles.begin_bundle(tmp_path, 'sample')
    (stage / 'sentinel').write_bytes(b'stage')
    with pytest.raises(WorkflowError):
        bundles.begin_bundle(tmp_path, 'sample')
    assert (stage / 'sentinel').read_bytes() == b'stage'
    (tmp_path / 'other').mkdir()
    with pytest.raises(WorkflowError):
        bundles.begin_bundle(tmp_path, 'other')


@pytest.mark.parametrize('empty', [True, False])
def test_colliding_commit_preserves_existing_final(tmp_path, empty):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    (stage / 'payload').write_text('new')
    final = tmp_path / value['bundle_id']
    final.mkdir()
    if not empty:
        (final / 'sentinel').write_bytes(b'original')
    before = list(final.iterdir())
    with pytest.raises(WorkflowError) as caught:
        bundles.commit_bundle(stage, final, value)
    assert caught.value.code == 'IDENTITY'
    assert list(final.iterdir()) == before
    if not empty:
        assert (final / 'sentinel').read_bytes() == b'original'
    assert stage.is_dir() and not (stage / 'manifest.json').exists()


def test_racing_empty_final_is_not_overwritten(tmp_path, monkeypatch):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    final = tmp_path / value['bundle_id']
    writer = documents.atomic_write_json

    def competing_writer(path: Path, value: Any) -> None:
        writer(path, value)
        final.mkdir()  # Created after commit's existence check, before rename.

    monkeypatch.setattr(documents, 'atomic_write_json', competing_writer)
    with pytest.raises(WorkflowError) as caught:
        bundles.commit_bundle(stage, final, value)
    assert caught.value.code == 'IDENTITY'
    assert list(final.iterdir()) == []
    assert (stage / 'manifest.json').is_file()


@pytest.mark.parametrize('defect', ['status', 'kind', 'metadata', 'date_time'])
def test_failed_staged_validation_never_publishes(tmp_path, defect):
    value = manifest()
    if defect == 'status':
        value['status'] = 'running'
    elif defect == 'kind':
        value['bundle_kind'] = 'invented'
    elif defect == 'date_time':
        value['metadata']['started_at'] = 'not-a-date'
    else:
        del value['metadata']['identity']
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    final = tmp_path / value['bundle_id']
    with pytest.raises(WorkflowError):
        bundles.commit_bundle(stage, final, value)
    assert not final.exists()
    assert stage.exists() and not (stage / 'manifest.json').exists()


@pytest.mark.parametrize('location', ['payload_file', 'payload_directory', 'dangling', 'manifest', 'root'])
def test_verification_rejects_symlinks(tmp_path, location):
    final = commit(tmp_path)
    if location == 'manifest':
        external = tmp_path / 'external.json'
        (final / 'manifest.json').rename(external)
        (final / 'manifest.json').symlink_to(external)
    elif location == 'root':
        link = tmp_path / 'linked-bundle'
        link.symlink_to(final, target_is_directory=True)
        final = link
    else:
        target = {'payload_file': final / 'a.txt', 'payload_directory': final / 'payload',
                  'dangling': final / 'absent'}[location]
        (final / 'link').symlink_to(target)
    with pytest.raises(WorkflowError):
        bundles.verify_bundle(final)


@pytest.mark.parametrize('link_type', ['file', 'directory', 'dangling'])
def test_commit_rejects_symlink_payloads(tmp_path, link_type):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    target = tmp_path / 'outside'
    if link_type == 'file':
        target.write_text('external')
    elif link_type == 'directory':
        target.mkdir()
    (stage / 'link').symlink_to(target)
    with pytest.raises(WorkflowError):
        bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)
    assert not (tmp_path / value['bundle_id']).exists()


def test_staging_parent_symlink_is_rejected(tmp_path):
    external = tmp_path / 'external'
    external.mkdir()
    (tmp_path / '.staging').symlink_to(external, target_is_directory=True)
    with pytest.raises(WorkflowError):
        bundles.begin_bundle(tmp_path, 'sample')
    assert not list(external.iterdir())


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='FIFO requires POSIX')
def test_special_file_is_rejected_without_opening_it(tmp_path):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    os.mkfifo(stage / 'fifo')
    with pytest.raises(WorkflowError):
        bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)


def test_manifest_cannot_inventory_itself_on_commit(tmp_path):
    value = manifest()
    value['files'] = [{'path': 'manifest.json', 'sha256': '0' * 64, 'size_bytes': 0}]
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    with pytest.raises(WorkflowError):
        bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)


@pytest.mark.parametrize('path', ['manifest.json', '../escape', '/absolute', './a.txt',
                                'payload//z.bin', 'payload\\z.bin', 'C:/escape'])
def test_verification_rejects_noncanonical_inventory_paths(tmp_path, path):
    final = commit(tmp_path)
    value = json.loads((final / 'manifest.json').read_text())
    value['files'][0]['path'] = path
    (final / 'manifest.json').write_text(json.dumps(value))
    with pytest.raises(WorkflowError):
        bundles.verify_bundle(final)


def test_duplicate_inventory_and_wrong_kind_fail(tmp_path):
    final = commit(tmp_path)
    with pytest.raises(WorkflowError):
        bundles.verify_bundle(final, 'analysis')
    value = json.loads((final / 'manifest.json').read_text())
    value['files'].append(copy.deepcopy(value['files'][0]))
    (final / 'manifest.json').write_text(json.dumps(value))
    with pytest.raises(WorkflowError):
        bundles.verify_bundle(final)


def test_terminal_directory_cannot_be_recommitted(tmp_path):
    final = commit(tmp_path)
    before = (final / 'manifest.json').read_bytes()
    with pytest.raises(WorkflowError):
        bundles.commit_bundle(final, tmp_path / 'replacement', manifest())
    assert (final / 'manifest.json').read_bytes() == before
    assert not (tmp_path / 'replacement').exists()


@pytest.mark.parametrize('defect', ['absent', 'invalid_json', 'array'])
def test_missing_or_malformed_manifest(tmp_path, defect):
    tmp_path.mkdir(exist_ok=True)
    if defect != 'absent':
        (tmp_path / 'manifest.json').write_text('{' if defect == 'invalid_json' else '[]')
    with pytest.raises(WorkflowError):
        bundles.verify_bundle(tmp_path)


def test_fingerprint_is_canonical_and_includes_all_terminal_metadata(tmp_path):
    complete = bundles.verify_bundle(commit(tmp_path))

    def reverse_objects(value):
        if isinstance(value, dict):
            return {k: reverse_objects(v) for k, v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reverse_objects(v) for v in value]
        return value

    digest = bundles.bundle_fingerprint(complete)
    assert digest == bundles.bundle_fingerprint(reverse_objects(complete))
    assert digest == hashlib.sha256(documents.canonical_bytes(complete)).hexdigest()
    for field in ('status', 'created_at', 'metadata'):
        altered = copy.deepcopy(complete)
        if field == 'status':
            altered[field] = 'failed'
        elif field == 'created_at':
            altered[field] = '2026-10-26T00:00:00Z'
        else:
            altered[field]['process_status'] = 'different-provenance'
        assert bundles.bundle_fingerprint(altered) != digest
    complete['status'] = 'running'
    with pytest.raises(WorkflowError):
        bundles.bundle_fingerprint(complete)


def test_cross_device_failure_never_publishes(tmp_path, monkeypatch):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    final = tmp_path / value['bundle_id']

    def cross_device(source, target):
        raise OSError(errno.EXDEV, 'Synthetic cross-device rename')

    monkeypatch.setattr(bundles, '_rename_exclusive', cross_device)
    with pytest.raises(WorkflowError) as caught:
        bundles.commit_bundle(stage, final, value)
    assert caught.value.code == 'EXECUTION'
    assert not final.exists() and stage.exists()


def test_empty_bundle_and_generated_inventory(tmp_path):
    value = manifest()
    del value['files']  # Inventory is generated, not required from the caller.
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    final = bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)
    assert bundles.verify_bundle(final)['files'] == []


@pytest.mark.parametrize('inventory', [None, {}, 'invalid'])
def test_malformed_caller_inventory_raises_shared_error(tmp_path, inventory):
    value = manifest()
    value['files'] = inventory
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    with pytest.raises(WorkflowError) as caught:
        bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)
    assert caught.value.code == 'CONTRACT'
    assert not (tmp_path / value['bundle_id']).exists()


def test_failed_manifest_write_never_publishes(tmp_path, monkeypatch):
    value = manifest()
    stage = bundles.begin_bundle(tmp_path, value['bundle_id'])
    (stage / 'payload').write_text('retained diagnostic payload')

    def failed_writer(path: Path, value: Any) -> None:
        raise OSError(errno.ENOSPC, 'Synthetic disk full')

    monkeypatch.setattr(documents, 'atomic_write_json', failed_writer)
    with pytest.raises(WorkflowError) as caught:
        bundles.commit_bundle(stage, tmp_path / value['bundle_id'], value)
    assert caught.value.code == 'EXECUTION'
    assert (stage / 'payload').read_text() == 'retained diagnostic payload'
    assert not (tmp_path / value['bundle_id']).exists()
