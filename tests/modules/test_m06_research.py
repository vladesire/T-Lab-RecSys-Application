"""M06 isolated checks: M01 is faked, using the supplied schemas as oracle."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import pytest
import yaml

from researchflow import research
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def m01_fakes(monkeypatch):
    def load_document(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())

    def validate_document(kind: str, value: Mapping[str, Any]) -> None:
        schema = json.loads((ROOT / 'src/researchflow/schemas' / f'{kind}.json').read_text())
        checker = jsonschema.FormatChecker()

        @checker.checks('date-time', raises=ValueError)
        def date_time(value):
            if not isinstance(value, str):
                return True
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})', value):
                return False
            return datetime.fromisoformat(value.upper().replace('Z', '+00:00')).tzinfo is not None

        try:
            jsonschema.Draft202012Validator(schema, format_checker=checker).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError('CONTRACT', exc.message) from exc

    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def contained_path(root: Path, relative: str) -> Path:
        result = (root / relative).resolve()
        if Path(relative).is_absolute() or not result.is_relative_to(root.resolve()):
            raise WorkflowError('CONTRACT', 'Unsafe path')
        return result

    for name, fake in locals().copy().items():
        if name in ('load_document', 'validate_document', 'sha256_file', 'contained_path'):
            monkeypatch.setattr(research, name, fake)


def example(kind):
    return json.loads((ROOT / 'tests/fixtures/schema_examples' / f'{kind}.json').read_text())


def write_record(root, plural, record, name=None):
    path = root / 'research' / plural / (name or f"{record['id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return path


def write_journal(root, header, body='', name=None, newline='\n'):
    path = root / 'research/journal' / (name or f"{header['id']}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    front = '---\n' + yaml.safe_dump(header, sort_keys=False) + '---\n'
    path.write_bytes(front.replace('\n', newline).encode() + body.encode())
    return path


@pytest.fixture
def linked(tmp_path):
    for plural, kind in [('hypotheses', 'hypothesis'), ('branches', 'branch'), ('decisions', 'decision')]:
        write_record(tmp_path, plural, example(kind))
    return tmp_path


def test_empty_project(tmp_path):
    assert research.load_research(tmp_path) == example('research_index')
    assert list(tmp_path.iterdir()) == []


def test_linked_records_exact_bytes_and_declared_metadata(linked, monkeypatch, tmp_path):
    header = example('journal_header')
    header.update(status='retrospective', recorded_at='2026-10-24T11:12:13.1200+03:00',
                  occurred_at=None, decision_ids=['D-fixture'])
    body = '\r\n# Наблюдение\r\n  line with trailing spaces  \r\n\r\nno final newline'
    path = write_journal(linked, header, body, newline='\r\n')
    # CWD has no effect on resolution against the explicit project root.
    other = tmp_path / 'elsewhere'
    other.mkdir()
    monkeypatch.chdir(other)
    index = research.load_research(linked)
    for plural in ['hypotheses', 'branches', 'decisions', 'journal']:
        for envelope in index[plural]:
            source = linked / envelope['source_path']
            assert envelope['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    journal = index['journal'][0]
    assert journal == {'header': header, 'body': body, 'legacy': False,
                       'source_path': path.relative_to(linked).as_posix(),
                       'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert index['hypotheses'][0]['record'] == example('hypothesis')
    assert index['decisions'][0]['record'] == example('decision')


@pytest.mark.parametrize('kind,plural', [('hypothesis', 'hypotheses'), ('branch', 'branches'),
                                         ('decision', 'decisions'), ('journal_header', 'journal')])
def test_duplicate_ids(linked, kind, plural):
    if plural == 'journal':
        write_journal(linked, example(kind), name='first.md')
        write_journal(linked, example(kind), name='another.md')
    else:
        write_record(linked, plural, example(kind), name='copy.json')
    with pytest.raises(WorkflowError) as err:
        research.load_research(linked)
    assert err.value.code == 'IDENTITY'


@pytest.mark.parametrize('kind,plural,field,value', [
    ('branch', 'branches', 'parent_id', 'B-missing'),
    ('branch', 'branches', 'hypothesis_ids', ['H-missing']),
    ('branch', 'branches', 'status_decision_id', 'D-missing'),
    ('hypothesis', 'hypotheses', 'branch_ids', ['B-missing']),
    ('decision', 'decisions', 'hypothesis_ids', ['H-missing']),
    ('decision', 'decisions', 'branch_ids', ['B-missing']),
    ('decision', 'decisions', 'supersedes', ['D-missing']),
    ('journal_header', 'journal', 'decision_ids', ['D-missing']),
    ('journal_header', 'journal', 'supersedes', ['J-missing']),
])
def test_required_references_fail(linked, kind, plural, field, value):
    record = example(kind)
    record[field] = value
    if plural == 'journal':
        write_journal(linked, record)
    else:
        write_record(linked, plural, record)
    with pytest.raises(WorkflowError) as err:
        research.load_research(linked)
    assert err.value.code == 'IDENTITY'
    assert err.value.details['field'] == field


def test_future_evidence_retained_explicitly_and_deterministically(linked):
    header = example('journal_header')
    header.update(run_ids=['R-future'], analysis_ids=['A-future'])
    write_journal(linked, header)
    decision = example('decision')
    decision['evidence'] = ['run:R-future', 'analysis:A-other', 'analyses/A-future-hash/analysis.json#/metrics/0']
    write_record(linked, 'decisions', decision)
    index = research.load_research(linked)
    unresolved = {(row['kind'], row['id']) for row in index['unresolved']}
    assert unresolved == {('run', 'R-future'), ('analysis', 'A-future'), ('analysis', 'A-other'),
                          ('experiment', 'E-fixture'),
                          ('evidence', 'analyses/A-future-hash/analysis.json#/metrics/0')}
    assert all(row['reason'] for row in index['unresolved'])
    assert research.load_research(linked) == index
    assert index['decisions'][0]['record']['evidence'] == decision['evidence']


def test_legacy_is_opaque_checksummed_and_does_not_invent_history(tmp_path):
    path = tmp_path / 'research/journal/legacy.md'
    path.parent.mkdir(parents=True)
    raw = '---\r\nThis is not YAML: [\r\nJ001-J009 retrospectively described.\r\n\r\n  Русский текст  '.encode()
    path.write_bytes(raw)
    index = research.load_research(tmp_path)
    assert index['journal'] == [{'header': None, 'body': raw.decode(), 'legacy': True,
                                  'source_path': 'research/journal/legacy.md',
                                  'source_sha256': hashlib.sha256(raw).hexdigest()}]
    assert path.read_bytes() == raw


def test_append_preserves_input_body_unknowns_and_old_files(linked):
    header = example('journal_header')
    for field in ['id', 'recorded_at', 'occurred_at', 'model']:
        header.pop(field)
    before = deepcopy(header)
    start = datetime.now(timezone.utc)
    first = research.append_journal(linked, header, '\ntext  \r\nno final newline')
    second = research.append_journal(linked, header, '')
    assert first != second
    assert header == before
    entries = research.load_research(linked)['journal']
    for entry in entries:
        assert entry['header']['occurred_at'] is None
        assert entry['header']['model'] is None
        recorded = datetime.fromisoformat(entry['header']['recorded_at'].replace('Z', '+00:00'))
        assert start <= recorded <= datetime.now(timezone.utc)
    assert next(e['body'] for e in entries if e['source_path'].endswith(first.name)) == '\ntext  \r\nno final newline'
    assert not list(first.parent.glob('*.tmp'))


@pytest.mark.parametrize('historical_name', ['J-fixture.md', 'renamed-history.md'])
def test_collision_never_changes_previous_bytes(linked, historical_name):
    header = example('journal_header')
    old = write_journal(linked, header, 'original\r\n', name=historical_name)
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    with pytest.raises(WorkflowError) as err:
        research.append_journal(linked, header, 'replacement')
    assert err.value.code == 'IDENTITY'
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest
    assert list(old.parent.iterdir()) == [old]


def test_simultaneous_same_id_has_one_success_and_no_partial_file(linked, monkeypatch):
    # Both writers deliberately receive the same preflight snapshot to force
    # their collision into the publication step, rather than sequential checks.
    snapshot = research.load_research(linked)
    monkeypatch.setattr(research, 'load_research', lambda root: deepcopy(snapshot))
    header = example('journal_header')

    def append(body):
        try:
            return research.append_journal(linked, header, body)
        except WorkflowError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, ['first complete body', 'second complete body']))
    assert sum(isinstance(result, Path) for result in results) == 1
    assert [result.code for result in results if isinstance(result, WorkflowError)] == ['IDENTITY']
    path = next(result for result in results if isinstance(result, Path))
    assert path.read_text().endswith(('first complete body', 'second complete body'))
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize('edit', ['bad-id', 'bad-time', 'unknown-field'])
def test_record_schema_errors(linked, edit):
    record = example('hypothesis')
    if edit == 'bad-id':
        record['id'] = '../escape'
    elif edit == 'bad-time':
        record['created_at'] = 'not-a-date'
    else:
        record['invented'] = True
    write_record(linked, 'hypotheses', record, name='H-fixture.json')
    with pytest.raises(WorkflowError) as err:
        research.load_research(linked)
    assert err.value.code == 'CONTRACT'


@pytest.mark.parametrize('raw', ['no header', '---\nid: J-1\n', '---\n[]\n---\n',
                                 '---\nid: J-1\nid: J-2\n---\n',
                                 '---\nx: !!python/object:builtins.object {}\n---\n'])
def test_malformed_journal_rejected(tmp_path, raw):
    path = tmp_path / 'research/journal/bad.md'
    path.parent.mkdir(parents=True)
    path.write_text(raw)
    with pytest.raises(WorkflowError) as err:
        research.load_research(tmp_path)
    assert err.value.code == 'CONTRACT'


def test_unquoted_timestamp_keeps_exact_lexeme_and_yaml_scalar_types(linked):
    path = write_journal(linked, example('journal_header'))
    text = path.read_text().replace("'2026-10-25T00:00:00Z'", '2026-10-25T00:00:00.000+00:00')
    path.write_text(text)
    header = research.load_research(linked)['journal'][0]['header']
    assert header['recorded_at'] == '2026-10-25T00:00:00.000+00:00'
    path.write_text(text.replace('id: J-fixture', 'id: true'))
    with pytest.raises(WorkflowError):
        research.load_research(linked)


def test_missing_reference_append_leaves_no_file(linked):
    header = example('journal_header')
    header['hypothesis_ids'] = ['H-missing']
    with pytest.raises(WorkflowError):
        research.append_journal(linked, header, '')
    assert not (linked / 'research/journal').exists()


def test_branch_cycle_is_rejected(linked):
    branch = example('branch')
    branch['parent_id'] = 'B-second'
    write_record(linked, 'branches', branch)
    second = dict(branch, id='B-second', parent_id='B-fixture')
    write_record(linked, 'branches', second)
    with pytest.raises(WorkflowError) as err:
        research.load_research(linked)
    assert err.value.code == 'IDENTITY'


def test_symlink_escape_rejected(tmp_path):
    root, external = tmp_path / 'project', tmp_path / 'external'
    root.mkdir()
    external.mkdir()
    (root / 'research').symlink_to(external, target_is_directory=True)
    with pytest.raises(WorkflowError) as err:
        research.load_research(root)
    assert err.value.code == 'CONTRACT'


def test_reserved_legacy_id_cannot_be_appended(linked):
    header = example('journal_header')
    header['id'] = 'legacy'
    with pytest.raises(WorkflowError) as err:
        research.append_journal(linked, header, '')
    assert err.value.code == 'IDENTITY'
    assert not (linked / 'research/journal').exists()


def test_valid_supersession_preserves_both_decisions_and_journal_entries(linked):
    decision = dict(example('decision'), id='D-new', supersedes=['D-fixture'], status='retrospective')
    write_record(linked, 'decisions', decision)
    first_header = example('journal_header')
    first = research.append_journal(linked, first_header, 'Earlier account\n')
    old_bytes = first.read_bytes()
    next_header = dict(first_header, id='J-next', supersedes=['J-fixture'], decision_ids=['D-new'],
                       status='retrospective')
    research.append_journal(linked, next_header, 'Correction\n')
    index = research.load_research(linked)
    assert len(index['decisions']) == 2
    assert len(index['journal']) == 2
    assert first.read_bytes() == old_bytes
    assert next(row['header'] for row in index['journal'] if row['header']['id'] == 'J-next') == next_header


def test_publication_failure_cleans_temporary_file_without_creating_source(linked, monkeypatch):
    def fail_link(source, destination):
        raise OSError('synthetic unavailable hard link')

    monkeypatch.setattr(research.os, 'link', fail_link)
    with pytest.raises(WorkflowError) as err:
        research.append_journal(linked, example('journal_header'), 'Complete text')
    assert err.value.code == 'EXECUTION'
    assert list((linked / 'research/journal').iterdir()) == []


@pytest.mark.parametrize('field,value', [('recorded_at', 'invalid-date-time'), ('id', '../escape'),
                                       ('schema_version', '1')])
def test_invalid_append_metadata_has_no_side_effect(linked, field, value):
    header = example('journal_header')
    header[field] = value
    with pytest.raises(WorkflowError) as err:
        research.append_journal(linked, header, '')
    assert err.value.code == 'CONTRACT'
    assert not (linked / 'research/journal').exists()


def test_yaml_source_document_supported(tmp_path):
    record = example('hypothesis')
    record['branch_ids'] = []
    path = tmp_path / 'research/hypotheses/not-the-record-id.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(record))
    envelope = research.load_research(tmp_path)['hypotheses'][0]
    assert envelope['record'] == record
    assert envelope['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
