"""M11 isolation tests. M01 remains untouched and is replaced only by test fakes."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest

from researchflow import documents, tables
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'tests/fixtures/schema_examples'
SCHEMAS = ROOT / 'src/researchflow/schemas'


def example(name):
    return json.loads((EXAMPLES / (name + '.json')).read_text())


@pytest.fixture
def analysis():
    return example('analysis_result')


@pytest.fixture(autouse=True)
def fake_m01(monkeypatch, analysis):
    def validate(kind, value):
        schema = json.loads((SCHEMAS / (kind + '.json')).read_text())
        try:
            jsonschema.Draft202012Validator(schema).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError('CONTRACT', str(exc)) from exc

    def resolve(document, pointer):
        # Deliberately bounded lookup fake, not an implementation of RFC 6901.
        assert document is analysis
        available = {'': document, '/analysis_id': document['analysis_id'],
                     '/limitations/0': document['limitations'][0]}
        for collection in ('metrics', 'comparisons'):
            for i, row in enumerate(document[collection]):
                available[f'/{collection}/{i}'] = row
                available.update({f'/{collection}/{i}/{key}': value for key, value in row.items()})
        try:
            return available[pointer]
        except KeyError as exc:
            raise WorkflowError('CONTRACT', 'Missing evidence pointer', {'pointer': pointer}) from exc

    def contained(root, relative):
        result = (root / relative).resolve()
        if not result.is_relative_to(root.resolve()):
            raise WorkflowError('CONTRACT', 'Path escape')
        return result

    monkeypatch.setattr(documents, 'validate_document', validate)
    monkeypatch.setattr(documents, 'resolve_pointer', resolve)
    monkeypatch.setattr(documents, 'contained_path', contained)
    monkeypatch.setattr(documents, 'sha256_file', lambda p: hashlib.sha256(p.read_bytes()).hexdigest())


def render(analysis, tmp_path, spec=None):
    return tables.render_tables(analysis, [spec or example('table_spec')], tmp_path)


def test_fixture_precision_descriptor_and_evidence(analysis, tmp_path):
    before = copy.deepcopy(analysis)
    spec = example('table_spec')
    original_spec = copy.deepcopy(spec)
    result = render(analysis, tmp_path, spec)
    text = (tmp_path / 'main.tex').read_text()
    assert '0.6667' in text
    assert text.count(r'\begin{tabular}') == text.count(r'\end{tabular}') == 1
    assert r'\documentclass' not in text
    descriptor, = result
    assert descriptor['id'] == 'main'
    assert descriptor['path'] == 'main.tex'
    assert descriptor['kind'] == 'table'
    assert descriptor['sha256'] == hashlib.sha256((tmp_path / 'main.tex').read_bytes()).hexdigest()
    expected = ['series_id', 'value', 'protocol_id', 'split', 'subgroup', 'weighting', 'metric']
    assert descriptor['evidence'] == [
        {'analysis_id': 'A-fixture', 'pointer': '/metrics/0/' + key} for key in expected]
    assert analysis == before and spec == original_spec


def test_latex_escapes_all_text_positions(analysis, tmp_path):
    raw = 'x_%\\&#${}~^\nnext'
    escaped = r'x\_\%\textbackslash{}\&\#\$\{\}\textasciitilde{}\textasciicircum{} next'
    analysis['metrics'][0]['metric'] = raw
    spec = example('table_spec')
    spec['filters'] = {}
    spec['columns'] = [{'field': 'metric', 'label': raw}]
    spec['caption'] = raw
    render(analysis, tmp_path, spec)
    text = (tmp_path / 'main.tex').read_text()
    assert text.count(escaped) == 3
    assert raw not in text


def test_null_and_zero_are_distinct(analysis, tmp_path):
    analysis['metrics'][0]['value'] = None
    analysis['metrics'].append(dict(analysis['metrics'][0], series_id='B', value=0))
    render(analysis, tmp_path)
    text = (tmp_path / 'main.tex').read_text()
    assert 'A & ' + r'\textemdash{}' in text
    assert 'B & 0.0000' in text


def test_conflicting_contexts_preserved(analysis, tmp_path):
    analysis['metrics'].append(dict(analysis['metrics'][0], protocol_id='P-other', split='final',
                                     subgroup='novel', weighting='user'))
    descriptor, = render(analysis, tmp_path)
    text = (tmp_path / 'main.tex').read_text()
    assert 'Protocol & Split & Subgroup & Weighting & Metric' in text
    assert 'P-fixture & development & all & event' in text
    assert 'P-other & final & novel & user' in text
    assert {'analysis_id': 'A-fixture', 'pointer': '/metrics/1/protocol_id'} in descriptor['evidence']


def test_comparison_filters_and_missing_uncertainty(analysis, tmp_path):
    row = dict(id='AvB', left='A', right='B', protocol_id='P', split='dev', subgroup='all',
               metric='ndcg@2', weighting='event', estimate=0, ci_low=None, ci_high=None,
               n_queries=3, n_users=2, status='unavailable', reason='No paired evidence', evidence=[])
    analysis['comparisons'] = [dict(row, id='discarded', estimate=9), row]
    spec = example('table_spec')
    spec.update(kind='comparison_table', filters={'id': ['AvB']}, columns=[
        {'field': 'estimate', 'label': 'Difference', 'digits': 2},
        {'field': 'ci_low', 'label': 'Lower', 'digits': 2}])
    descriptor, = render(analysis, tmp_path, spec)
    text = (tmp_path / 'main.tex').read_text()
    assert '0.00 & ' + r'\textemdash{}' in text
    assert '9.00' not in text
    assert 'unavailable & No paired evidence' in text
    assert {'analysis_id': 'A-fixture', 'pointer': '/comparisons/1/id'} in descriptor['evidence']
    assert all('/comparisons/1/' in e['pointer'] for e in descriptor['evidence'])


def test_multiple_number_macros_share_one_file_and_descriptor(analysis, tmp_path):
    analysis['metrics'][0]['value'] = 0
    specs = [example('number_spec'), dict(example('number_spec'), name='Count', pointer='/metrics/0/n_users', digits=0)]
    descriptor, = tables.render_numbers(analysis, specs, tmp_path)
    assert (tmp_path / 'numbers.tex').read_text() == (
        '\\newcommand{\\FixtureNDCG}{0.0000}\n\\newcommand{\\Count}{2}\n')
    assert descriptor['kind'] == descriptor['id'] == 'numbers'
    assert descriptor['evidence'] == [
        {'analysis_id': 'A-fixture', 'pointer': '/metrics/0/value'},
        {'analysis_id': 'A-fixture', 'pointer': '/metrics/0/n_users'}]


def test_missing_number_and_empty_numbers(analysis, tmp_path):
    analysis['metrics'][0]['value'] = None
    tables.render_numbers(analysis, [example('number_spec')], tmp_path / 'null')
    assert r'{\textemdash{}}' in (tmp_path / 'null/numbers.tex').read_text()
    descriptor, = tables.render_numbers(analysis, [], tmp_path / 'empty')
    assert (tmp_path / 'empty/numbers.tex').read_bytes() == b''
    assert descriptor['evidence'] == []


def test_duplicate_ids_preflight_all_outputs(analysis, tmp_path):
    spec = example('table_spec')
    with pytest.raises(WorkflowError, match='Duplicate artifact'):
        tables.render_tables(analysis, [spec, spec], tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_duplicate_macros_preflight(analysis, tmp_path):
    spec = example('number_spec')
    with pytest.raises(WorkflowError, match='Duplicate macro'):
        tables.render_numbers(analysis, [spec, spec], tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('pointer', ['/metrics/99/value', '/missing', '/metrics/0/unknown'])
def test_missing_pointer_fails_before_write(analysis, tmp_path, pointer):
    spec = dict(example('number_spec'), pointer=pointer)
    with pytest.raises(WorkflowError) as error:
        tables.render_numbers(analysis, [example('number_spec'), spec], tmp_path)
    assert error.value.code == 'CONTRACT'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['bad1', 'X_y', 'Bad\\input', '', '../escape'])
def test_invalid_macro_names(analysis, tmp_path, name):
    with pytest.raises(WorkflowError):
        tables.render_numbers(analysis, [dict(example('number_spec'), name=name)], tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('filters', [{'invented': 'A'}, {'series_id': 'missing'},
                                   {'series_id': []}, {'metric': {'eval': 'something'}},
                                   {'value': True}])
def test_invalid_filters(analysis, tmp_path, filters):
    spec = dict(example('table_spec'), filters=filters)
    with pytest.raises(WorkflowError):
        render(analysis, tmp_path, spec)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field', ['unknown', '__import__("os")', 'evidence'])
def test_no_unknown_fields_or_expressions(analysis, tmp_path, field):
    spec = dict(example('table_spec'), columns=[{'field': field, 'label': 'X'}])
    with pytest.raises(WorkflowError):
        render(analysis, tmp_path, spec)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_rejected(analysis, tmp_path, value):
    analysis['metrics'][0]['value'] = value
    with pytest.raises(WorkflowError):
        render(analysis, tmp_path)
    with pytest.raises(WorkflowError):
        tables.render_numbers(analysis, [example('number_spec')], tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('pointer', ['/limitations/0', '/metrics/0', ''])
def test_number_macros_reject_non_numeric_values(analysis, tmp_path, pointer):
    with pytest.raises(WorkflowError):
        tables.render_numbers(analysis, [dict(example('number_spec'), pointer=pointer)], tmp_path)


@pytest.mark.parametrize('artifact_id', ['../escape', '/tmp/escape', 'a/b'])
def test_unsafe_ids_rejected(analysis, tmp_path, artifact_id):
    with pytest.raises(WorkflowError):
        render(analysis, tmp_path, dict(example('table_spec'), id=artifact_id))
    assert not list(tmp_path.iterdir())


def test_existing_files_and_symlinks_are_not_overwritten(analysis, tmp_path):
    outside = tmp_path / 'outside.tex'
    outside.write_text('authored text')
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'main.tex').symlink_to(outside)
    with pytest.raises(WorkflowError):
        render(analysis, output)
    assert outside.read_text() == 'authored text'
    (output / 'main.tex').unlink()
    (output / 'main.tex').write_text('existing artifact')
    with pytest.raises(WorkflowError) as error:
        render(analysis, output)
    assert error.value.code == 'IDENTITY'
    assert (output / 'main.tex').read_text() == 'existing artifact'


def test_no_tables_and_empty_analysis(analysis, tmp_path):
    assert tables.render_tables(analysis, [], tmp_path / 'unused') == []
    assert not (tmp_path / 'unused').exists()
    analysis['metrics'] = []
    render(analysis, tmp_path, dict(example('table_spec'), filters={}))
    assert r'\end{tabular}' in (tmp_path / 'main.tex').read_text()


def test_bad_later_spec_prevents_partial_output(analysis, tmp_path):
    spec = dict(example('table_spec'), id='second', columns=[{'field': 'typo', 'label': 'X'}])
    with pytest.raises(WorkflowError):
        tables.render_tables(analysis, [example('table_spec'), spec], tmp_path)
    assert not list(tmp_path.iterdir())


def test_declared_precision_zero_and_integer_columns(analysis, tmp_path):
    spec = dict(example('table_spec'), columns=[{'field': 'value', 'label': 'Score', 'digits': 0},
                                               {'field': 'n_queries', 'label': 'Queries'}])
    render(analysis, tmp_path, spec)
    assert '1 & 3 & P-fixture' in (tmp_path / 'main.tex').read_text()


def test_real_latex_smoke_when_available(analysis, tmp_path):
    engine = shutil.which('pdflatex')
    if engine is None:
        pytest.skip('pdflatex is not installed; isolated string tests still run')
    analysis['metrics'][0]['metric'] = 'x_%\\&#${}~^'
    analysis['metrics'][0]['value'] = None
    spec = dict(example('table_spec'), filters={}, columns=[
        {'field': 'metric', 'label': 'Score_%\\'}, {'field': 'value', 'label': 'Value'}])
    render(analysis, tmp_path, spec)
    tables.render_numbers(analysis, [example('number_spec')], tmp_path)
    (tmp_path / 'smoke.tex').write_text(
        '\\documentclass{article}\n\\begin{document}\n'
        '\\input{main.tex}\n\\input{numbers.tex}\n\\FixtureNDCG\n\\end{document}\n')
    completed = subprocess.run(
        [engine, '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode', 'smoke.tex'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / 'smoke.pdf').read_bytes().startswith(b'%PDF-')
