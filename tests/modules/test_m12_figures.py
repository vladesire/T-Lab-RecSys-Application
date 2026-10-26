"""Isolated M12 tests; M01 functions are replaced by signature-matching fakes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.figure import Figure

from researchflow import figures
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def m01_fakes(monkeypatch):
    def validate_document(kind: str, value: Mapping[str, Any]) -> None:
        schema = json.loads((ROOT / 'src/researchflow/schemas' / (kind + '.json')).read_text())
        try:
            jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError('CONTRACT', exc.message, {'path': list(exc.path)}) from exc

    def contained_path(root: Path, relative: str) -> Path:
        # M12 supplies only schema-validated IDs followed by '.pdf'.
        assert '/' not in relative and '\\' not in relative
        return root.resolve() / relative

    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(figures, 'validate_document', validate_document)
    monkeypatch.setattr(figures, 'contained_path', contained_path)
    monkeypatch.setattr(figures, 'sha256_file', sha256_file)


@pytest.fixture
def analysis():
    result = json.loads((ROOT / 'tests/fixtures/schema_examples/analysis_result.json').read_text())
    result['metrics'][0]['value'] = 0.3
    result['metrics'].append(dict(result['metrics'][0], series_id='B', method_id='B', value=0.8))
    base = {k: result['metrics'][0][k] for k in ('protocol_id', 'split', 'subgroup', 'metric', 'weighting')}
    result['comparisons'] = [dict(base, id='B-A', left='B', right='A', estimate=0.5,
        ci_low=-1.25, ci_high=2.75, n_queries=3, n_users=2,
        status='computed', reason=None, evidence=['/metrics/0', '/metrics/1'])]
    return result


@pytest.fixture
def spec():
    return json.loads((ROOT / 'tests/fixtures/schema_examples/figure_spec.json').read_text())


@pytest.fixture
def captured(monkeypatch):
    snapshots = []
    savefig = Figure.savefig

    def capture(self, *args, **kwargs):
        ax = self.axes[0]
        snapshots.append({
            'bars': [(p.get_x() + p.get_width() / 2, p.get_height()) for p in ax.patches],
            'ticks': [t.get_text() for t in ax.get_xticklabels()],
            'ylim': ax.get_ylim(),
            'segments': [s.copy() for c in ax.collections for s in c.get_segments()],
            'texts': [t.get_text() for t in ax.texts],
            'points': [(list(p.get_xdata()), list(p.get_ydata())) for p in ax.lines if p.get_marker() == 'o'],
            'ylabel': ax.get_ylabel(), 'title': ax.get_title(),
        })
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    return snapshots


def interval_spec(spec, order=None):
    return dict(spec, id='intervals', kind='comparison_intervals', order=['B-A'] if order is None else order,
                title='Synthetic comparison fixture', ylabel='Difference')


def test_pdfs_descriptors_evidence_and_vector_content(analysis, spec, tmp_path, captured):
    originals = copy.deepcopy((analysis, spec))
    artifacts = figures.render_figures(analysis, [dict(spec, order=['B', 'A']), interval_spec(spec)], tmp_path)
    assert (analysis, spec) == originals
    assert [a['id'] for a in artifacts] == ['mainfig', 'intervals']
    for artifact in artifacts:
        path = tmp_path / artifact['path']
        assert path.read_bytes().startswith(b'%PDF-') and path.stat().st_size > 1000
        assert not Path(artifact['path']).is_absolute()
        assert artifact['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact['kind'] == 'figure'
        # Vector PDF, with embedded text/fonts rather than a rasterized plot.
        assert b'/Subtype /Image' not in path.read_bytes()
        for item in artifact['evidence']:
            assert item['analysis_id'] == analysis['analysis_id']
            collection, index = item['pointer'].strip('/').split('/')
            assert analysis[collection][int(index)]
    assert [e['pointer'] for e in artifacts[0]['evidence']] == ['/metrics/1', '/metrics/0']
    assert artifacts[1]['evidence'][0]['pointer'] == '/comparisons/0'
    assert 'P-fixture' in captured[0]['title']
    assert 'ndcg@2' in captured[0]['ylabel'] and 'event' in captured[0]['ylabel']


def test_requested_order_moves_values_without_changing_them(analysis, spec, tmp_path, captured):
    figures.render_figures(analysis, [dict(spec, order=['B', 'A'])], tmp_path / 'ba')
    figures.render_figures(analysis, [dict(spec, order=['A', 'B'])], tmp_path / 'ab')
    assert captured[0]['ticks'] == ['B', 'A']
    assert captured[1]['ticks'] == ['A', 'B']
    assert captured[0]['bars'] == [(0, 0.8), (1, 0.3)]
    assert captured[1]['bars'] == [(0, 0.3), (1, 0.8)]
    figures.render_figures(analysis, [spec], tmp_path / 'subset')
    assert captured[2]['bars'] == [(0, 0.3)]
    figures.render_figures(analysis, [dict(spec, order=[])], tmp_path / 'all')
    assert captured[3]['ticks'] == ['A', 'B']


def test_actual_saved_asymmetric_limits_and_bounds(analysis, spec, tmp_path, captured):
    figures.render_figures(analysis, [interval_spec(spec)], tmp_path)
    plot = captured[0]
    np.testing.assert_allclose(plot['segments'][0], [[0, -1.25], [0, 2.75]])
    assert plot['ylim'][0] < -1.25 and plot['ylim'][1] > 2.75
    assert plot['points'] == [([0], [0.5])]


def test_ci_not_containing_estimate_is_not_recentered(analysis, spec, tmp_path, captured):
    analysis['comparisons'][0].update(estimate=-3, ci_low=1, ci_high=2)
    figures.render_figures(analysis, [interval_spec(spec)], tmp_path)
    np.testing.assert_allclose(captured[0]['segments'][0], [[0, 1], [0, 2]])
    assert captured[0]['ylim'][0] < -3 and captured[0]['ylim'][1] > 2


@pytest.mark.parametrize('missing_estimate', [False, True])
def test_null_interval_has_no_zero_interval(analysis, spec, tmp_path, captured, missing_estimate):
    analysis['comparisons'][0].update(ci_low=None, ci_high=None)
    if missing_estimate:
        analysis['comparisons'][0].update(estimate=None, status='unavailable', reason='No paired data')
    figures.render_figures(analysis, [interval_spec(spec)], tmp_path)
    assert captured[0]['segments'] == []
    assert captured[0]['texts'] == ['missing' if missing_estimate else 'CI missing']
    assert captured[0]['points'] == ([] if missing_estimate else [([0], [0.5])])


def test_null_bar_is_missing_but_real_zero_is_plotted(analysis, spec, tmp_path, captured):
    analysis['metrics'][0]['value'] = None
    analysis['metrics'][1]['value'] = 0.0
    figures.render_figures(analysis, [dict(spec, order=['A', 'B'])], tmp_path)
    assert captured[0]['bars'] == [(1, 0.0)]
    assert captured[0]['texts'] == ['missing']


@pytest.mark.parametrize('change', ['series', 'metric', 'duplicate_id', 'duplicate_order', 'kind', 'filter', 'mixed', 'ambiguous'])
def test_invalid_selection_fails_before_writing(analysis, spec, tmp_path, change):
    specs = [spec]
    if change == 'series': spec['order'] = ['unknown']
    if change == 'metric': spec['filters']['metric'] = 'unknown'
    if change == 'duplicate_id': specs.append(copy.deepcopy(spec))
    if change == 'duplicate_order': spec['order'] = ['A', 'A']
    if change == 'kind': spec['kind'] = 'scatter'
    if change == 'filter': spec['filters'] = {'p_value': 'significant'}
    if change == 'mixed':
        spec['order'] = ['A', 'B']
        analysis['metrics'][1]['weighting'] = 'user'
    if change == 'ambiguous': analysis['metrics'].append(copy.deepcopy(analysis['metrics'][0]))
    with pytest.raises(WorkflowError) as err:
        figures.render_figures(analysis, specs, tmp_path)
    assert err.value.code == 'CONTRACT'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('case', ['absent', 'partial', 'reversed', 'no_estimate', 'nan', 'inf', 'unavailable'])
def test_invalid_intervals_fail(analysis, spec, tmp_path, case):
    row = analysis['comparisons'][0]
    if case == 'absent': del row['ci_low']
    if case == 'partial': row['ci_low'] = None
    if case == 'reversed': row.update(ci_low=3, ci_high=1)
    if case == 'no_estimate': row['estimate'] = None
    if case == 'nan': row['ci_low'] = float('nan')
    if case == 'inf': row['ci_high'] = float('inf')
    if case == 'unavailable': row['status'] = 'unavailable'
    with pytest.raises(WorkflowError):
        figures.render_figures(analysis, [interval_spec(spec)], tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonfinite_bar_rejected(analysis, spec, tmp_path):
    analysis['metrics'][0]['value'] = float('inf')
    with pytest.raises(WorkflowError):
        figures.render_figures(analysis, [spec], tmp_path)


def test_style_and_backend_restored_and_no_tex(analysis, spec, tmp_path):
    backend = mpl.get_backend()
    with mpl.rc_context({'text.usetex': True, 'font.size': 19, 'axes.facecolor': 'pink'}):
        before = dict(mpl.rcParams)
        figures.render_figures(analysis, [dict(spec, title='$literal$ and \\text{literal}')], tmp_path)
        assert dict(mpl.rcParams) == before
    assert mpl.get_backend() == backend


def test_existing_output_preserved_and_empty_specs(analysis, spec, tmp_path):
    path = tmp_path / 'mainfig.pdf'
    path.write_bytes(b'previous')
    with pytest.raises(WorkflowError):
        figures.render_figures(analysis, [spec], tmp_path)
    assert path.read_bytes() == b'previous'
    assert figures.render_figures(analysis, [], tmp_path) == []


def test_relative_output_directory(analysis, spec, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = figures.render_figures(analysis, [spec], Path('relative'))
    assert result[0]['path'] == 'mainfig.pdf'
    assert Path('relative/mainfig.pdf').is_file()
