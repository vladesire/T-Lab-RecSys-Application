"""Independent synthetic checks; no production dependency fakes are used."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from researchflow.contracts import WorkflowError
from researchflow.ranking import aggregate, paired_interval, per_query_metrics

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'


def fixture(method='A'):
    data = json.loads((FIXTURES / 'ranking_rows.json').read_text())
    return {key: np.asarray(value) for key, value in data.items() if not key.startswith('score_')} | {
        'score': np.asarray(data['score_' + method])}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def interval(left, right, **kwargs):
    options = dict(metric='ndcg', weighting='event', subgroup='all', replicates=101, seed=41)
    options.update(kwargs)
    return paired_interval(left, right, **options)


def test_fixture_definitions_and_unequal_user_means():
    expected = json.loads((FIXTURES / 'ranking_expected.json').read_text())
    for method in ('A', 'B'):
        rows = per_query_metrics(fixture(method), k=2)
        ranks = expected[method + '_ranks']
        assert [r['ndcg'] for r in rows] == [1 / math.log2(r + 1) if r <= 2 else 0 for r in ranks]
        for weighting, metric in itertools.product(('event', 'user'), ('ndcg', 'recall')):
            result = aggregate(rows, metric=metric, weighting=weighting)
            assert result == dict(value=pytest.approx(expected[f'{method}_{weighting}_{metric}']),
                                  n_queries=3, n_users=2)


def test_subgroups_and_empty():
    rows = per_query_metrics(fixture(), k=2)
    assert aggregate(rows, metric='ndcg', weighting='event', subgroup='repeat') == dict(value=1, n_queries=1, n_users=1)
    assert aggregate(rows, metric='ndcg', weighting='user', subgroup='novel') == dict(value=.5, n_queries=2, n_users=2)
    assert aggregate(rows[:1], metric='recall', weighting='user', subgroup='novel') == dict(value=None, n_queries=0, n_users=0)
    assert aggregate([], metric='recall', weighting='event') == dict(value=None, n_queries=0, n_users=0)
    with pytest.raises(WorkflowError, match='empty'):
        interval(rows[:1], rows[:1], subgroup='novel')


def test_ties_use_keys_then_lexical_item_never_labels_or_row_order():
    data = {key: value[:3].copy() for key, value in fixture().items()}
    data['item_id'] = np.array(['z', 'a', 'b'])
    data['score'] = np.array([4., 4., 4.])
    data['tie_key'] = np.array([2**64-1, 2**64-1, 2**64-2], dtype=np.uint64)
    # b then a then z; target z is last, despite coming first in the input.
    for order in itertools.permutations(range(3)):
        permuted = {key: value[list(order)] for key, value in data.items()}
        row, = per_query_metrics(permuted, k=3)
        assert row['ndcg'] == .5
        assert row['recall'] == 1
        assert per_query_metrics(permuted, k=2)[0]['ndcg'] == 0
    assert per_query_metrics(data, k=100)[0]['ndcg'] == .5
    # Change only the positive and corresponding familiarity: rankings remain b,a,z.
    data['label'] = np.array([0, 1, 0])
    data['is_familiar'] = np.array([1, 1, 0])
    assert per_query_metrics(data, k=3)[0]['ndcg'] == 1 / math.log2(3)


def test_exact_unicode_hashes_counts_and_permutation():
    data = {key: value[:3].copy() for key, value in fixture().items()}
    data['query_id'] = np.array(['é' * 200] * 3)
    data['user_id'] = np.array([2**64-1] * 3, dtype=np.uint64)
    data['item_id'] = np.array(['目标' * 100, 'a', 'z'])
    data['is_familiar'] = np.array([1, 1, 0])
    row, = per_query_metrics(data, k=2)
    record = dict(query_id='é'*200, user_id=str(2**64-1), target_item_id='目标'*100,
                  timestamp=100, is_repeat_target=True)
    candidates = [dict(item_id='a', is_familiar=True, tie_key=2),
                  dict(item_id='z', is_familiar=False, tie_key=3),
                  dict(item_id='目标'*100, is_familiar=True, tie_key=1)]
    assert {key: row[key] for key in record} == record
    assert row['query_identity_sha256'] == digest(record)
    assert row['candidate_set_sha256'] == digest(candidates)
    assert row['n_candidates'] == 3 and row['n_familiar_alternatives'] == 1
    assert per_query_metrics({key: value[::-1] for key, value in data.items()}, k=2) == [row]
    changed = deepcopy(data)
    changed['score'] *= -1
    new, = per_query_metrics(changed, k=2)
    assert new['candidate_set_sha256'] == row['candidate_set_sha256']
    assert new['query_identity_sha256'] == row['query_identity_sha256']


def test_integer_score_extremes_do_not_overflow():
    data = {key: value[:3].copy() for key, value in fixture().items()}
    for scores in (np.array([0, 2**64-1, 1], dtype=np.uint64),
                   np.array([-2**63, 2**63-1, 0], dtype=np.int64)):
        data['score'] = scores
        assert per_query_metrics(data, k=2)[0]['recall'] == 0


@pytest.mark.parametrize('field,replacement', [
    ('label', [1, 1, 0, 0, 0, 1, 0, 1, 0]),
    ('label', [0]*9),
    ('item_id', ['a', 'a', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
    ('user_id', ['u1', 'other', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2']),
    ('timestamp', [100, 101, 100, 200, 200, 200, 300, 300, 300]),
    ('is_repeat_target', [1, 0, 1, 0, 0, 0, 0, 0, 0]),
    ('is_familiar', [0]*9),
    ('score', [float('nan')]*9), ('score', [float('inf')]*9),
    ('score', np.ones(9, dtype=complex)), ('score', ['1']*9),
    ('query_id', np.array(['q']*9, dtype=object)), ('user_id', [1.5]*9),
    ('item_id', [b'x']*9), ('timestamp', [-1]*9), ('timestamp', [1.0]*9),
    ('tie_key', [-1]*9), ('tie_key', [0.5]*9), ('tie_key', [True]*9),
    ('label', [2]*9), ('is_familiar', [1.0]*9),
    ('score', np.ones((9, 1))), ('score', [1]*8), ('score', 1),
])
def test_invalid_arrays_fail(field, replacement):
    data = fixture()
    data[field] = np.asarray(replacement)
    with pytest.raises(WorkflowError) as error:
        per_query_metrics(data, k=2)
    assert error.value.code == 'CONTRACT'


def test_missing_array_and_invalid_k():
    data = fixture()
    del data['label']
    with pytest.raises(WorkflowError):
        per_query_metrics(data, k=1)
    for k in (0, -1, True, 1.5, '2'):
        with pytest.raises(WorkflowError):
            per_query_metrics(fixture(), k=k)


def test_typed_empty_arrays_and_integer_ids():
    data = fixture()
    assert per_query_metrics({key: value[:0] for key, value in data.items()}, k=1) == []
    for key in ('query_id', 'user_id'):
        data[key] = np.ones(9, dtype=np.int64)
    data['item_id'] = np.arange(9)
    data['timestamp'][:] = 100
    data['is_repeat_target'][:] = 0
    data['is_familiar'][:] = 0
    data['label'][:] = 0
    data['label'][0] = 1
    row, = per_query_metrics(data, k=1)
    assert row['query_id'] == '1' and row['target_item_id'] == '0'


def test_constant_paired_difference_and_one_user():
    left = per_query_metrics(fixture(), k=2)
    right = deepcopy(left)
    for a, b in zip(left, right):
        a['ndcg'], b['ndcg'] = .75, .25
    for weighting in ('event', 'user'):
        assert interval(left, right, weighting=weighting) == dict(
            estimate=.5, ci_low=.5, ci_high=.5, n_queries=3, n_users=2)
        assert interval(left[:1], right[:1], weighting=weighting) == dict(
            estimate=.5, ci_low=.5, ci_high=.5, n_queries=1, n_users=1)
    assert interval(right, left)['estimate'] == -.5


@pytest.mark.parametrize('weighting', ['event', 'user'])
def test_bootstrap_matches_independent_small_cluster_oracle(weighting):
    left = per_query_metrics(fixture(), k=2)
    right = per_query_metrics(fixture('B'), k=2)
    # User u1 differences: (1-d), -1; u2 difference: (1-d).
    d = 1 / math.log2(3)
    totals, counts = [-d, 1-d], [2, 1]
    rng = np.random.default_rng(9)
    samples = rng.choice(2, size=(37, 2), replace=True)
    draws = []
    for a, b in samples:
        draws.append((totals[a]+totals[b]) / (counts[a]+counts[b]) if weighting == 'event'
                     else (totals[a]/counts[a] + totals[b]/counts[b])/2)
    # Deliberately low confidence exercises interpolated, non-extreme quantiles.
    expected_ci = np.quantile(draws, [.35, .65], method='linear')
    got = interval(left, right, weighting=weighting, replicates=37, seed=9, confidence=.3)
    expected = (1-2*d)/3 if weighting == 'event' else (-d/2 + 1-d)/2
    assert got['estimate'] == pytest.approx(expected)
    assert [got['ci_low'], got['ci_high']] == pytest.approx(expected_ci)
    assert interval(left[::-1], right[::-1], weighting=weighting, replicates=37,
                    seed=9, confidence=.3) == got
    assert interval(left, left, weighting=weighting)['ci_low'] == 0


def test_event_mode_detects_mean_of_user_means_bug():
    left = per_query_metrics(fixture(), k=2)
    right = deepcopy(left)
    for a, b in zip(left, right):
        a['recall'] = float(a['user_id'] == 'u1')
        b['recall'] = 0.
    assert interval(left, right, metric='recall')['estimate'] == pytest.approx(2/3)
    assert interval(left, right, metric='recall', weighting='user')['estimate'] == .5
    # With seed 9, 37 draws and central 30%, one endpoint lies between mixed
    # and single-user replicates. A wrong event estimator would use .5 here.
    result = interval(left, right, metric='recall', seed=9, replicates=37, confidence=.3)
    assert result['ci_low'] == pytest.approx(2/3)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'candidate', 'user', 'target',
                                   'timestamp', 'flag', 'hash', 'count'])
def test_misalignment_rejected_before_rng_and_before_subgroup(monkeypatch, mutation):
    left = per_query_metrics(fixture(), k=2)
    right = deepcopy(left)
    if mutation == 'missing':
        right.pop()
    elif mutation == 'duplicate':
        right.append(right[0])
    else:
        row = right[-1]  # novel row must be checked even for a repeat-only interval
        field, value = {
            'candidate': ('candidate_set_sha256', 'f'*64), 'user': ('user_id', 'changed'),
            'target': ('target_item_id', 'changed'), 'timestamp': ('timestamp', 999),
            'flag': ('is_repeat_target', True), 'hash': ('query_identity_sha256', 'f'*64),
            'count': ('n_candidates', 10)}[mutation]
        row[field] = value
        if mutation in ('user', 'target', 'timestamp', 'flag'):
            row['query_identity_sha256'] = digest({key: row[key] for key in (
                'query_id', 'user_id', 'target_item_id', 'timestamp', 'is_repeat_target')})
    def forbidden(*args, **kwargs):
        raise AssertionError('RNG must not be initialized before identity validation')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    with pytest.raises(WorkflowError) as error:
        interval(left, right, subgroup='repeat')
    assert error.value.code == 'IDENTITY'


@pytest.mark.parametrize('options', [dict(metric='ndcg@2'), dict(weighting='queries'),
    dict(subgroup='unknown'), dict(replicates=0), dict(replicates=True),
    dict(seed=-1), dict(seed=1.5), dict(confidence=0), dict(confidence=1),
    dict(confidence=float('nan')), dict(confidence=True)])
def test_invalid_bootstrap_options(options):
    rows = per_query_metrics(fixture(), k=2)
    with pytest.raises(WorkflowError) as error:
        interval(rows, rows, **options)
    assert error.value.code == 'CONTRACT'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1.1, True, '1'])
def test_invalid_metric_rows(value):
    rows = per_query_metrics(fixture(), k=2)
    rows[0]['ndcg'] = value
    with pytest.raises(WorkflowError):
        aggregate(rows, metric='ndcg', weighting='event')


def test_invalid_aggregate_options_and_duplicate_rows():
    rows = per_query_metrics(fixture(), k=2)
    for options in (dict(metric='x', weighting='event'), dict(metric='recall', weighting='x'),
                    dict(metric='recall', weighting='event', subgroup='x')):
        with pytest.raises(WorkflowError):
            aggregate(rows, **options)
    with pytest.raises(WorkflowError):
        aggregate(rows + rows[:1], metric='recall', weighting='event')
