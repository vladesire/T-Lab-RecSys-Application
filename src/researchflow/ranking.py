"""Single-positive ranking metrics and user-cluster paired bootstrap.

All identities follow ranking_npz_v1. Bootstrap estimates are left minus right;
users and queries are sorted by normalized ID so input order cannot change draws.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from researchflow.contracts import WorkflowError


_FIELDS = ('query_id', 'user_id', 'item_id', 'timestamp', 'label', 'score',
           'tie_key', 'is_familiar', 'is_repeat_target')


def _fail(message: str, *, identity: bool = False) -> None:
    raise WorkflowError('IDENTITY' if identity else 'CONTRACT', message)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        _fail(f'{name} must be an integer >= {minimum}')
    return int(value)


def _id(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, np.str_, Integral)):
        _fail('Identifiers must be Unicode strings or integers')
    result = str(value)
    try:
        result.encode('utf-8')
    except UnicodeError:
        _fail('Identifiers must be valid UTF-8')
    return result


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _query_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ('query_id', 'user_id', 'target_item_id',
                                     'timestamp', 'is_repeat_target')}


def per_query_metrics(arrays: Mapping[str, Any], *, k: int) -> list[dict[str, Any]]:
    """Validate candidate arrays and return one deterministic row per query."""
    k = _integer(k, 'k', 1)
    if not isinstance(arrays, Mapping):
        _fail('arrays must be a mapping')
    values = {}
    for name in _FIELDS:
        if name not in arrays:
            _fail(f'Missing array: {name}')
        try:
            value = np.asarray(arrays[name])
        except (ValueError, TypeError):
            _fail(f'{name} must be a one-dimensional array')
        if value.ndim != 1:
            _fail(f'{name} must be one-dimensional')
        allowed = 'Uiu' if name.endswith('_id') else (
            'iuf' if name == 'score' else 'biu' if name in
            ('label', 'is_familiar', 'is_repeat_target') else 'iu')
        if value.dtype.kind not in allowed:
            _fail(f'Invalid dtype for {name}: {value.dtype}')
        values[name] = value
    if len({len(value) for value in values.values()}) != 1:
        _fail('All arrays must have equal lengths')
    if not np.all(np.isfinite(values['score'])):
        _fail('Scores must be finite')
    for name in ('label', 'is_familiar', 'is_repeat_target'):
        if not np.all((values[name] == 0) | (values[name] == 1)):
            _fail(f'{name} must contain only boolean/0-or-1 values')
    if np.any(values['timestamp'] < 0):
        _fail('Timestamps must be nonnegative')
    if any(int(v) < 0 or int(v) > 2**64 - 1 for v in values['tie_key']):
        _fail('tie_key must be in the uint64 range')
    for name in ('query_id', 'user_id', 'item_id'):
        values[name] = [_id(v) for v in values[name]]
    groups: dict[str, list[int]] = {}
    for i, query in enumerate(values['query_id']):
        groups.setdefault(query, []).append(i)
    result = []
    for query in sorted(groups):
        indices = groups[query]
        for name in ('user_id', 'timestamp', 'is_repeat_target'):
            if len({values[name][i].item() if isinstance(values[name][i], np.generic)
                    else values[name][i] for i in indices}) != 1:
                _fail(f'Inconsistent {name} within query {query}')
        if len({values['item_id'][i] for i in indices}) != len(indices):
            _fail(f'Duplicate candidate item in query {query}')
        positives = [i for i in indices if values['label'][i]]
        if len(positives) != 1:
            _fail(f'Query {query} must have exactly one positive')
        positive = positives[0]
        repeat = bool(values['is_repeat_target'][positive])
        if bool(values['is_familiar'][positive]) != repeat:
            _fail(f'Positive familiarity disagrees with target flag in query {query}')
        # Python scalar negation avoids overflow at signed/unsigned integer limits.
        order = sorted(indices, key=lambda i: (-values['score'][i].item(),
                       int(values['tie_key'][i]), values['item_id'][i]))
        rank = order.index(positive) + 1
        row = dict(query_id=query, user_id=values['user_id'][positive],
                   target_item_id=values['item_id'][positive],
                   timestamp=int(values['timestamp'][positive]), is_repeat_target=repeat)
        candidates = [dict(item_id=values['item_id'][i],
                           is_familiar=bool(values['is_familiar'][i]),
                           tie_key=int(values['tie_key'][i]))
                      for i in sorted(indices, key=lambda i: values['item_id'][i])]
        row.update(query_identity_sha256=_hash(row), candidate_set_sha256=_hash(candidates),
                   ndcg=1.0 / math.log2(rank + 1) if rank <= k else 0.0,
                   recall=float(rank <= k), n_candidates=len(indices),
                   n_familiar_alternatives=sum(bool(values['is_familiar'][i])
                                               for i in indices if i != positive))
        result.append(row)
    return result


def _options(metric: str, weighting: str, subgroup: str) -> None:
    if metric not in ('ndcg', 'recall'):
        _fail('metric must be ndcg or recall')
    if weighting not in ('event', 'user'):
        _fail('weighting must be event or user')
    if subgroup not in ('all', 'repeat', 'novel'):
        _fail('subgroup must be all, repeat or novel')


def _rows(rows: Sequence[Mapping[str, Any]], metric: str, *, paired: bool = False) -> dict[str, dict[str, Any]]:
    result = {}
    for original in rows:
        required = ['query_id', 'user_id', 'is_repeat_target', metric]
        if paired:
            required += ['target_item_id', 'timestamp', 'query_identity_sha256',
                         'candidate_set_sha256', 'n_candidates', 'n_familiar_alternatives']
        if not isinstance(original, Mapping) or any(key not in original for key in required):
            _fail('Incomplete per-query row')
        row = dict(original)
        for key in ('query_id', 'user_id'):
            row[key] = _id(row[key])
        query = row['query_id']
        if query in result:
            _fail(f'Duplicate query row: {query}', identity=True)
        flag = row['is_repeat_target']
        if not isinstance(flag, (bool, np.bool_, Integral)) or flag not in (0, 1):
            _fail('is_repeat_target must be boolean/0-or-1')
        row['is_repeat_target'] = bool(flag)
        value = row[metric]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            _fail(f'{metric} must be finite and in [0, 1]')
        row[metric] = float(value)
        if paired:
            row['target_item_id'] = _id(row['target_item_id'])
            row['timestamp'] = _integer(row['timestamp'], 'timestamp')
            n = _integer(row['n_candidates'], 'n_candidates', 1)
            familiar = _integer(row['n_familiar_alternatives'], 'n_familiar_alternatives')
            if familiar >= n:
                _fail('Familiar alternatives must exclude the positive')
            for key in ('query_identity_sha256', 'candidate_set_sha256'):
                digest = row[key]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    _fail(f'Invalid {key}', identity=True)
            if row['query_identity_sha256'] != _hash(_query_record(row)):
                _fail('Query identity hash disagrees with row', identity=True)
        result[query] = row
    return result


def _selected(row: Mapping[str, Any], subgroup: str) -> bool:
    return subgroup == 'all' or row['is_repeat_target'] == (subgroup == 'repeat')


def _totals(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    by_user: dict[str, list[float]] = {}
    for row in rows:
        by_user.setdefault(row['user_id'], []).append(row[metric])
    users = sorted(by_user)
    totals = np.array([math.fsum(by_user[u]) for u in users], dtype=float)
    counts = np.array([len(by_user[u]) for u in users], dtype=np.int64)
    return users, totals, counts


def aggregate(rows: Sequence[Mapping[str, Any]], *, metric: str, weighting: str, subgroup: str = 'all') -> dict[str, Any]:
    """Average events or average within-user means in the chosen subgroup."""
    _options(metric, weighting, subgroup)
    validated = _rows(rows, metric)
    chosen = [validated[q] for q in sorted(validated) if _selected(validated[q], subgroup)]
    users, totals, counts = _totals(chosen, metric)
    value = None
    if chosen:
        value = float(totals.sum() / counts.sum() if weighting == 'event'
                      else np.mean(totals / counts))
    return dict(value=value, n_queries=len(chosen), n_users=len(users))


def paired_interval(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]], *, metric: str, weighting: str, subgroup: str, replicates: int, seed: int, confidence: float = 0.95) -> dict[str, Any]:
    """Percentile interval for a paired left-minus-right user-cluster bootstrap."""
    _options(metric, weighting, subgroup)
    replicates = _integer(replicates, 'replicates', 1)
    seed = _integer(seed, 'seed')
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real) or not math.isfinite(confidence) or not 0 < confidence < 1:
        _fail('confidence must be finite and strictly between zero and one')
    left, right = _rows(left_rows, metric, paired=True), _rows(right_rows, metric, paired=True)
    if left.keys() != right.keys():
        _fail('Paired query sets differ', identity=True)
    chosen = []
    for query in sorted(left):
        a, b = left[query], right[query]
        for key in (*_query_record(a), 'query_identity_sha256', 'candidate_set_sha256',
                    'n_candidates', 'n_familiar_alternatives'):
            if a[key] != b[key]:
                _fail(f'Paired identity mismatch for {query}: {key}', identity=True)
        if _selected(a, subgroup):
            chosen.append(dict(a, **{metric: a[metric] - b[metric]}))
    if not chosen:
        _fail('Paired subgroup is empty')
    users, totals, counts = _totals(chosen, metric)
    means = totals / counts
    estimate = float(totals.sum() / counts.sum() if weighting == 'event' else means.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sampled = rng.integers(0, len(users), size=len(users))
        draws[i] = (totals[sampled].sum() / counts[sampled].sum()
                    if weighting == 'event' else means[sampled].mean())
    low, high = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2], method='linear')
    return dict(estimate=estimate, ci_low=float(low), ci_high=float(high),
                n_queries=len(chosen), n_users=len(users))
