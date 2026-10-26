"""Configured, immutable experiment attempts using the external runner port."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import stat
import time
from typing import Any
from uuid import uuid4

import numpy as np

from researchflow.bundles import begin_bundle, commit_bundle
from researchflow.configuration import capture_provenance, resolve_experiment
from researchflow.contracts import WorkflowError
from researchflow.documents import (
    atomic_write_json, contained_path, digest_document, load_document,
    sha256_file, validate_document,
)
from researchflow.freeze import check_final_gate
from researchflow.processes import run_process


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input(root: Path, path: Path | str) -> Path:
    return (root / path).resolve()


def _file(root: Path, relative: str) -> Path:
    """Require an ordinary file, including no symlink in its path."""
    target = contained_path(root, relative)
    lexical = root / relative
    if any(p.is_symlink() for p in (lexical, *lexical.parents) if p.is_relative_to(root)):
        raise WorkflowError('CONTRACT', 'Symlink artifact is not immutable', {'path': relative})
    if not target.is_file():
        raise WorkflowError('CONTRACT', 'Declared artifact is missing or not a file', {'path': relative})
    return target


def _ranking(path: Path, descriptor: dict[str, Any], protocol: Any) -> None:
    """Validate the port's arrays and identities without computing any scores."""
    names = ('query_id', 'user_id', 'item_id', 'timestamp', 'label', 'score',
             'tie_key', 'is_familiar', 'is_repeat_target')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names}
    n = len(arrays['query_id'])
    if any(a.ndim != 1 or len(a) != n for a in arrays.values()):
        raise ValueError('Ranking arrays must be equal-length and one-dimensional')
    for name in ('query_id', 'user_id', 'item_id'):
        if arrays[name].dtype.kind not in 'Uiu':
            raise ValueError(f'{name} must use Unicode or integer identifiers')
    for name in ('label', 'is_familiar', 'is_repeat_target'):
        if arrays[name].dtype.kind not in 'biu' or not np.isin(arrays[name], [0, 1]).all():
            raise ValueError(f'{name} must be boolean or binary integer')
    for name in ('timestamp', 'tie_key'):
        if arrays[name].dtype.kind not in 'iu' or (arrays[name] < 0).any():
            raise ValueError(f'{name} must be a nonnegative integer')
    if any(int(x) > 2**64 - 1 for x in arrays['tie_key']):
        raise ValueError('tie_key exceeds uint64')
    if arrays['score'].dtype.kind not in 'iuf' or not np.isfinite(arrays['score']).all():
        raise ValueError('score must be finite real numeric')
    groups: dict[str, list[int]] = {}
    for i, query in enumerate(arrays['query_id']):
        groups.setdefault(str(query), []).append(i)
    queries, candidates = [], []
    for query, ix in sorted(groups.items()):
        items = [str(arrays['item_id'][i]) for i in ix]
        positives = [i for i in ix if arrays['label'][i]]
        if len(set(items)) != len(items) or len(positives) != 1:
            raise ValueError('Each query needs distinct items and exactly one positive')
        for name in ('user_id', 'timestamp', 'is_repeat_target'):
            if len({str(arrays[name][i]) for i in ix}) != 1:
                raise ValueError(f'{name} must be constant within a query')
        pos = positives[0]
        if arrays['is_familiar'][pos] != arrays['is_repeat_target'][pos]:
            raise ValueError('Positive familiarity differs from repeat target')
        queries.append(dict(query_id=query, user_id=str(arrays['user_id'][pos]),
                            target_item_id=str(arrays['item_id'][pos]),
                            timestamp=int(arrays['timestamp'][pos]),
                            is_repeat_target=bool(arrays['is_repeat_target'][pos])))
        records = sorted((dict(item_id=str(arrays['item_id'][i]),
                               is_familiar=bool(arrays['is_familiar'][i]),
                               tie_key=int(arrays['tie_key'][i])) for i in ix),
                         key=lambda row: row['item_id'])
        candidates.append(dict(query_id=query, candidate_set_sha256=digest_document(records)))
    expected = dict(protocol_sha256=digest_document(protocol),
                    query_set_sha256=digest_document(queries),
                    candidate_set_sha256=digest_document(candidates))
    if descriptor['identity'] != expected:
        raise ValueError('Ranking descriptor identity does not match its arrays/protocol')


def _result(payload: Path, protocol: Any) -> dict[str, Any]:
    result = load_document(_file(payload, 'result.json'))
    validate_document('runner_result', result)
    for artifact in result['artifacts']:
        _file(payload, artifact['path'])
    ids = [p['id'] for p in result['predictions']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prediction IDs')
    for prediction in result['predictions']:
        _ranking(_file(payload, prediction['path']), prediction, protocol)
    if not result['checks']['passed'] or any(
        item['status'] == 'failed' for item in result['checks']['items']
    ):
        raise ValueError('Runner reported failed checks')
    return result


def _inventory(stage: Path, checks: dict[str, Any]) -> list[dict[str, Any]]:
    # Record and remove unsafe entries so M03 can commit self-contained evidence.
    for path in sorted(stage.rglob('*')):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            checks['passed'] = False
            checks['items'].append({'id': 'unsafe_output', 'status': 'failed',
                                    'detail': f'{path.relative_to(stage)}: mode={mode}; '
                                    f'link={str(path.readlink()) if path.is_symlink() else None}',
                                    'evidence': []})
            path.unlink()
    atomic_write_json(stage / 'checks.json', checks)
    return [dict(path=p.relative_to(stage).as_posix(), sha256=sha256_file(p),
                 size_bytes=p.stat().st_size)
            for p in sorted(stage.rglob('*')) if p.is_file() and p != stage / 'manifest.json']


def execute_experiment(project_root: Path, config_path: Path, *, run_id: str | None = None, freeze_path: Path | None = None) -> Path:
    root = Path(project_root).resolve()
    config = _input(root, config_path)
    resolved = resolve_experiment(root, config)
    validate_document('resolved_experiment', resolved)
    identifier = run_id if run_id is not None else f'run-{uuid4().hex}'
    if not isinstance(identifier, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier) is None:
        raise WorkflowError('CONTRACT', 'Invalid run ID', {'run_id': identifier})
    scientific, runtime, identity = (resolved[k] for k in ('scientific', 'runtime', 'identity'))
    frozen = None
    if scientific['phase'] == 'final':
        if freeze_path is None:
            raise WorkflowError('CONTRACT', 'Final execution requires a study freeze')
        freeze = _input(root, freeze_path)
        check_final_gate(freeze, resolved)  # Fail before staging or launching.
        frozen = load_document(freeze)
        validate_document('study_freeze', frozen)
    output_root = root / 'runs'
    final = output_root / identifier
    stage = begin_bundle(output_root, identifier).resolve()
    started, clock = _now(), time.monotonic()
    checks: dict[str, Any] = {'passed': True, 'items': []}
    argv = [*runtime['command'], '--request', str(stage / 'runner_request.json'),
            '--output', str(stage / 'payload')]
    process = None
    valid_result = False
    failure = None
    try:
        atomic_write_json(stage / 'resolved_config.json', resolved)
        snapshot = stage / 'snapshot'
        snapshot.mkdir()
        provenance = capture_provenance(root, resolved, snapshot)
        atomic_write_json(stage / 'provenance.json', provenance)
        validate_document('provenance_record', provenance)
        if provenance['identity'] != identity:
            raise WorkflowError('IDENTITY', 'Snapshot identity differs from resolved configuration')
        for item in provenance['files']:
            file = _file(snapshot, item['path'])
            if file.stat().st_size != item['size_bytes'] or sha256_file(file) != item['sha256']:
                raise WorkflowError('IDENTITY', 'Snapshot file differs from provenance', item)
        data = [dict(id=ref['id'],
                     local_path=str(_input(root, ref['local_path'])) if ref['local_path'] is not None else None,
                     manifest=load_document(_input(root, ref['manifest'])))
                for ref in runtime['data_refs']]
        request = dict(schema_version=1, run_id=identifier, scientific=scientific,
                       data=data, identity=identity)
        validate_document('runner_request', request)
        atomic_write_json(stage / 'runner_request.json', request)
        if frozen is not None:
            atomic_write_json(stage / 'study_freeze.json', frozen)
            gate = check_final_gate(stage / 'study_freeze.json', resolved)
            atomic_write_json(stage / 'final_gate.json', gate)
        # Reject changes between resolution, snapshotting and request construction.
        if resolve_experiment(root, config) != resolved:
            raise WorkflowError('IDENTITY', 'Inputs changed while preparing execution')
        protected = {p.relative_to(stage).as_posix(): sha256_file(p)
                     for p in stage.rglob('*') if p.is_file()}
        (stage / 'payload').mkdir()
        process = run_process(argv, root, runtime['resources']['timeout_seconds'],
                              stage / 'stdout.log', stage / 'stderr.log',
                              memory_limit_bytes=runtime['resources']['memory_limit_bytes'])
        atomic_write_json(stage / 'process_response.json', process)
        validate_document('process_report', process)
        for relative, expected in protected.items():
            if sha256_file(_file(stage, relative)) != expected:
                raise WorkflowError('IDENTITY', 'Runner changed captured evidence', {'path': relative})
        if resolve_experiment(root, config) != resolved:
            raise WorkflowError('IDENTITY', 'Inputs changed during execution')
        _result(stage / 'payload', scientific['protocol'])
        valid_result = True
        if process['status'] != 'succeeded' or process['exit_code'] != 0:
            raise WorkflowError('EXECUTION', 'Runner process did not succeed', process)
        checks['items'].append(dict(id='runner_result', status='passed',
                                    detail='Validated response, declared files and ranking identities',
                                    evidence=['payload/result.json', 'process_report.json']))
    except Exception as exc:
        failure = exc
        checks['passed'] = False
        checks['items'].append(dict(id='execution', status='failed',
                                    detail=f'{type(exc).__name__}: {exc}', evidence=[]))
    # Invalid/absent responses never become a successful process report.
    try:
        if process is None:
            raise ValueError('Process report unavailable')
        validate_document('process_report', process)
    except Exception:
        process = dict(schema_version=1, argv=argv, status='failed', exit_code=None,
                       started_at=started, finished_at=_now(),
                       elapsed_seconds=max(0., time.monotonic() - clock),
                       peak_rss_bytes=None,
                       warnings=['Executor-generated report: no valid process report was returned'])
    atomic_write_json(stage / 'process_report.json', process)
    files = _inventory(stage, checks)
    status = ('cancelled' if process['status'] == 'cancelled' else
              'succeeded' if checks['passed'] else 'failed')
    metadata = {k: scientific[k] for k in ('experiment_id', 'branch_id', 'hypothesis_ids', 'phase')}
    metadata.update(origin='fresh_execution', identity=identity,
                    result_path='payload/result.json' if valid_result else None,
                    started_at=started, finished_at=_now(), process_status=process['status'],
                    source_lineage=[])
    manifest = dict(schema_version=1, bundle_id=identifier, bundle_kind='run', status=status,
                    created_at=started, files=files, metadata=metadata)
    validate_document('run_manifest', manifest)
    committed = commit_bundle(stage, final, manifest)
    if status != 'succeeded':
        raise WorkflowError('EXECUTION', f'Experiment {status}; evidence: {committed}',
                            {'bundle_path': str(committed), 'status': status}) from failure
    return committed
