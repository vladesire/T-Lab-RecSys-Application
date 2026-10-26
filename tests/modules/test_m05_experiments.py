"""Isolated M05 tests: M01/02/03/04/15 are signature-compatible fakes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jsonschema
import numpy as np
import pytest

from researchflow import experiments as m
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
STAMP = '2026-10-25T10:00:00+00:00'


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(kind, value):
    schema = read(ROOT / 'src/researchflow/schemas' / (kind + '.json'))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)


def contained(root, relative):
    if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
        raise WorkflowError('CONTRACT', 'Unsafe path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise WorkflowError('CONTRACT', 'Escaping path')
    return path


def verify(bundle):
    manifest = read(bundle / 'manifest.json')
    validate('run_manifest', manifest)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    assert actual == {f['path'] for f in manifest['files']} | {'manifest.json'}
    for item in manifest['files']:
        path = contained(bundle, item['path'])
        assert not path.is_symlink()
        assert path.stat().st_size == item['size_bytes']
        assert file_hash(path) == item['sha256']
    return manifest


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'project with spaces'
    root.mkdir()
    for rel, content in [('config.json', '{}'), ('runner.py', '# synthetic runner'),
                         ('research/H.md', '# synthetic research'), ('data.bin', 'fixture')]:
        path = root / rel
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    write(root / 'data-manifest.json', {'id': 'D-fixture', 'sha256': file_hash(root / 'data.bin')})
    resolved = read(ROOT / 'tests/fixtures/schema_examples/resolved_experiment.json')
    resolved['runtime'].update(project_root=str(root), command=[sys.executable, 'runner.py'],
                               code_paths=['runner.py'], research_paths=['research/H.md'],
                               data_refs=[{'id': 'D-fixture', 'manifest': 'data-manifest.json',
                                           'local_path': 'data.bin'}])
    resolved['source_files'] = ['config.json']
    resolved['identity'] = {'scientific_sha256': digest(resolved['scientific']),
                            'code_sha256': file_hash(root / 'runner.py'),
                            'data_sha256': file_hash(root / 'data-manifest.json')}
    calls = []
    state = {'resolved': resolved, 'calls': calls, 'mode': 'ok', 'process_status': 'succeeded',
             'exit_code': 0, 'gate_calls': []}

    def resolve_experiment(project_root: Path, config_path: Path) -> dict:
        assert project_root == root
        assert config_path == root / 'config.json'
        return deepcopy(state['resolved'])

    def capture_provenance(project_root: Path, resolved, output_dir: Path) -> dict:
        files = []
        for rel in ['config.json', 'runner.py', 'research/H.md', 'data-manifest.json']:
            target = output_dir / rel
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(project_root / rel, target)
            files.append(dict(path=rel, sha256=file_hash(target), size_bytes=target.stat().st_size))
        if state['mode'] == 'corrupt_snapshot':
            (output_dir / 'runner.py').write_text('corrupt')
        return dict(schema_version=1, identity=deepcopy(resolved['identity']), files=files,
                    environment={'note': 'Synthetic dependency fake'})

    def begin_bundle(output_root: Path, bundle_id: str) -> Path:
        output_root.mkdir(exist_ok=True)
        stage = output_root / ('.' + bundle_id + '.stage')
        if (output_root / bundle_id).exists() or stage.exists():
            raise WorkflowError('IDENTITY', 'Bundle ID collision')
        stage.mkdir()
        return stage

    def commit_bundle(stage_dir: Path, final_dir: Path, manifest) -> Path:
        assert not final_dir.exists()
        validate('run_manifest', manifest)
        write(stage_dir / 'manifest.json', manifest)
        stage_dir.rename(final_dir)
        verify(final_dir)
        return final_dir

    def check_final_gate(freeze_path: Path, resolved) -> dict:
        state['gate_calls'].append(freeze_path)
        freeze = read(freeze_path)
        validate('study_freeze', freeze)
        if not any(x['identity'] == resolved['identity'] and
                   x['experiment_id'] == resolved['scientific']['experiment_id'] for x in freeze['configs']):
            raise WorkflowError('IDENTITY', 'Freeze mismatch')
        return freeze

    def run_process(argv, cwd: Path, timeout_seconds: float, stdout_path: Path,
                    stderr_path: Path, env=None, *, stop_file=None, memory_limit_bytes=None) -> dict:
        calls.append(list(argv))
        assert cwd == root and timeout_seconds == 30 and memory_limit_bytes is None
        request_path, payload = Path(argv[-3]), Path(argv[-1])
        assert argv[-4] == '--request' and argv[-2] == '--output'
        assert request_path.is_absolute() and payload.is_absolute()
        assert list(payload.iterdir()) == []
        request = read(request_path)
        validate('runner_request', request)
        assert request['identity'] == resolved['identity']
        assert request['scientific'] == resolved['scientific']
        assert read(request_path.parent / 'resolved_config.json') == resolved
        stdout_path.write_text('synthetic stdout')
        stderr_path.write_text('synthetic stderr')
        mode = state['mode']
        if mode == 'exception':
            raise OSError('synthetic launch failure')
        if mode == 'external':
            completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=5)
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            state['exit_code'] = completed.returncode
        elif mode != 'missing':
            result = dict(schema_version=1, kind='metrics', predictions=[], metrics=[],
                          coverage=[], costs=[], checks={'passed': True, 'items': []},
                          limitations=['Synthetic fixture only'], artifacts=[])
            if mode in ('missing_artifact', 'traversal', 'absolute', 'symlink', 'directory'):
                artifact = {'missing_artifact': 'absent.txt', 'traversal': '../stdout.log',
                            'absolute': str(root / 'data.bin'), 'symlink': 'link',
                            'directory': 'directory'}[mode]
                result['artifacts'] = [dict(path=artifact, kind='text', description='Synthetic')]
                if mode == 'symlink':
                    (payload / 'link').symlink_to(root / 'data.bin')
                if mode == 'directory':
                    (payload / 'directory').mkdir()
            if mode in ('ranking', 'bad_ranking', 'wrong_identity', 'pickle'):
                result['kind'] = 'ranking'
                arrays = dict(query_id=np.array(['q', 'q']), user_id=np.array(['u', 'u']),
                              item_id=np.array(['a', 'b']), timestamp=np.array([3, 3]),
                              label=np.array([1, 0]), score=np.array([.1, .2]), tie_key=np.array([0, 1]),
                              is_familiar=np.array([0, 1]), is_repeat_target=np.array([0, 0]))
                if mode == 'bad_ranking':
                    arrays['score'][0] = np.nan
                if mode == 'pickle':
                    arrays['query_id'] = np.array(['q', 'q'], dtype=object)
                np.savez(payload / 'pred.npz', **arrays)
                candidates = [{'item_id': 'a', 'is_familiar': False, 'tie_key': 0},
                              {'item_id': 'b', 'is_familiar': True, 'tie_key': 1}]
                identities = dict(protocol_sha256=digest(request['scientific']['protocol']),
                                  query_set_sha256=digest([dict(query_id='q', user_id='u',
                                      target_item_id='a', timestamp=3, is_repeat_target=False)]),
                                  candidate_set_sha256=digest([dict(query_id='q',
                                      candidate_set_sha256=digest(candidates))]))
                if mode == 'wrong_identity':
                    identities['query_set_sha256'] = 'a' * 64
                result['predictions'] = [dict(id='pred', method_id='m', protocol_id='P-fixture',
                    split='development', format='ranking_npz_v1', path='pred.npz',
                    feature_count=None, identity=identities)]
            if mode == 'failed_checks':
                result['checks']['passed'] = False
            if mode == 'contradictory_checks':
                result['checks']['items'] = [dict(id='bad', status='failed', detail='bad', evidence=[])]
            if mode == 'null_metric':
                result['metrics'] = [read(ROOT / 'tests/fixtures/schema_examples/metric_row.json')]
                result['metrics'][0]['value'] = None
            if mode == 'malformed':
                (payload / 'result.json').write_text('{invalid')
            else:
                write(payload / 'result.json', result)
            if mode == 'schema':
                write(payload / 'result.json', {'kind': 'metrics'})
        if mode == 'bad_process':
            return {'status': 'succeeded'}
        if mode == 'mutate_snapshot':
            (request_path.parent / 'snapshot/runner.py').write_text('changed during execution')
        if mode == 'mutate_inputs':
            state['resolved']['identity']['code_sha256'] = 'f' * 64
        return dict(schema_version=1, argv=list(argv), status=state['process_status'],
                    exit_code=state['exit_code'], started_at=STAMP, finished_at=STAMP,
                    elapsed_seconds=0., peak_rss_bytes=None, warnings=[])

    for name, fake in dict(load_document=read, atomic_write_json=write, sha256_file=file_hash,
                           digest_document=digest, validate_document=validate, contained_path=contained,
                           resolve_experiment=resolve_experiment, capture_provenance=capture_provenance,
                           begin_bundle=begin_bundle, commit_bundle=commit_bundle,
                           check_final_gate=check_final_gate, run_process=run_process).items():
        monkeypatch.setattr(m, name, fake)
    state['root'] = root
    return state


def execute(env, **kwargs):
    return m.execute_experiment(env['root'], Path('config.json'), **kwargs)


def test_success_snapshots_request_inventory_and_paths(env, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bundle = execute(env, run_id='attempt-1')
    manifest = verify(bundle)
    assert bundle == env['root'] / 'runs/attempt-1'
    assert manifest['status'] == 'succeeded'
    assert manifest['metadata']['identity'] == env['resolved']['identity']
    assert manifest['metadata']['result_path'] == 'payload/result.json'
    for rel in ['config.json', 'runner.py', 'research/H.md', 'data-manifest.json']:
        assert (bundle / 'snapshot' / rel).read_bytes() == (env['root'] / rel).read_bytes()
    request = read(bundle / 'runner_request.json')
    assert request['data'][0]['local_path'] == str(env['root'] / 'data.bin')
    assert request['data'][0]['manifest'] == read(env['root'] / 'data-manifest.json')
    assert read(bundle / 'payload/result.json')['metrics'] == []
    assert read(bundle / 'process_report.json')['exit_code'] == 0


@pytest.mark.parametrize('mode', ['missing', 'malformed', 'schema', 'missing_artifact', 'traversal',
    'absolute', 'symlink', 'directory', 'failed_checks', 'contradictory_checks', 'bad_ranking',
    'wrong_identity', 'pickle', 'exception', 'bad_process', 'corrupt_snapshot',
    'mutate_snapshot', 'mutate_inputs'])
def test_failure_commits_inspectable_evidence(env, mode):
    env['mode'] = mode
    with pytest.raises(WorkflowError) as caught:
        execute(env, run_id='failed-attempt')
    bundle = env['root'] / 'runs/failed-attempt'
    assert caught.value.code == 'EXECUTION'
    assert caught.value.details['bundle_path'] == str(bundle)
    assert str(bundle) in str(caught.value)
    manifest = verify(bundle)
    assert manifest['status'] == 'failed'
    assert manifest['metadata']['result_path'] is None
    assert not read(bundle / 'checks.json')['passed']
    assert (bundle / 'process_report.json').is_file()
    if mode == 'corrupt_snapshot':
        assert not env['calls']
    else:
        assert (bundle / 'stderr.log').read_text() == 'synthetic stderr'


@pytest.mark.parametrize('status,exit_code,terminal', [('failed', 1, 'failed'),
    ('timed_out', None, 'failed'), ('memory_exceeded', None, 'failed'),
    ('cancelled', None, 'cancelled'), ('succeeded', 2, 'failed')])
def test_process_status_wins_over_valid_payload(env, status, exit_code, terminal):
    env.update(process_status=status, exit_code=exit_code)
    with pytest.raises(WorkflowError):
        execute(env, run_id='process-failure')
    bundle = env['root'] / 'runs/process-failure'
    manifest = verify(bundle)
    assert manifest['status'] == terminal
    assert manifest['metadata']['process_status'] == status
    assert read(bundle / 'process_report.json')['status'] == status


def test_collision_cannot_overwrite_and_new_attempt_is_distinct(env):
    first = execute(env, run_id='same')
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    with pytest.raises(WorkflowError) as caught:
        execute(env, run_id='same')
    assert caught.value.code == 'IDENTITY'
    assert len(env['calls']) == 1
    assert before == {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    assert execute(env) != execute(env)


@pytest.mark.parametrize('identifier', ['', '../escape', '/absolute', 'has space'])
def test_bad_id_never_stages_or_launches(env, identifier):
    with pytest.raises(WorkflowError):
        execute(env, run_id=identifier)
    assert not env['calls'] and not (env['root'] / 'runs').exists()


def final_setup(env):
    env['resolved']['scientific']['phase'] = 'final'
    env['resolved']['identity']['scientific_sha256'] = digest(env['resolved']['scientific'])
    freeze = dict(schema_version=1, freeze_id='F-fixture', created_at=STAMP,
                  configs=[dict(experiment_id='E-fixture', identity=deepcopy(env['resolved']['identity']))])
    write(env['root'] / 'freeze.json', freeze)
    return freeze


@pytest.mark.parametrize('case', ['missing', 'mismatch', 'invalid'])
def test_final_gate_blocks_before_staging(env, case):
    freeze = final_setup(env)
    if case == 'mismatch':
        freeze['configs'][0]['identity']['code_sha256'] = 'f' * 64
        write(env['root'] / 'freeze.json', freeze)
    if case == 'invalid':
        write(env['root'] / 'freeze.json', {})
    with pytest.raises((WorkflowError, jsonschema.ValidationError)):
        execute(env, freeze_path=None if case == 'missing' else Path('freeze.json'))
    assert not env['calls'] and not (env['root'] / 'runs').exists()


def test_matching_freeze_is_snapshotted_and_checked_before_launch(env):
    freeze = final_setup(env)
    bundle = execute(env, freeze_path=Path('freeze.json'))
    assert verify(bundle)['status'] == 'succeeded'
    assert read(bundle / 'study_freeze.json') == freeze
    assert len(env['gate_calls']) == 2


@pytest.mark.parametrize('mode', ['ranking', 'null_metric'])
def test_ranking_identity_and_missing_metric_preserved(env, mode):
    env['mode'] = mode
    bundle = execute(env)
    assert verify(bundle)['status'] == 'succeeded'
    if mode == 'null_metric':
        assert read(bundle / 'payload/result.json')['metrics'][0]['value'] is None


def test_prelaunch_configuration_change_is_failed_without_launch(env, monkeypatch):
    original = m.resolve_experiment
    calls = 0
    def changed(project_root, config_path):
        nonlocal calls
        value = original(project_root, config_path)
        calls += 1
        if calls > 1:
            value['identity']['code_sha256'] = 'f' * 64
        return value
    monkeypatch.setattr(m, 'resolve_experiment', changed)
    with pytest.raises(WorkflowError):
        execute(env, run_id='drift')
    assert not env['calls']
    assert verify(env['root'] / 'runs/drift')['status'] == 'failed'


def test_external_runner_uses_only_file_protocol(env):
    # M04 remains a fake, but this case actually launches an independent script.
    (env['root'] / 'runner.py').write_text('''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.request).read_text())
assert r['run_id']=='external'
Path(a.output,'observed.json').write_text(json.dumps(r))
result={'schema_version':1,'kind':'metrics','predictions':[],'metrics':[],
'coverage':[],'costs':[],'checks':{'passed':True,'items':[]},'limitations':[],
'artifacts':[{'path':'observed.json','kind':'request','description':'Synthetic echo'}]}
Path(a.output,'result.json').write_text(json.dumps(result))
''')
    env['resolved']['identity']['code_sha256'] = file_hash(env['root'] / 'runner.py')
    env['mode'] = 'external'
    bundle = execute(env, run_id='external')
    assert verify(bundle)['status'] == 'succeeded'
    assert read(bundle / 'payload/observed.json') == read(bundle / 'runner_request.json')
