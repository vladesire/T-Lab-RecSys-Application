"""Synthetic isolation tests; no record below attests to real visual inspection."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import jsonschema
import pytest

from researchflow import release as m
from researchflow.contracts import WorkflowError

SCHEMAS = Path(__file__).resolve().parents[2] / 'src/researchflow/schemas'
EXAMPLES = Path(__file__).resolve().parents[1] / 'fixtures/schema_examples'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(kind, value):
    try:
        jsonschema.Draft202012Validator(json.loads((SCHEMAS / (kind + '.json')).read_text()),
                                       format_checker=jsonschema.FormatChecker()).validate(value)
    except jsonschema.ValidationError as exc:
        raise WorkflowError('CONTRACT', 'Invalid document', {'path': list(exc.path)}) from exc


def example(kind):
    return json.loads((EXAMPLES / (kind + '.json')).read_text())


def clean_checks():
    return {'passed': True, 'items': [{'id': 'evidence', 'status': 'passed',
                                     'detail': 'Synthetic isolation test', 'evidence': []}]}


def seal(path, kind, metadata):
    files = [{'path': p.relative_to(path).as_posix(), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
             for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    value = {'schema_version': 1, 'bundle_id': kind + '-test', 'bundle_kind': kind,
             'status': 'succeeded', 'created_at': (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
             'files': files, 'metadata': metadata}
    validate(kind + '_manifest', value)
    write(path / 'manifest.json', value)
    return value


@pytest.fixture
def environment(tmp_path, monkeypatch):
    calls = []
    def contained(root, rel):
        dest = (root / rel).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise WorkflowError('CONTRACT', 'Escape')
        return dest
    def verify(path, expected_kind=None):
        value = json.loads((path / 'manifest.json').read_text())
        validate(value['bundle_kind'] + '_manifest', value)
        assert value['bundle_kind'] == expected_kind
        for entry in value['files']:
            p = contained(path, entry['path'])
            if sha(p) != entry['sha256'] or p.stat().st_size != entry['size_bytes']:
                raise WorkflowError('IDENTITY', 'Tampered input')
        return value
    def begin(output_root, bundle_id):
        stage = output_root / ('.' + bundle_id + '.stage')
        stage.mkdir(parents=True)
        calls.append(stage)
        return stage
    def commit(stage_dir, final_dir, manifest):
        write(stage_dir / 'manifest.json', manifest)
        stage_dir.rename(final_dir)
        return final_dir
    for name, fake in {'load_document': lambda p: json.loads(p.read_text()),
                       'sha256_file': sha, 'atomic_write_json': write,
                       'validate_document': validate, 'contained_path': contained,
                       'verify_bundle': verify, 'bundle_fingerprint': fingerprint,
                       'begin_bundle': begin, 'commit_bundle': commit,
                       'inspect_build': lambda compiled_dir: clean_checks()}.items():
        monkeypatch.setattr(m, name, fake)
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'src').mkdir()
    (root / 'src/science.py').write_text('print("analysis")\n')
    (root / 'pyproject.toml').write_text('[project]\nname="tiny"\n')
    (root / 'uv.lock').write_text('version = 1\n')
    (root / 'README.md').write_text('Run uv sync --frozen, then uv run python src/science.py. Obtain data explicitly.\n')
    analysis_path = root / 'analyses/a'
    result = example('analysis_result')
    result.update(analysis_id='a', inputs=[{'run_id': 'r', 'manifest_sha256': 'c'*64,
                                         'phase': 'final', 'origin': 'fresh_execution'}],
                  phases=['final'], metrics=[], limitations=['One protocol only.'], checks=clean_checks(),
                  comparisons=[{'id': 'main', 'left': 'a', 'right': 'b', 'protocol_id': 'p',
                                'split': 'test', 'subgroup': 'all', 'metric': 'recall@10',
                                'weighting': 'event', 'estimate': 0.1, 'ci_low': None, 'ci_high': None,
                                'n_queries': 10, 'n_users': 5, 'status': 'computed', 'reason': None,
                                'evidence': ['/metrics']}])
    write(analysis_path / 'analysis.json', result)
    ameta = example('analysis_manifest')['metadata']
    ameta.update(analysis_id='a', input_fingerprints=['c'*64], phases=['final'])
    am = seal(analysis_path, 'analysis', ameta)
    snapshot = root / 'snap'
    config = example('publication_config')
    config.update(publication_id='p', mode='release', analyses=[{'alias': 'main', 'path': 'analyses/a',
                                                              'sha256': fingerprint(am)}],
                  required_comparison_ids=['main'])
    write(snapshot / 'publication.json', config)
    write(snapshot / 'source_inventory.json', {'raw_data_dependencies': [
        {'dataset': 'licensed-events', 'acquisition': 'Obtain with permission', 'sha256': 'd'*64}]})
    (snapshot / 'main.tex').write_text('\\documentclass{article}\\begin{document}Evidence.\\end{document}\n')
    pmeta = example('publication_manifest')['metadata']
    pmeta.update(publication_id='p', mode='release', is_fixture=False, phases=['final'],
                 analysis_fingerprints=[fingerprint(am)])
    pm = seal(snapshot, 'publication', pmeta)
    compiled = root / 'compiled'
    compiled.mkdir()
    (compiled / 'main.pdf').write_bytes(b'%PDF-1.7\nsynthetic PDF bytes for unit tests only\n')
    (compiled / 'main.log').write_text('No TeX was run in this test.\n')
    cmeta = example('compiled_manifest')['metadata']
    cmeta.update(snapshot_sha256=fingerprint(pm), pdf_sha256=sha(compiled / 'main.pdf'), checks=clean_checks())
    cm = seal(compiled, 'compiled', cmeta)
    review = {'schema_version': 1, 'reviewer': 'Unit Test Actor',
              'recorded_at': datetime.now(timezone.utc).isoformat(),
              'pdf_sha256': cmeta['pdf_sha256'], 'outcome': 'passed', 'notes': 'Isolated test input only.'}
    write(root / 'review.json', review)
    return {'root': root, 'snapshot': snapshot, 'compiled': compiled, 'review': root / 'review.json',
            'pm': pm, 'cm': cm, 'am': am, 'result': result, 'calls': calls}


def run(e, output=Path('releases'), review=True):
    return m.package_release(e['root'], Path('snap'), Path('compiled'), output,
                             visual_review=Path('review.json') if review else None)


def reseal(e):
    root = e['root']
    e['am'] = seal(root / 'analyses/a', 'analysis', e['am']['metadata'])
    config = json.loads((root / 'snap/publication.json').read_text())
    config['analyses'][0]['sha256'] = fingerprint(e['am'])
    write(root / 'snap/publication.json', config)
    e['pm']['metadata']['analysis_fingerprints'] = [fingerprint(e['am'])]
    e['pm'] = seal(root / 'snap', 'publication', e['pm']['metadata'])
    e['cm']['metadata']['snapshot_sha256'] = fingerprint(e['pm'])
    e['cm'] = seal(root / 'compiled', 'compiled', e['cm']['metadata'])


def test_portable_archive_checksums_evidence_and_dependencies(environment, monkeypatch, tmp_path):
    e = environment
    # Unselected raw/cache/secret files must never be swept into the release.
    (e['root'] / 'raw').mkdir()
    (e['root'] / 'raw/data.csv').write_text('private raw data')
    (e['root'] / '.env').write_text('SECRET=value')
    monkeypatch.chdir(tmp_path)
    out = run(e)
    manifest = json.loads((out / 'manifest.json').read_text())
    validate('release_manifest', manifest)
    assert manifest['metadata']['snapshot_sha256'] == fingerprint(e['pm'])
    assert manifest['files'][0]['sha256'] == sha(out / 'release.tar.gz')
    extracted = tmp_path / 'extracted'
    with tarfile.open(out / 'release.tar.gz') as archive:
        names = archive.getnames()
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        assert all(x.isfile() for x in archive.getmembers())
        archive.extractall(extracted, filter='data')
    checksums = json.loads((extracted / 'CHECKSUMS.json').read_text())['files']
    assert set(names) == {x['path'] for x in checksums} | {'CHECKSUMS.json'}
    for row in checksums:
        assert sha(extracted / row['path']) == row['sha256']
        assert (extracted / row['path']).stat().st_size == row['size_bytes']
    assert (extracted / 'compiled/main.pdf').read_bytes() == (e['compiled'] / 'main.pdf').read_bytes()
    portability = json.loads((extracted / 'portability.json').read_text())
    assert portability['offline_complete'] is False
    assert 'One protocol only.' in portability['limitations']
    assert {x['kind'] for x in portability['external_dependencies']} == {'run_reconstruction', 'source_inventory_dependency'}
    assert not any('.env' in n or 'raw/' in n or 'releases/' in n for n in names)
    with pytest.raises(WorkflowError, match='already exists'):
        run(e)


def test_mismatch_fails_before_staging(environment):
    e = environment
    e['cm']['metadata']['snapshot_sha256'] = 'f'*64
    write(e['compiled'] / 'manifest.json', e['cm'])
    with pytest.raises(WorkflowError) as err:
        run(e)
    assert err.value.code == 'IDENTITY'
    assert not e['calls']


@pytest.mark.parametrize('case', ['fixture', 'draft', 'citation', 'inspection', 'no_items', 'unverified',
                                 'missing_comparison', 'unavailable', 'nonfinal', 'analysis_fixture', 'analysis_check'])
def test_bad_evidence(environment, monkeypatch, case):
    e = environment
    if case in {'fixture', 'draft'}:
        e['pm']['metadata']['is_fixture'] = case == 'fixture'
        e['pm']['metadata']['mode'] = 'draft' if case == 'draft' else 'release'
    elif case in {'citation', 'no_items', 'unverified'}:
        checks = e['cm']['metadata']['checks']
        if case == 'no_items':
            checks['items'] = []
        else:
            checks['items'][0].update(id='citations', status='failed' if case == 'citation' else 'unverified')
    elif case == 'inspection':
        monkeypatch.setattr(m, 'inspect_build', lambda compiled_dir: {'passed': False, 'items': []})
    else:
        result = e['result']
        if case == 'missing_comparison':
            result['comparisons'] = []
        elif case == 'unavailable':
            result['comparisons'][0].update(status='unavailable', estimate=None, reason='No pairing')
        elif case == 'nonfinal':
            result['inputs'][0]['phase'] = 'development'
        elif case == 'analysis_fixture':
            result['phases'] = ['fixture']
        elif case == 'analysis_check':
            result['checks']['items'][0]['status'] = 'failed'
        write(e['root'] / 'analyses/a/analysis.json', result)
    reseal(e)
    with pytest.raises(WorkflowError):
        run(e)
    assert not e['calls']


@pytest.mark.parametrize('case', ['missing', 'wrong_pdf', 'empty_reviewer', 'fixture', 'invalid_date', 'naive', 'future', 'before_build', 'failed'])
def test_visual_review_required_and_exact(environment, case):
    e = environment
    review = json.loads(e['review'].read_text())
    if case == 'wrong_pdf':
        review['pdf_sha256'] = '0'*64
    elif case == 'empty_reviewer':
        review['reviewer'] = '  '
    elif case == 'fixture':
        review['notes'] = 'Synthetic fixture only'
    elif case == 'invalid_date':
        review['recorded_at'] = 'not-a-date'
    elif case == 'naive':
        review['recorded_at'] = '2026-01-01T00:00:00'
    elif case == 'future':
        review['recorded_at'] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    elif case == 'before_build':
        review['recorded_at'] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    elif case == 'failed':
        review['outcome'] = 'failed'
    write(e['review'], review)
    with pytest.raises(WorkflowError):
        run(e, review=case != 'missing')
    assert not e['calls']


@pytest.mark.parametrize('rel', ['snap/releases', 'compiled/releases', 'snap', '.'])
def test_output_overlap_fails_without_recursive_scan(environment, rel):
    with pytest.raises(WorkflowError, match='overlaps'):
        run(environment, Path(rel))
    assert not environment['calls']


@pytest.mark.parametrize('rel', ['../escape.tex', '/absolute.tex', 'a\\b.tex', 'a/../b.tex'])
def test_unsafe_inventory_paths(environment, monkeypatch, rel):
    e = environment
    pm = copy.deepcopy(e['pm'])
    pm['files'].append({'path': rel, 'sha256': 'a'*64, 'size_bytes': 0})
    original = m.verify_bundle
    monkeypatch.setattr(m, 'verify_bundle', lambda p, expected_kind=None:
                        pm if expected_kind == 'publication' else original(p, expected_kind))
    e['cm']['metadata']['snapshot_sha256'] = fingerprint(pm)
    write(e['compiled'] / 'manifest.json', e['cm'])
    with pytest.raises(WorkflowError, match='Unsafe'):
        run(e)


@pytest.mark.parametrize('case', ['symlink', 'private_path', 'secret', 'raw_file', 'no_lock', 'tamper'])
def test_public_archive_boundaries(environment, case):
    e = environment
    root = e['root']
    if case == 'symlink':
        (root / 'src/link.py').symlink_to(root / 'src/science.py')
    elif case == 'private_path':
        (root / 'README.md').write_text('cd /Users/researcher/private/project')
    elif case == 'secret':
        (root / 'src/credentials.py').write_text('api_key = "do-not-package-this"')
    elif case == 'raw_file':
        (root / 'snap/raw').mkdir()
        (root / 'snap/raw/events.csv').write_text('events')
        reseal(e)
    elif case == 'no_lock':
        (root / 'uv.lock').unlink()
    elif case == 'tamper':
        (root / 'compiled/main.pdf').write_bytes(b'%PDF-corrupted')
    with pytest.raises(WorkflowError):
        run(e)
    assert not e['calls']


def test_copy_failure_cleans_stage_without_success(environment, monkeypatch):
    e = environment
    def fail(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(m.shutil, 'copyfile', fail)
    with pytest.raises(WorkflowError, match='packaging failed') as err:
        run(e)
    assert err.value.code == 'EXECUTION'
    assert e['calls'] and not e['calls'][0].exists()
    assert not list((e['root'] / 'releases').glob('release-*'))


def test_output_in_source_tree_is_never_selected(environment):
    e = environment
    output = e['root'] / 'src/release-output'
    output.mkdir()
    (output / 'old.tar.gz').write_bytes(b'old output must be excluded')
    out = run(e, Path('src/release-output'))
    with tarfile.open(out / 'release.tar.gz') as archive:
        assert not any('release-output' in name for name in archive.getnames())


def test_input_mutation_during_copy_cannot_commit(environment, monkeypatch):
    e = environment
    original = m.shutil.copyfile
    def mutate(source, target):
        if source == e['snapshot'] / 'main.tex':
            source.write_text('Changed after bundle verification')
        return original(source, target)
    monkeypatch.setattr(m.shutil, 'copyfile', mutate)
    with pytest.raises(WorkflowError, match='changed while packaging'):
        run(e)
    assert e['calls'] and not e['calls'][0].exists()


def test_empty_required_comparisons_rejected(environment):
    e = environment
    path = e['snapshot'] / 'publication.json'
    config = json.loads(path.read_text())
    config['required_comparison_ids'] = []
    write(path, config)
    reseal(e)
    with pytest.raises(WorkflowError, match='No required main comparisons'):
        run(e)


def test_inventory_completeness_is_not_invented(environment):
    e = environment
    (e['snapshot'] / 'source_inventory.json').unlink()
    reseal(e)
    out = run(e)
    with tarfile.open(out / 'release.tar.gz') as archive:
        portability = json.load(archive.extractfile('portability.json'))
    assert any('data dependency completeness is unverified' in x for x in portability['limitations'])
    assert portability['offline_complete'] is False
