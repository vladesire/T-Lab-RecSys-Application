"""Synthetic fake-CLI tests; never evidence of actual-data anchor reproduction."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("m19_bridge", ROOT / "runners/warmup/bridge.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)

BASE_CONFIG = {
    "schema": "tlab-experiments-v1", "threads": 4, "data": None,
    "wall_hours": 6, "per_job_seconds": 2700, "max_retries": 1,
    "target_rss_gib": 8, "stop_rss_gib": 10, "minimum_available_gib": 4,
    "low_memory_seconds": 10, "maximum_selected_rows": 8000000,
    "bootstrap_replicates": 2000, "search_rounds": 3, "search_fit_cap": 24,
    "search_min_gain": .0001, "search_min_queries": 200, "search_min_users": 20,
    "search_min_novel": 50, "additional_model_seed_offsets": [11, 29],
    "iteration1_output": "runs/iteration1",
    "panels": {"iteration1": {"users": 3, "train": 1, "development": 2},
               "calibration": {"users": 3, "train": 1, "development": 2},
               "main": {"users": 4, "train": 2, "development": 3}},
    "optional_artist": "blocked_pending_mapping_semantics_validation",
    "optional_3000_user_panel": "blocked_pending_secondary_scale_calibration",
    "optional_100_alternatives": "blocked_pending_variable_cardinality_validation",
}
FEATURES = {"names": ["user_count", "user_item_count", "user_item_recency_seconds"],
            "original": ["user_count", "user_item_count", "user_item_recency_seconds"]}
ENV = {"python": "FAKE interpreter, synthetic CLI only", "platform": "synthetic", "features": FEATURES}

# This is an independent executable implementing the INSPECTED CLI/file boundary,
# not a replacement for any production module. -c probes and -m launches are real
# subprocesses. Its emitted predictions have known scores, histories and IDs.
FAKE = r'''import gzip,hashlib,json,os,pathlib,sys,tarfile
import numpy as np
P=pathlib.Path
if '-c' in sys.argv:
 print((P(__file__).parent/'fake-env.json').read_text());sys.exit(0)
assert sys.argv[1:4]==['-B','-m','tlab_warmup.batch']
a=sys.argv
phase=a[4]
cfg=json.loads(P(a[a.index('--config')+1]).read_text())
anchor=P(a[a.index('--anchor')+1])
assert P(a[a.index('--data')+1]).is_file()
run=P(a[a.index('--output')+1]);run.mkdir(parents=True,exist_ok=True)
P('argv.json').write_text(json.dumps(a))
(run/'effective.json').write_text(json.dumps(cfg))
if os.environ.get('M19_FAKE_STOP') and not (run/'stopped').exists():
 (run/'stopped').write_text('native interrupted work');sys.exit(4)
prep=P.cwd()/'data/batches/example/prepare_U20';prep.mkdir(parents=True,exist_ok=True)
features=json.loads((P(__file__).parent/'fake-env.json').read_text())['features']
queries=[{'query_id':1,'uid':7,'timestamp':50,'positive':2,'candidates':list(range(21)),'split':'train'},
 {'query_id':9007199254740993,'uid':7,'timestamp':101,'positive':2,'candidates':list(range(21)),'split':'development'},
 {'query_id':9007199254740995,'uid':9,'timestamp':150,'positive':20,'candidates':list(reversed(range(21))),'split':'development'}]
with gzip.open(prep/'queries.json.gz','wt') as f:json.dump(queries,f)
mat=np.zeros((63,3));mat[21+2,1]=4;mat[21+5,1]=1;mat[42+4,1]=3
np.save(prep/'features.npy',mat)
query_hash=hashlib.sha256(json.dumps(queries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
meta={'config':{'seed':17,'train_end_timestamp':100,'development_end_timestamp':200},'test_scored':False,'protocol':'U20','query_candidate_hash':query_hash}
(prep/'result.json').write_text(json.dumps(meta));(prep/'feature_manifest.json').write_text(json.dumps(features))
def commit(d):
 files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
 (d/'COMPLETE.json').write_text(json.dumps({'files':files}))
commit(prep)
ledger={}
for name in ['original','popularity']:
 job=run/'jobs'/('U20_'+name);job.mkdir(parents=True,exist_ok=True)
 scores=np.array([[.1]*21,list(np.arange(21)/21)])
 np.savez_compressed(job/'predictions.npz',scores=scores,query_ids=np.array([q['query_id'] for q in queries[1:]]),repeat=np.array([True,False]))
 result={'method':name,'seed':17,'protocol':'U20','query_candidate_hash':query_hash,'anchor_prediction_max_absolute_difference':0.0}
 if name=='original':result['feature_names']=features['original']
 (job/'result.json').write_text(json.dumps(result));commit(job)
 ledger['U20_'+name]={'status':'completed','output':str(job),'request':{'kind':'fit','protocol_result':str(prep),'seed':17,'feature_set':name,'config':cfg},'attempts':[]}
(run/'ledger.json').write_text(json.dumps(ledger))
(run/'readiness.json').write_text(json.dumps({'ready':True,'test_scored':False}))
(run/'plan_manifest.json').write_text(json.dumps({'config':cfg,'environment':'synthetic fake only','phase':phase}))
with tarfile.open(run.with_name(run.name+'-results.tar.gz'),'w:gz') as t:t.add(run/'plan_manifest.json',arcname='plan_manifest.json')
'''


def make_fake_python(directory, interpreter=sys.executable):
    """Use pytest's Python, including its venv; never resolve via shell PATH.

    A shell launcher also supports spaces/quotes in interpreter and script paths,
    which a direct Python shebang does not support portably on macOS/Linux.
    """
    script = directory / "fake-cli.py"
    script.write_text(FAKE)
    launcher = directory / "fake-python"
    launcher.write_text(
        "#!/bin/sh\nexec " + shlex.quote(str(interpreter)) + " "
        + shlex.quote(str(script)) + ' "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


def tar_json(path, name, value):
    with tarfile.open(path, "w:gz") as tar:
        data = json.dumps(value).encode()
        info = tarfile.TarInfo("anchor/" + name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "warmup project"
    for folder in ("src/tlab_warmup", "tests", "configs"):
        (project / folder).mkdir(parents=True)
    (project / "src/tlab_warmup/batch.py").write_text("# synthetic placeholder; fake executable owns CLI\n")
    (project / "src/tlab_warmup/core.py").write_text("import hashlib\ndef tie_key(seed,uid,timestamp,item):\n return int.from_bytes(hashlib.blake2b(f'{seed}:{uid}:{timestamp}:{item}'.encode(),digest_size=8).digest(),'big')\n")
    (project / "pyproject.toml").write_text("# synthetic environment\n")
    (project / "uv.lock").write_text("# synthetic lock\n")
    (project / "configs/experiments.json").write_text(json.dumps(BASE_CONFIG))
    python = make_fake_python(tmp_path)
    (tmp_path / "fake-env.json").write_text(json.dumps(ENV))
    raw = tmp_path / "fixture.parquet"
    raw.write_bytes(b"synthetic fake raw; no actual Parquet computation")
    anchor = {"seed": 17, "users": 3, "training_queries": 1, "development_queries": 2,
              "alternatives": 20, "ratio_smoothing": 1,
              "source": {"revision": "synthetic-fixture-v1", "synthetic": True, "bytes": raw.stat().st_size, "sha256": b.sha(raw)},
              "resolved_boundaries": {"train_end_exclusive": 100, "development_end_exclusive": 200}}
    archive = tmp_path / "anchor.tar.gz"
    tar_json(archive, "resolved_config.json", anchor)
    source = b.capture_source(project, python)
    protocol = {"id": "warmup-development-v1", "candidates": ["U20", "P20"], "history": "strict-past-complete", "train_end_timestamp": 100, "development_end_timestamp": 200, "timestamp_unit": "seconds", "final_test": "not-scored"}
    request = {"schema_version": 1, "run_id": "R-tiny", "scientific": {"experiment_id": "E-tiny", "branch_id": "B-tiny", "hypothesis_ids": [], "phase": "fixture", "runner_id": "warmup", "protocol": protocol,
               "parameters": {"batch_phase": "iteration1", "controls_only": False, "seed": 17, "anchor_config_sha256": b.digest(anchor), "feature_manifest_sha256": b.digest(FEATURES), "batch_config": copy.deepcopy(BASE_CONFIG)}},
               "data": [{"id": "source", "local_path": str(project), "manifest": source},
                        {"id": "anchor", "local_path": str(archive), "manifest": {"kind": "warmup_anchor_v1", "sha256": b.sha(archive)}},
                        {"id": "raw", "local_path": str(raw), "manifest": {"kind": "warmup_raw_v1", "sha256": b.sha(raw)}}],
               "identity": {k: hashlib.sha256(k.encode()).hexdigest() for k in ("scientific_sha256", "code_sha256", "data_sha256")}}
    return request, tmp_path / "payload", project


@pytest.mark.parametrize("target,key,value", [
    ("parameters", "typo_seed", 3), ("parameters", "selected_features", ["arbitrary"]),
    ("parameters", "seed", 999), ("parameters", "controls_only", True),
    ("config", "unknown", 1), ("config", "search_fit_cap", 15),
    ("config", "additional_model_seed_offsets", [5]), ("config", "threads", True),
    ("protocol", "history", "truncate"), ("protocol", "development_end_timestamp", 300),
])
def test_unsupported_settings_fail(setup, target, key, value):
    request, output, _ = setup
    p = request["scientific"]["parameters"]
    {"parameters": p, "config": p["batch_config"], "protocol": request["scientific"]["protocol"]}[target][key] = value
    with pytest.raises(b.BridgeError):
        b.execute(request, output)
    assert not output.exists()


def test_fake_cli_exact_mapping_and_immutable_inputs(setup):
    request, output, project = setup
    before = b.files_for(project)
    p = request["scientific"]["parameters"]
    p["batch_config"]["threads"] = 2
    p["batch_config"]["panels"]["iteration1"]["development"] = 5
    result = b.run_warmup_bridge(request, output)
    schema = json.loads((ROOT / "interfaces/runner/runner_result.schema.json").read_text())
    jsonschema.validate(result, schema)
    workspace = output / "workspace"
    cfg = json.loads((output / "resolved-batch-config.json").read_text())
    expected = copy.deepcopy(p["batch_config"])
    expected["data"] = request["data"][2]["local_path"]
    assert cfg == expected
    argv = json.loads((workspace / "argv.json").read_text())
    assert argv[1:] == ["-B", "-m", "tlab_warmup.batch", "iteration1", "--config", str(output/"resolved-batch-config.json"), "--anchor", request["data"][1]["local_path"], "--data", request["data"][2]["local_path"], "--output", str(workspace/"runs/R-tiny")]
    assert b.files_for(project) == before
    assert not (project / "data").exists() and not (project / "runs").exists()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["source_manifest"]["tree_sha256"] == b.digest(before)
    assert provenance["source_manifest"]["files"]["uv.lock"] == b.sha(project/"uv.lock")
    assert provenance["environment"] == ENV
    assert provenance["anchor"]["sha256"] == request["data"][1]["manifest"]["sha256"]
    assert provenance["result_archive"]["sha256"] == b.sha(output/provenance["result_archive"]["path"])
    assert result["predictions"][0]["tags"]["feature_names"] == FEATURES["original"]
    with pytest.raises(b.BridgeError, match="Existing output"):
        b.execute(request, output)


def test_saved_export_preserves_scores_ids_familiarity_ties_and_hashes(setup):
    request, output, _ = setup
    result = b.execute(request, output)
    desc = result["predictions"][0]
    with np.load(output/desc["path"], allow_pickle=False) as a:
        assert all(a[k].ndim == 1 and a[k].dtype.kind != 'O' for k in a.files)
        assert a["query_id"][0] == "9007199254740993"
        assert a["item_id"][21:].tolist() == list(map(str, reversed(range(21))))
        np.testing.assert_array_equal(a["score"], [.1]*21 + list(np.arange(21)/21))
        assert np.flatnonzero(a["label"]).tolist() == [2, 21]
        assert np.flatnonzero(a["is_familiar"]).tolist() == [2, 5, 25]
        assert a["is_repeat_target"].tolist() == [True]*21 + [False]*21
        # Independent reference via hashlib.hexdigest rather than adapter helpers.
        ties = [int(hashlib.blake2b(f"17:7:101:{it}".encode(), digest_size=8).hexdigest(),16) for it in range(21)]
        assert a["tie_key"][:21].tolist() == ties
        q = [{"query_id":"9007199254740993","user_id":"7","target_item_id":"2","timestamp":101,"is_repeat_target":True},
             {"query_id":"9007199254740995","user_id":"9","target_item_id":"20","timestamp":150,"is_repeat_target":False}]
        canon = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False).encode()).hexdigest()
        assert desc["identity"]["query_set_sha256"] == canon(q)
        candidate_records=[]
        for i, qr in enumerate(q):
            rows = [{"item_id":str(a['item_id'][j]),"is_familiar":bool(a['is_familiar'][j]),"tie_key":int(a['tie_key'][j])} for j in range(i*21,(i+1)*21)]
            candidate_records.append({"query_id":qr['query_id'],"candidate_set_sha256":canon(sorted(rows,key=lambda r:r['item_id']))})
        assert desc['identity']['candidate_set_sha256']==canon(candidate_records)
    assert result['predictions'][0]['identity']==result['predictions'][1]['identity']


@pytest.mark.parametrize("change", ["code", "lock", "new_code", "anchor", "raw", "environment"])
def test_provenance_changes_fail_before_launch(setup, change):
    request, output, project = setup
    if change in ("code", "lock", "new_code"):
        path = project / {"code":"src/tlab_warmup/batch.py","lock":"uv.lock","new_code":"src/tlab_warmup/new.py"}[change]
        path.write_text("modified")
    elif change in ("anchor", "raw"):
        Path(request['data'][1 if change=='anchor' else 2]['local_path']).write_bytes(b'changed')
    else:
        path = project.parent / "fake-env.json"
        env=copy.deepcopy(ENV);env['python']='changed';path.write_text(json.dumps(env))
    with pytest.raises(b.BridgeError):
        b.execute(request, output)
    assert not (output/"workspace/argv.json").exists()


def test_output_protection_and_aliases(setup, tmp_path):
    request, output, project = setup
    old=tmp_path/'old-results';old.mkdir();(old/'keep').write_bytes(b'original')
    for target in [project/'data/new', project/'runs/new', tmp_path, old, Path(request['data'][2]['local_path'])]:
        with pytest.raises(b.BridgeError):b.execute(request,target)
    assert list(old.iterdir())==[old/'keep']
    alias=tmp_path/'alias';alias.symlink_to(project,target_is_directory=True)
    with pytest.raises(b.BridgeError):b.execute(request,alias/'runs/new')
    assert (old/'keep').read_bytes()==b'original'


def test_identical_interrupted_attempt_resumes_native_directory(setup, monkeypatch):
    request, output, _ = setup
    monkeypatch.setenv('M19_FAKE_STOP','1')
    with pytest.raises(b.BridgeError,match='failed/stopped'):b.execute(request,output)
    stopped=output/'workspace/runs/R-tiny/stopped'
    assert stopped.exists()
    changed=copy.deepcopy(request);changed['scientific']['parameters']['batch_config']['threads']=1
    with pytest.raises(b.BridgeError,match='Existing output'):b.execute(changed,output)
    assert b.execute(request,output)['kind']=='ranking'
    assert stopped.read_text()=='native interrupted work'


def test_standalone_cli_without_researchflow(setup, tmp_path):
    request, output, _ = setup
    runner=tmp_path/'standalone';runner.mkdir()
    for name in ('runner.py','bridge.py'):
        (runner/name).write_bytes((ROOT/'runners/warmup'/name).read_bytes())
    req=tmp_path/'request.json';req.write_text(json.dumps(request))
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    p=subprocess.run([sys.executable,str(runner/'runner.py'),'--request',str(req),'--output',str(output)],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    assert (output/'result.json').exists()


def test_boundary_errors_are_shared_workflow_errors(setup):
    from researchflow.contracts import WorkflowError
    request,output,_=setup
    request['scientific']['parameters']['unsupported']=1
    with pytest.raises(WorkflowError) as error:b.run_warmup_bridge(request,output)
    assert error.value.code=='CONTRACT'


def test_historical_ties_preserved_and_final_and_corruption_fail():
    q={'query_id':1,'uid':7,'timestamp':101,'positive':0,'candidates':list(range(21)),'tie_keys':list(reversed(range(21)))}
    protocol={'train_end_timestamp':100,'development_end_timestamp':200}
    scores=np.zeros((1,21));familiar=np.zeros((1,21),dtype=bool)
    arrays,_=b.ranking_arrays([q],scores,familiar,17,protocol)
    assert arrays['tie_key'].tolist()==list(reversed(range(21)))
    for key,value in [('timestamp',200),('candidates',[0]*21),('tie_keys',[-1]*21)]:
        bad={**q,key:value}
        with pytest.raises(b.BridgeError):b.ranking_arrays([bad],scores,familiar,17,protocol)
    with pytest.raises(b.BridgeError):b.ranking_arrays([q],scores+np.nan,familiar,17,protocol)


def test_normalize_rejects_query_order_or_repeat_corruption(setup):
    request,output,_=setup
    b.execute(request,output)
    run=output/'workspace/runs/R-tiny';job=run/'jobs/U20_original'
    with np.load(job/'predictions.npz') as z:arrays={k:z[k].copy() for k in z.files}
    arrays['query_ids']=arrays['query_ids'][::-1]
    np.savez_compressed(job/'predictions.npz',**arrays)
    with pytest.raises(b.BridgeError,match='checksum'):b.normalize(run,output/'workspace',output,request['scientific']['protocol'],17,FEATURES)
    # Update synthetic completion record so alignment is tested independently.
    files={str(p.relative_to(job)):b.sha(p) for p in job.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    (job/'COMPLETE.json').write_text(json.dumps({'files':files}))
    with pytest.raises(b.BridgeError,match='alignment'):b.normalize(run,output/'workspace',output,request['scientific']['protocol'],17,FEATURES)


def test_overnight_readiness_and_controls_mapping(setup, tmp_path):
    request,output,_=setup
    request['scientific']['parameters'].update(batch_phase='overnight', controls_only=True)
    readiness=tmp_path/'readiness.json'
    readiness.write_text(json.dumps({'ready':True,'test_scored':False,'code_hash':'synthetic'}))
    request['data'].append({'id':'prior-readiness','local_path':str(readiness),'manifest':{'kind':'warmup_readiness_v1','sha256':b.sha(readiness)}})
    result=b.execute(request,output)
    args=json.loads((output/'workspace/argv.json').read_text())
    assert args[4]=='overnight' and args[-1]=='--controls-only'
    assert (output/'workspace/runs/iteration1/readiness.json').read_bytes()==readiness.read_bytes()
    assert result['checks']['items'][0]['status']=='unverified'


def test_declared_venv_python_path_is_preserved(setup, tmp_path):
    request,_,project=setup
    alias=tmp_path/'venv/bin/python';alias.parent.mkdir(parents=True)
    alias.symlink_to(request['data'][0]['manifest']['python'])
    # capture uses an absolute executable spelling, never resolves a venv symlink
    # into the base interpreter (which would select a different environment).
    original=b.probe
    seen=[]
    try:
        b.probe=lambda python,project:seen.append(str(python)) or ENV
        source=b.capture_source(project,alias)
    finally:b.probe=original
    assert source['python']==str(alias) and seen==[str(alias)]


def test_supplied_saved_anchor_export_when_available():
    """Actual archived DEVELOPMENT predictions only; no raw-data execution.

    Opt in on any host with M19_WARMUP_PROJECT. This is not the local anchor
    reproduction acceptance: the model is not fitted or scored here.
    """
    root=os.environ.get('M19_WARMUP_PROJECT')
    if not root:pytest.skip('Set M19_WARMUP_PROJECT to inspect supplied archived development export')
    import csv
    project=Path(root)
    archive=project/'provenance/iteration0/results.tar.gz'
    with tarfile.open(archive) as tar:
        members={Path(m.name).name:m for m in tar.getmembers() if m.isfile()}
        def raw(name):return tar.extractfile(members[name]).read()
        for name,expected in json.loads(raw('artifact_checksums.json')).items():
            assert hashlib.sha256(raw(name)).hexdigest()==expected
        config=json.loads(raw('resolved_config.json'))
        rows=list(csv.DictReader(io.StringIO(gzip.decompress(raw('development_predictions.csv.gz')).decode())))
    groups={}
    for row in rows:groups.setdefault(row['query_id'],[]).append(row)
    groups=list(groups.values())[:5]
    queries=[];scores=[];familiar=[]
    for group in groups:
        positives=[r for r in group if int(r['label'])==1]
        assert len(positives)==1
        r=positives[0]
        queries.append({'query_id':int(r['query_id']),'uid':int(r['uid']),'timestamp':int(r['timestamp']), 'positive':int(r['item_id']), 'candidates':[int(r['item_id']) for r in group]})
        scores.append([float(r['boosting']) for r in group])
        familiar.append([float(r['user_item_count'])>0 for r in group])
    bounds=config['resolved_boundaries']
    protocol={'train_end_timestamp':bounds['train_end_exclusive'],'development_end_timestamp':bounds['development_end_exclusive']}
    arrays,identity=b.ranking_arrays(queries,np.asarray(scores),np.asarray(familiar),config['seed'],protocol,lambda seed,u,t,it:int.from_bytes(hashlib.blake2b(f"{seed}:{u}:{t}:{it}".encode(),digest_size=8).digest(),"big"))
    assert len(queries)==5
    assert arrays['item_id'].tolist()==[r['item_id'] for group in groups for r in group]
    assert arrays['query_id'].tolist()==[r['query_id'] for group in groups for r in group]
    assert arrays['label'].tolist()==[bool(int(r['label'])) for group in groups for r in group]
    np.testing.assert_array_equal(arrays['score'],np.asarray(scores).ravel())
    # Load only the actual upstream pure numerical core, with no pyarrow/ML imports.
    import types
    core=types.ModuleType('actual_warmup_core')
    core_path=project/'src/tlab_warmup/core.py'
    exec(compile(core_path.read_bytes(),str(core_path),'exec'),core.__dict__)
    assert arrays['tie_key'].tolist()==[core.tie_key(config['seed'],int(r['uid']),int(r['timestamp']),int(r['item_id'])) for group in groups for r in group]
    _,second=b.ranking_arrays(queries,np.zeros_like(scores),np.asarray(familiar),config['seed'],protocol,core.tie_key)
    assert identity==second


def test_final_phase_and_feature_manifest_override_fail(setup):
    request,output,_=setup
    request['scientific']['phase']='final'
    with pytest.raises(b.BridgeError,match='never scores final'):b.execute(request,output)
    request['scientific']['phase']='fixture'
    request['scientific']['parameters']['feature_manifest_sha256']='a'*64
    with pytest.raises(b.BridgeError,match='feature selection'):b.execute(request,output)


def test_configuration_example_conforms_to_envelope():
    import yaml
    example=yaml.safe_load((ROOT/'configs/examples/warmup_experiment.yaml').read_text())
    jsonschema.validate(example,json.loads((ROOT/'src/researchflow/schemas/experiment_config.json').read_text()))
    b.validate_config(example['scientific']['parameters']['batch_config'])


def test_fake_cli_uses_pytest_interpreter_when_path_python_has_no_numpy(tmp_path, monkeypatch, request):
    # Reproduce `uv run pytest` while an unrelated Python is first on shell PATH.
    # Create it without pip or system packages; no dependency install is needed.
    foreign = tmp_path / "unrelated python"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(foreign)], check=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    missing = subprocess.run(
        [str(foreign / "bin/python3"), "-c", "import numpy"],
        env=env, text=True, capture_output=True,
    )
    assert missing.returncode != 0 and "No module named 'numpy'" in missing.stderr
    monkeypatch.setenv("PATH", str(foreign / "bin") + os.pathsep + os.defpath)
    legacy = tmp_path / "legacy-fake-python"
    legacy.write_text("#!/usr/bin/env python3\n" + FAKE)
    legacy.chmod(0o755)
    old = subprocess.run([str(legacy), "-c", "probe"], text=True, capture_output=True)
    assert old.returncode != 0 and "No module named 'numpy'" in old.stderr
    # Request setup AFTER the PATH change, so its initial environment probe is
    # exercised too (the old fixture fails here, before any test body runs).
    data, output, _ = request.getfixturevalue("setup")
    assert b.execute(data, output)["kind"] == "ranking"


def test_fake_launcher_quotes_interpreter_and_script_paths(tmp_path):
    directory = tmp_path / "helper's files with spaces"
    directory.mkdir()
    interpreter = directory / "chosen python"
    interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    interpreter.chmod(0o755)
    launcher = make_fake_python(directory, interpreter)
    (directory / "fake-env.json").write_text(json.dumps(ENV))
    process = subprocess.run([str(launcher), "-B", "-c", "probe"], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == ENV
