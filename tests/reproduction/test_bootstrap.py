from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import pytest

ROOT=Path(__file__).resolve().parents[2]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
cold=module('cold_start');restore=module('restore_evidence')

def test_setup_preserves_two_project_locks(tmp_path,monkeypatch):
 engine=tmp_path/'engines/tlab_warmup';engine.mkdir(parents=True)
 for p in [tmp_path,engine]:(p/'uv.lock').write_text('fixed')
 calls=[];monkeypatch.setattr(cold.shutil,'which',lambda p:'/bin/uv')
 monkeypatch.setattr(cold,'run',lambda argv,cwd:calls.append((argv,cwd)))
 a,b=cold.setup(tmp_path,'uv')
 assert len(calls)==2
 assert calls[0][1]==tmp_path and calls[1][1]==engine
 assert all('--frozen' in c[0] and '--python' in c[0] for c in calls)
 assert a==cold.python_at(tmp_path) and b==cold.python_at(engine)
 assert all((p/'uv.lock').read_text()=='fixed' for p in [tmp_path,engine])

def test_no_tex_baseline_has_explicit_existing_data_and_fresh_identity(tmp_path,monkeypatch):
 for project in [tmp_path,tmp_path/'engines/tlab_warmup']:
  p=cold.python_at(project);p.parent.mkdir(parents=True);p.write_text('fake executable')
 raw=tmp_path/'raw with spaces.parquet';raw.write_bytes(b'fixture')
 calls=[]
 def run(argv,cwd):
  argv=list(map(str,argv));calls.append(argv)
  if any(x.endswith('fetch_pinned_data.py') for x in argv):
   p=tmp_path/'data/source.json';p.parent.mkdir();p.write_text(json.dumps({'path':str(raw)}))
 monkeypatch.setattr(cold,'run',run)
 monkeypatch.setattr(sys,'argv',['cold_start.py','baseline','--root',str(tmp_path),'--skip-setup','--existing-data',str(raw),'--no-tex'])
 assert cold.main()==0
 assert len(calls)==5
 assert calls[0][-2:]==['--existing-data',str(raw)]
 assert '--no-tex' in calls[-1]
 experiment=next(x for x in calls if 'experiment' in x)
 assert experiment[experiment.index('--run-id')+1].startswith('R-baseline-')
 assert not any('doctor(' in part for call in calls for part in call)

def archive(root,entries):
 root.mkdir(exist_ok=True);p=root/'evidence';p.mkdir(exist_ok=True)
 output=root.parent/(root.name+'.tar.gz')
 with tarfile.open(output,'w:gz') as t:
  for name,data in entries:
   info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
 (p/'distribution.json').write_text(json.dumps({'sha256':restore.sha(output),'size_bytes':output.stat().st_size,'complete_bundles':[]}))
 return output

def test_restore_is_idempotent_and_refuses_overwrite(tmp_path):
 root=tmp_path/'project';a=archive(root,[('local-history/example.json',b'{}')])
 restore.restore(root,a);restore.restore(root,a)
 target=root/'local-history/example.json';target.write_text('changed')
 with pytest.raises(ValueError,match='overwrite'):restore.restore(root,a)
 assert target.read_text()=='changed'

@pytest.mark.parametrize('name',['../escape','/absolute','safe/../../escape','safe\\escape'])
def test_restore_rejects_traversal(tmp_path,name):
 root=tmp_path/'project';a=archive(root,[(name,b'{}')])
 with pytest.raises(ValueError,match='Unsafe'):restore.restore(root,a)

def test_restore_checks_archive_hash_before_writing(tmp_path):
 root=tmp_path/'project';a=archive(root,[('local-history/example.json',b'{}')]);a.write_bytes(a.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='differs'):restore.restore(root,a)
 assert not (root/'local-history').exists()
