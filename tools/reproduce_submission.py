#!/usr/bin/env python3
"""Run one fresh bounded study, freeze finalists, score holdout, regenerate paper.

Run after make setup and make data, using the root environment. All expensive
stages are explicit. Run IDs/output directories are never silently reused.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_completion_study import prepare

def run(command,root,capture=False):
    print('+ '+' '.join(map(str,command)),flush=True)
    return subprocess.run(list(map(str,command)),cwd=root,check=True,text=True,
                          stdout=subprocess.PIPE if capture else None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run-id',required=True)
    p.add_argument('--data',type=Path);p.add_argument('--planning-hours',type=float,default=1.5)
    p.add_argument('--author',default='Vladislav Neklydov');p.add_argument('--no-pdf',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    raw=a.data or Path(json.loads((root/'data/source.json').read_text())['path'])
    if not 0<a.planning_hours<=8:p.error('planning-hours must be in (0,8]')
    destination=root/'build/reproductions'/a.run_id
    destination.mkdir(parents=True,exist_ok=False)
    config,command=prepare(root,raw,Path('reproducibility/baseline-readiness.json'),a.run_id,None)
    c=json.loads(config.read_text());c['scientific']['parameters']['batch_config']['wall_hours']=a.planning_hours
    c['runtime']['resources']['timeout_seconds']=math.ceil(a.planning_hours*3600)+120
    config.write_text(json.dumps(c,indent=2)+'\n')
    python=Path(sys.executable).absolute();engine=root/'engines/tlab_warmup/.venv/bin/python'
    native_run=root/'runs'/a.run_id
    run([python,'-m','researchflow.cli','experiment','--root',root,'--config',config,'--run-id',a.run_id],root)
    study=destination/'study.json'
    run([engine,root/'tools/collect_submission_results.py','--root',root,'--run',native_run,'--output',study],root)
    analysis_id='A-finish-'+a.run_id.removeprefix('R-finish-')
    run([python,root/'tools/analyze_completion_study.py','--root',root,'--run',native_run,'--analysis-id',analysis_id],root)
    ac=root/'configs/completion/science'/analysis_id/'analysis.json'
    response=run([python,'-m','researchflow.cli','analyze','--root',root,'--config',ac],root,capture=True)
    analysis=Path(response.stdout.strip())
    if not (analysis/'manifest.json').is_file():raise ValueError('Analyzer did not return a selected bundle')
    final=destination/'holdout'
    run([engine,root/'tools/evaluate_frozen_final.py','--root',root,'--run',native_run,'--output',final],root)
    run([engine,root/'tools/evaluate_frozen_final.py','--root',root,'--freeze',final/'freeze.json'],root)
    paper=destination/'paper'
    run([python,root/'tools/render_submission.py','--root',root,'--study',study,'--final',final/'result.json',
         '--analysis',analysis,'--output',paper,'--author',a.author,'--reading-output',destination/'reading'],root)
    if not a.no_pdf:
        run([python,root/'tools/build_scientific_paper.py','--source',paper,'--output',destination/'reading/main.pdf'],root)
    else:
        index=destination/'reading/README.md'
        index.write_text(index.read_text().replace('[the paper](main.pdf)','the authored paper source (PDF omitted by --no-pdf)'))
    summary={'run':str(native_run.relative_to(root)),'analysis':str(analysis.relative_to(root)),
             'study':str(study.relative_to(root)),'freeze':str((final/'freeze.json').relative_to(root)),
             'final':str((final/'result.json').relative_to(root)),
             'paper_source':str((paper/'main.tex').relative_to(root)),
             'visual_review':'Owner review remains separate; inspect the exact returned PDF'}
    (destination/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
