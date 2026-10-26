#!/usr/bin/env python3
"""Rebuild the explicitly pinned figures/paper from saved evidence; never train."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--archive',type=Path);p.add_argument('--build',action='store_true');p.add_argument('--author')
    a=p.parse_args();root=a.root.resolve();selection=json.loads((root/'evidence/submission/selection.json').read_text())
    if a.archive is None and not (root/selection['analysis_path']/'manifest.json').is_file():
        a.archive=root/'runs/published/development/analysis.tar.xz'
    if a.archive:
        from restore_submission_analysis import restore
        print(restore(root,a.archive.resolve()))
    for item in selection['selected_files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Selected evidence escapes repository')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=item['sha256']:raise ValueError('Selected evidence changed: '+item['path'])
    analysis=root/selection['analysis_path']
    if not (analysis/'manifest.json').exists():
        raise SystemExit('Restore the pinned analysis artifact first: --archive /path/to/tlab-submission-analysis.tar.xz')
    from render_submission import render
    result=render(root,root/selection['study_path'],root/selection['final_path'],analysis,
                  root/'paper/submission',a.author or selection['author'])
    print(json.dumps(result,indent=2))
    if a.build:
        from build_scientific_paper import build
        print(build(root/'paper/submission',root/'reports/submission/main.pdf'))

if __name__=='__main__':main()
