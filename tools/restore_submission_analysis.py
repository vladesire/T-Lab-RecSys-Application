#!/usr/bin/env python3
"""Restore the exact complete selected analysis artifact; no raw-data dependency."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def restore(root,archive):
    from researchflow.bundles import verify_bundle
    declaration=json.loads((root/'evidence/submission/distribution.json').read_text())
    if archive.stat().st_size!=declaration['size_bytes'] or sha(archive)!=declaration['sha256']:
        raise ValueError('Archive differs from the explicitly pinned submission artifact')
    destination=root/declaration['analysis_path']
    if any(p.is_symlink() for p in [destination,*destination.parents] if p!=root.parent):
        raise ValueError('Refusing symlink destination')
    if destination.exists():
        verify_bundle(destination,'analysis')
        if sha(destination/'manifest.json')!=declaration['manifest_sha256']:
            raise ValueError('Existing analysis differs; never overwrite immutable evidence')
        return destination
    with tempfile.TemporaryDirectory(dir=root,prefix='.restore-') as tmp:
        stage=Path(tmp);seen=set();total=0
        with tarfile.open(archive) as tar:
            for member in tar:
                p=PurePosixPath(member.name)
                if (p.is_absolute() or '..' in p.parts or '\\' in member.name or ':' in member.name
                        or member.name in seen or not (member.isfile() or member.isdir())):
                    raise ValueError('Unsafe/duplicate artifact member')
                if not (member.name==declaration['analysis_path'] or member.name.startswith(declaration['analysis_path']+'/')):
                    raise ValueError('Archive member outside the declared analysis')
                seen.add(member.name)
                if member.isdir():continue
                total+=member.size
                if total>declaration['uncompressed_bytes']:raise ValueError('Unexpected expanded artifact size')
                target=stage/Path(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('xb') as out:shutil.copyfileobj(src,out)
        if total!=declaration['uncompressed_bytes']:raise ValueError('Expanded artifact is incomplete')
        selected=stage/declaration['analysis_path'];verify_bundle(selected,'analysis')
        if sha(selected/'manifest.json')!=declaration['manifest_sha256']:raise ValueError('Analysis identity mismatch')
        destination.parent.mkdir(parents=True,exist_ok=True);selected.replace(destination)
    verify_bundle(destination,'analysis');return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();print(restore(a.root.resolve(),a.archive.resolve()))
