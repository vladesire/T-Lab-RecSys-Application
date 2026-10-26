#!/usr/bin/env python3
"""Restore the separately distributed evidence archive without rewriting history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(root: Path, archive: Path) -> dict:
    from researchflow.bundles import verify_bundle
    root, archive = root.resolve(), archive.resolve()
    declaration = json.loads((root / "evidence/distribution.json").read_text())
    if sha(archive) != declaration["sha256"] or archive.stat().st_size != declaration["size_bytes"]:
        raise ValueError("Evidence archive differs from the pinned distribution")
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=root) as temporary:
        stage = Path(temporary)
        with tarfile.open(archive) as tar:
            seen = set()
            for member in tar:
                path = PurePosixPath(member.name)
                if (path.is_absolute() or any(p in {"", ".", ".."} for p in member.name.split("/"))
                        or "\\" in member.name or member.name in seen or not member.isfile()):
                    raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
                seen.add(member.name)
                target = stage / Path(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        for bundle in declaration["complete_bundles"]:
            verify_bundle(stage / bundle["path"], expected_kind=bundle["kind"])
        paths = sorted(p for p in stage.rglob("*") if p.is_file())
        # Preflight the entire restore before moving any bytes into the project.
        for source in paths:
            target = root / source.relative_to(stage)
            for parent in [target, *target.parents]:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError(f"Refusing a symlink destination: {parent}")
            if target.exists() and (not target.is_file() or sha(target) != sha(source)):
                raise ValueError(f"Refusing to overwrite different existing evidence: {target}")
        for source in paths:
            target = root / source.relative_to(stage)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
    for bundle in declaration["complete_bundles"]:
        verify_bundle(root / bundle["path"], expected_kind=bundle["kind"])
    return {"restored_complete_bundles": declaration["complete_bundles"],
            "historical_run": "Partial transfer only; local-history is not a verified run store."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.root, args.archive), indent=2))


if __name__ == "__main__":
    main()
