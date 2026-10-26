#!/usr/bin/env python3
"""Run with the vendored engine environment; acquire only the pinned raw file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--existing-data", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    engine = root / "engines/tlab_warmup"
    sys.path.insert(0, str(engine / "src"))
    from tlab_warmup.data import acquire
    with tarfile.open(engine / "provenance/iteration0/results.tar.gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == "resolved_config.json"]
        if len(members) != 1:
            raise ValueError("Expected exactly one archived anchor configuration")
        config = json.load(archive.extractfile(members[0]))
    declared = json.loads((root / "reproducibility/dataset.json").read_text())
    if config["source"] != declared:
        raise ValueError("Dataset declaration differs from the archived anchor")
    cache = root / "data" / declared["revision"]
    cache.mkdir(parents=True, exist_ok=True)
    raw = acquire(config, cache, str(args.existing_data) if args.existing_data else None, print)
    record = root / "data/source.json"
    temporary = record.with_suffix(".json.partial")
    temporary.write_text(json.dumps({"path": str(raw), "source": declared}, indent=2) + "\n")
    temporary.replace(record)
    print("Verified data location recorded: " + str(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
