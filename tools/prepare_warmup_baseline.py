#!/usr/bin/env python3
"""Prepare M19's documented iteration1 reproduction config; never run experiments.

Put this file in project/tools/ and run from project/:
  uv run python tools/prepare_warmup_baseline.py \
      --warmup /path/to/warmup --data /path/to/multi_event.parquet
The default output is inputs/warmup-anchor-001/ and must not already exist.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys


def prepare(root: Path, warmup: Path, python: Path, raw: Path, output: Path) -> Path:
    # Preserve the spelling of a virtualenv interpreter: resolving its symlink
    # could select the base environment instead.
    root, warmup, raw, output = (p.expanduser().resolve() for p in (root, warmup, raw, output))
    python = python.expanduser().absolute()
    bridge_path = root / "runners/warmup/bridge.py"
    runner_path = root / "runners/warmup/runner.py"
    archive = warmup / "provenance/iteration0/results.tar.gz"
    for path in (bridge_path, runner_path, archive, python, raw):
        if not path.is_file():
            raise ValueError(f"Required file missing: {path}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new --output: {output}")
    if output.is_relative_to(warmup):
        raise ValueError("Output must be outside the original warmup project")
    spec = importlib.util.spec_from_file_location("warmup_baseline_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    anchor = bridge.anchor_config(archive)
    cfg = json.loads((warmup / "configs/experiments.json").read_text())
    bridge.validate_config(cfg)
    panel = cfg["panels"]["iteration1"]
    if (panel["users"] != anchor["users"] or panel["train"] != anchor["training_queries"]
            or panel["development"] < anchor["development_queries"]):
        raise ValueError("Upstream iteration1 configuration does not preserve the anchor cohort/query caps")
    print("Checking the declared upstream environment and hashing the raw file...", file=sys.stderr)
    source = bridge.capture_source(warmup, python)
    raw_hash = bridge.sha(raw)
    if raw.stat().st_size != anchor["source"]["bytes"] or raw_hash != anchor["source"]["sha256"]:
        raise ValueError("Raw file does not match the pinned iteration0 anchor (bytes/SHA256)")
    bounds = anchor["resolved_boundaries"]
    scientific = {
        "experiment_id": "E-warmup-anchor", "branch_id": "B-warmup-anchor",
        "hypothesis_ids": [],
        "phase": "fixture" if anchor["source"]["synthetic"] else "development",
        "runner_id": "warmup",
        "protocol": {
            "id": "warmup-development-v1", "candidates": ["U20", "P20"],
            "history": "strict-past-complete",
            "train_end_timestamp": bounds["train_end_exclusive"],
            "development_end_timestamp": bounds["development_end_exclusive"],
            "timestamp_unit": "seconds", "final_test": "not-scored",
        },
        "parameters": {
            "batch_phase": "iteration1", "controls_only": False, "seed": anchor["seed"],
            "anchor_config_sha256": bridge.digest(anchor),
            "feature_manifest_sha256": bridge.digest(source["environment"]["features"]),
            "batch_config": cfg,
        },
    }
    manifests = [
        ("source", warmup, source),
        ("anchor", archive, {"kind": "warmup_anchor_v1", "sha256": bridge.sha(archive)}),
        ("raw", raw, {"kind": "warmup_raw_v1", "sha256": raw_hash}),
    ]
    refs = [{"id": name, "local_path": str(path),
             "manifest": str(output / (name + "-manifest.json"))}
            for name, path, _ in manifests]
    experiment = {"schema_version": 1, "scientific": scientific, "runtime": {
        "command": [str(python), str(runner_path)],
        "code_paths": [str(runner_path), str(bridge_path)],
        "data_refs": refs, "research_paths": [],
        "resources": {"timeout_seconds": 23000, "memory_limit_bytes": None},
    }}
    # Everything above is read-only. Write to a fresh directory only after all
    # input checks succeed; if a write fails, leave its partial evidence in place.
    output.mkdir(parents=True, exist_ok=False)
    for name, _, manifest in manifests:
        bridge.write(output / (name + "-manifest.json"), manifest)
    config_path = output / "warmup-experiment.json"
    bridge.write(config_path, experiment)
    print("Prepared only; no experiment has run.", file=sys.stderr)
    print("Run: " + shlex.join([
        "uv", "run", "researchflow", "experiment", "--root", str(root),
        "--config", str(config_path), "--run-id", "R-warmup-anchor-001",
    ]), file=sys.stderr)
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Infrastructure project root")
    parser.add_argument("--warmup", type=Path, required=True, help="Original warmup source project")
    parser.add_argument("--python", type=Path, dest="warmup_python",
                        help="Upstream interpreter (default: WARMUP/.venv/bin/python)")
    parser.add_argument("--data", type=Path, required=True, help="Pinned full-release raw Parquet")
    parser.add_argument("--output", type=Path, default=Path("inputs/warmup-anchor-001"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    python = args.warmup_python or args.warmup / ".venv/bin/python"
    try:
        print(prepare(args.root, args.warmup, python, args.data, output))
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
