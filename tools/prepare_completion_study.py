#!/usr/bin/env python3
"""Prepare one pilot-informed development batch. Never launch computation."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from jsonschema import Draft202012Validator, FormatChecker


HYPOTHESES = ["H-finish-compact", "H-finish-decay", "H-finish-repeat"]
BATCH_CHANGES = {"wall_hours": 0.4, "bootstrap_replicates": 1000,
                 "search_rounds": 1, "search_fit_cap": 8,
                 "additional_model_seed_offsets": []}
SCHEMA_DIR = "src/researchflow/schemas"
FORMATS = FormatChecker()


@FORMATS.checks("date-time", raises=ValueError)
def date_time(value):
    # jsonschema's optional date-time dependency need not be installed. Canonical
    # research records require a timestamp with explicit timezone, never a date.
    if not isinstance(value, str):
        return True
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value):
        return False
    datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
    return True


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(root, schema_name, value):
    schema = load(root / SCHEMA_DIR / (schema_name + ".json"))
    Draft202012Validator(schema, format_checker=FORMATS).validate(value)


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Unsafe or overlong identity: " + value)
    return value


def bridge_for(root):
    path = root / "runners/warmup/bridge.py"
    spec = importlib.util.spec_from_file_location("completion_warmup_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_paths(root):
    groups = [("hypotheses", "hypothesis", "H"), ("branches", "branch", "B"),
              ("decisions", "decision", "D")]
    paths = []
    for folder, schema, prefix in groups:
        for path in sorted((root / "research" / folder).glob(prefix + "-finish-*.json")):
            validate(root, schema, load(path))
            paths.append(path)
    for hid in HYPOTHESES:
        if root / "research/hypotheses" / (hid + ".json") not in paths:
            raise ValueError("Missing canonical hypothesis: " + hid)
    for bid in ["B-finish-study", "B-finish-representations", "B-finish-repetition"]:
        if root / "research/branches" / (bid + ".json") not in paths:
            raise ValueError("Missing canonical branch: " + bid)
    return paths


def prepare(root: Path, raw: Path, readiness: Path, run_id: str,
            python: Path | None = None) -> tuple[Path, list[str]]:
    root = root.expanduser().resolve()
    raw = raw.expanduser().resolve()
    readiness = readiness.expanduser()
    readiness = (readiness if readiness.is_absolute() else root / readiness).resolve()
    identifier(run_id)
    if not run_id.startswith("R-finish-"):
        raise ValueError("Choose a fresh R-finish-* run identity")
    output = root / "configs/completion/science" / run_id
    for occupied in [output, root / "runs" / run_id, root / "runs/.staging" / run_id,
                     root / "inputs" / run_id]:
        if occupied.exists() or occupied.is_symlink():
            raise ValueError(f"Identity already used; choose a fresh --run-id: {occupied}")
    warmup = root / "engines/tlab_warmup"
    python = (python or warmup / ".venv/bin/python").expanduser().absolute()
    for path in [raw, readiness, python]:
        if not path.is_file():
            raise ValueError(f"Required local input missing: {path}; no experiment executed")
    b = bridge_for(root)
    preserved = {f["path"]: f["sha256"]
                 for f in load(root / "reproducibility/engine-source.json")["files"]}
    if b.files_for(warmup) != preserved:
        raise ValueError("Frozen engine inventory differs from reproducibility/engine-source.json")
    archive = warmup / "provenance/iteration0/results.tar.gz"
    anchor = b.anchor_config(archive)
    if anchor != load(root / "reproducibility/anchor-config.json"):
        raise ValueError("Archived anchor differs from accepted scientific boundary")
    print("Capturing the local environment and verifying the pinned raw file...", file=sys.stderr)
    source = b.capture_source(warmup, python)
    features = source["environment"]["features"]
    if features != load(root / "reproducibility/feature-manifest.json"):
        raise ValueError("Realized feature dictionary differs from accepted manifest")
    raw_hash = b.sha(raw)
    if raw.stat().st_size != anchor["source"]["bytes"] or raw_hash != anchor["source"]["sha256"]:
        raise ValueError("Raw bytes/SHA256 differ from the pinned anchor")
    checkpoint = load(readiness)
    if (checkpoint.get("ready") is not True or checkpoint.get("test_scored") is not False
            or checkpoint.get("phase") != "iteration1" or checkpoint.get("stop_reason") is not None):
        raise ValueError("Readiness must be a successful, unscored iteration1 checkpoint")
    # Ask the unchanged native code for its own hash; its inventory excludes configs.
    probe = subprocess.run([str(python), "-B", "-c",
                            "from tlab_warmup.experiment_io import code_state; "
                            "from pathlib import Path; print(code_state(Path.cwd())['tree_sha256'])"],
                           cwd=warmup, env=b.process_env(warmup), text=True,
                           capture_output=True, timeout=30, check=True)
    if checkpoint.get("code_hash") != probe.stdout.strip():
        raise ValueError("Readiness code_hash differs from the unchanged executing engine")
    cfg = load(warmup / "configs/experiments.json")
    cfg.update(BATCH_CHANGES)
    b.validate_config(cfg)
    bounds = anchor["resolved_boundaries"]
    scientific = {
        "experiment_id": "E-finish-study", "branch_id": "B-finish-study",
        "hypothesis_ids": HYPOTHESES, "phase": "development", "runner_id": "warmup",
        "protocol": {"id": "warmup-development-v1", "candidates": ["U20", "P20"],
                     "history": "strict-past-complete",
                     "train_end_timestamp": bounds["train_end_exclusive"],
                     "development_end_timestamp": bounds["development_end_exclusive"],
                     "timestamp_unit": "seconds", "final_test": "not-scored"},
        "parameters": {"batch_phase": "overnight", "controls_only": False,
                       "seed": anchor["seed"], "anchor_config_sha256": b.digest(anchor),
                       "feature_manifest_sha256": b.digest(features), "batch_config": cfg}}
    manifests = [("source", warmup, source),
                 ("anchor", archive, {"kind": "warmup_anchor_v1", "sha256": b.sha(archive)}),
                 ("raw", raw, {"kind": "warmup_raw_v1", "sha256": raw_hash}),
                 ("readiness", readiness, {"kind": "warmup_readiness_v1", "sha256": b.sha(readiness)})]
    refs = [{"id": name, "local_path": str(path),
             "manifest": str(output / (name + "-manifest.json"))}
            for name, path, _ in manifests]
    experiment = {"schema_version": 1, "scientific": scientific, "runtime": {
        "command": [str(python), str(root / "runners/warmup/runner.py")],
        "code_paths": [str(root / "runners/warmup/runner.py"),
                       str(root / "runners/warmup/bridge.py")],
        "data_refs": refs, "research_paths": [str(p) for p in record_paths(root)],
        "resources": {"timeout_seconds": 1800, "memory_limit_bytes": None}}}
    validate(root, "experiment_config", experiment)
    # This read-only bridge check rejects unsupported or inactive choices before writing.
    # Identity values here are validation-envelope hashes, not infrastructure identities.
    request = {"schema_version": 1, "run_id": run_id, "scientific": scientific,
               "identity": {k: b.digest(scientific) for k in
                            ["scientific_sha256", "code_sha256", "data_sha256"]},
               "data": [{"id": n, "local_path": str(p), "manifest": m}
                        for n, p, m in manifests]}
    b.validate(request)
    output.mkdir(parents=True, exist_ok=False)
    for name, _, manifest in manifests:
        b.write(output / (name + "-manifest.json"), manifest)
    config_path = output / "experiment.json"
    b.write(config_path, experiment)
    command = ["uv", "run", "--frozen", "researchflow", "experiment", "--root", str(root),
               "--config", str(config_path), "--run-id", run_id]
    return config_path, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, default=Path("reproducibility/baseline-readiness.json"))
    parser.add_argument("--run-id", required=True,
                        help="Fresh R-finish-* ID; R-finish-study-001 is only an example")
    parser.add_argument("--python", type=Path, help="Explicit installed engine interpreter")
    args = parser.parse_args()
    try:
        config, command = prepare(args.root, args.data, args.readiness, args.run_id, args.python)
        root = args.root.expanduser().resolve()
        print(f"Prepared {config}; no experiment executed.")
        print("Launch exactly once: " + shlex.join(command))
        print(f"Expected committed output: {root / 'runs' / args.run_id}")
        print(f"Failed/interrupted staging: {root / 'runs/.staging' / args.run_id}")
        print("Native outputs: payload/workspace/runs/" + args.run_id)
        print("After success: " + shlex.join([
            "uv", "run", "--frozen", "python", str(root / "tools/analyze_completion_study.py"),
            "--root", str(root), "--run", str(root / "runs" / args.run_id),
            "--analysis-id", identifier("A-finish-" + args.run_id.removeprefix("R-finish-"))]))
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
