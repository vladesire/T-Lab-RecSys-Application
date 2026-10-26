"""Standalone adapter for the inspected tlab_warmup.batch interface.

Only run_warmup_bridge imports researchflow, lazily, to translate boundary errors.
The CLI uses execute directly and needs numpy plus the declared upstream environment.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Any, Mapping

import numpy as np


class BridgeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(ok, message, code="CONTRACT"):
    if not ok:
        raise BridgeError(code, message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected), f"{name}: expected exactly {sorted(expected)}")


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute(), "Input paths must be absolute; resolve against the caller project first")
    return Path(value).resolve()


def files_for(project):
    paths = [*project.glob("src/**/*.py"), *project.glob("tests/*.py"),
             *project.glob("configs/*.json"), project / "pyproject.toml", project / "uv.lock"]
    require(all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(project) for p in paths), "Missing/symlinked upstream files", "IDENTITY")
    require((project / "src/tlab_warmup/batch.py") in paths, "Not a warmup batch project")
    return {str(p.relative_to(project)): sha(p) for p in sorted(paths)}


def process_env(project):
    env = dict(os.environ)
    # Pin imports to the verified copy. No install, acquisition or pycache mutation.
    env.update(PYTHONPATH=str(project / "src"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               GIT_CEILING_DIRECTORIES=str(project.parent))
    return env


def probe(python, project):
    code = '''import importlib.metadata as m,json,platform,sys
from tlab_warmup.representations import manifest
print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
"machine":platform.machine(),"packages":{p:m.version(p) for p in
["numpy","pyarrow","scikit-learn","scipy","threadpoolctl","psutil","joblib"]},"features":manifest()}))'''
    p = subprocess.run([str(python), "-B", "-c", code], cwd=project, env=process_env(project),
                       text=True, capture_output=True, timeout=30)
    require(p.returncode == 0, "Upstream environment probe failed: " + p.stderr[-3000:], "DEPENDENCY")
    return json.loads(p.stdout)


def capture_source(project, python):
    """Explicit setup only: record source and the actual chosen environment."""
    project = absolute(str(project))
    require(Path(python).is_absolute(), "Python executable must be absolute")
    python = Path(python).absolute()
    files = files_for(project)
    return {"kind": "warmup_source_v1", "files": files, "tree_sha256": digest(files),
            "python": str(python), "python_sha256": sha(python), "environment": probe(python, project)}


def anchor_config(path):
    with tarfile.open(path) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "resolved_config.json"]
        require(len(members) == 1, "Anchor needs one resolved_config.json")
        require(members[0].size < 1024 * 1024, "Oversized anchor config")
        return json.load(tar.extractfile(members[0]))


CONFIG_KEYS = set("schema threads data wall_hours per_job_seconds max_retries target_rss_gib stop_rss_gib minimum_available_gib low_memory_seconds maximum_selected_rows bootstrap_replicates search_rounds search_fit_cap search_min_gain search_min_queries search_min_users search_min_novel additional_model_seed_offsets iteration1_output panels optional_artist optional_3000_user_panel optional_100_alternatives".split())


def validate_config(cfg):
    keys(cfg, CONFIG_KEYS, "batch_config")
    require(cfg["schema"] == "tlab-experiments-v1", "Unsupported batch schema")
    require(cfg["data"] is None and cfg["iteration1_output"] == "runs/iteration1", "Data/readiness paths must come from declared input references")
    keys(cfg["panels"], {"iteration1", "calibration", "main"}, "panels")
    for panel in cfg["panels"].values():
        keys(panel, {"users", "train", "development"}, "panel")
        require(all(type(v) is int and v > 0 for v in panel.values()), "Panel caps must be positive integers")
    for key in ["threads", "per_job_seconds", "low_memory_seconds", "maximum_selected_rows", "bootstrap_replicates", "search_fit_cap", "search_rounds", "search_min_queries", "search_min_users", "search_min_novel"]:
        require(type(cfg[key]) is int and cfg[key] > 0, f"Invalid {key}")
    for key in ["wall_hours", "target_rss_gib", "stop_rss_gib", "minimum_available_gib", "search_min_gain"]:
        require(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] > 0, f"Invalid {key}")
    require(type(cfg["max_retries"]) is int and 0 <= cfg["max_retries"] <= 1, "Invalid retries")
    require(cfg["threads"] <= 14 and cfg["wall_hours"] <= 8 and cfg["search_fit_cap"] <= 24 and cfg["search_rounds"] <= 3, "Upstream budget exceeded")
    require(cfg["target_rss_gib"] < cfg["stop_rss_gib"] <= 10, "Invalid RSS guards")
    offsets = cfg["additional_model_seed_offsets"]
    require(isinstance(offsets, list) and all(type(x) is int and x > 0 for x in offsets) and len(set(offsets)) == len(offsets), "Invalid seed offsets")
    for key, expected in {"optional_artist": "blocked_pending_mapping_semantics_validation", "optional_3000_user_panel": "blocked_pending_secondary_scale_calibration", "optional_100_alternatives": "blocked_pending_variable_cardinality_validation"}.items():
        require(cfg[key] == expected, f"Unsupported optional setting: {key}")


def validate(request):
    keys(request, {"schema_version", "run_id", "scientific", "data", "identity"}, "request")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1, "Unsupported request version")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request["run_id"]) is not None, "Invalid run_id")
    sci = request["scientific"]
    keys(sci, {"experiment_id", "branch_id", "hypothesis_ids", "phase", "runner_id", "protocol", "parameters"}, "scientific")
    for field in ("experiment_id", "branch_id", "runner_id"):
        require(isinstance(sci[field], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sci[field]) is not None, f"Invalid {field}")
    require(isinstance(sci["hypothesis_ids"], list) and all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", x) is not None for x in sci["hypothesis_ids"]), "Invalid hypothesis IDs")
    keys(request["identity"], {"scientific_sha256", "code_sha256", "data_sha256"}, "identity")
    require(all(isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x) is not None for x in request["identity"].values()), "Invalid identity hashes")
    require(sci["phase"] in ("fixture", "development"), "This upstream bridge never scores final targets")
    p = sci["parameters"]
    keys(p, {"batch_phase", "controls_only", "seed", "anchor_config_sha256", "feature_manifest_sha256", "batch_config"}, "scientific.parameters (unsupported parameter)")
    require(p["batch_phase"] in ("iteration1", "overnight"), "Unsupported batch phase")
    require(type(p["controls_only"]) is bool, "controls_only must be boolean")
    require(p["batch_phase"] != "iteration1" or not p["controls_only"], "controls_only has no effect in iteration1; unsupported")
    validate_config(p["batch_config"])
    refs = {}
    for ref in request["data"]:
        keys(ref, {"id", "local_path", "manifest"}, "data reference")
        kind = ref["manifest"].get("kind")
        require(kind not in refs and kind in {"warmup_source_v1", "warmup_anchor_v1", "warmup_raw_v1", "warmup_readiness_v1"}, "Unknown/duplicate input kind")
        refs[kind] = (absolute(ref["local_path"]), ref["manifest"])
    needed = {"warmup_source_v1", "warmup_anchor_v1", "warmup_raw_v1"}
    if p["batch_phase"] == "overnight":
        needed.add("warmup_readiness_v1")
    require(set(refs) == needed, "Incorrect explicit warmup input manifests")
    project, source = refs["warmup_source_v1"]
    keys(source, {"kind", "files", "tree_sha256", "python", "python_sha256", "environment"}, "source manifest")
    require(files_for(project) == source["files"] and digest(source["files"]) == source["tree_sha256"], "Upstream code/config/lock fingerprint changed", "IDENTITY")
    python = absolute(source["python"])
    require(sha(python) == source["python_sha256"], "Upstream Python executable changed", "IDENTITY")
    for kind in needed - {"warmup_source_v1"}:
        path, spec = refs[kind]
        keys(spec, {"kind", "sha256"}, kind)
        require(path.is_file() and sha(path) == spec["sha256"], f"{kind} fingerprint mismatch", "IDENTITY")
    baseline = json.loads((project / "configs/experiments.json").read_text())
    active = {"threads", "wall_hours", "per_job_seconds", "max_retries", "target_rss_gib", "stop_rss_gib", "minimum_available_gib", "low_memory_seconds", "maximum_selected_rows", "bootstrap_replicates", "panels"}
    if p["batch_phase"] == "overnight" and not p["controls_only"]:
        active |= {"search_rounds", "search_fit_cap", "search_min_gain", "search_min_queries", "search_min_users", "search_min_novel", "additional_model_seed_offsets"}
    for key in CONFIG_KEYS - active:
        require(p["batch_config"][key] == baseline[key], f"Setting {key} is inactive/unsupported in this batch mode")
    active_panels = {"iteration1"} if p["batch_phase"] == "iteration1" else {"calibration", "main"}
    for panel in {"iteration1", "calibration", "main"} - active_panels:
        require(p["batch_config"]["panels"][panel] == baseline["panels"][panel], f"Panel {panel} is inactive in this batch mode")
    anchor = anchor_config(refs["warmup_anchor_v1"][0])
    require(digest(anchor) == p["anchor_config_sha256"], "Frozen anchor configuration mismatch", "IDENTITY")
    require(type(p["seed"]) is int and p["seed"] == anchor["seed"], "Unsupported seed override; batch seed comes from anchor")
    require(digest(source["environment"]["features"]) == p["feature_manifest_sha256"], "Unsupported feature selection override", "IDENTITY")
    require(anchor["alternatives"] == 20 and anchor["ratio_smoothing"] == 1, "Batch supports 20 alternatives and smoothing=1 only")
    require(isinstance(anchor["source"]["revision"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", anchor["source"]["revision"]) is not None, "Unsafe upstream source revision")
    raw = refs["warmup_raw_v1"][0]
    require(raw.stat().st_size == anchor["source"]["bytes"] and sha(raw) == anchor["source"]["sha256"], "Raw data differs from pinned anchor", "IDENTITY")
    require(sci["phase"] != "fixture" or anchor["source"]["synthetic"] is True, "Fixture phase requires synthetic data")
    bounds = anchor["resolved_boundaries"]
    expected_protocol = {"id": "warmup-development-v1", "candidates": ["U20", "P20"], "history": "strict-past-complete",
                         "train_end_timestamp": bounds["train_end_exclusive"], "development_end_timestamp": bounds["development_end_exclusive"],
                         "timestamp_unit": "seconds", "final_test": "not-scored"}
    require(sci["protocol"] == expected_protocol, "Unsupported protocol/history/cutoff override")
    if not anchor["source"]["synthetic"]:
        require((bounds["train_end_exclusive"], bounds["development_end_exclusive"]) == (20800000, 23400000), "Unsupported real-data cutoffs")
    if p["batch_phase"] == "iteration1":
        panel = p["batch_config"]["panels"]["iteration1"]
        require(panel["users"] == anchor["users"] and panel["train"] == anchor["training_queries"] and panel["development"] >= anchor["development_queries"], "Iteration1 must retain anchor cohort and queries")
    return refs, anchor


def inside(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), "Upstream output reference escaped selected workspace", "IDENTITY")
    return path


def complete(directory):
    marker = json.loads((directory / "COMPLETE.json").read_text())
    files = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "COMPLETE.json" and not p.name.endswith(".partial")}
    require(marker["files"] == files, "Upstream completed output checksum mismatch", "IDENTITY")


def ranking_arrays(queries, scores, familiar, seed, protocol, tie_function=None):
    """Normalize the upstream saved order without re-sampling or re-scoring."""
    n = len(queries)
    require(n > 0 and scores.shape == familiar.shape == (n, 21), "Invalid upstream prediction shape")
    require(np.issubdtype(scores.dtype, np.number) and not np.iscomplexobj(scores) and np.isfinite(scores).all(), "Non-finite/nonreal scores")
    require(familiar.dtype == np.bool_, "Familiarity must come from strict-past pair counts")
    require(len({str(q["query_id"]) for q in queries}) == n, "Duplicate query IDs")
    rows = {k: [] for k in ("query_id", "user_id", "item_id", "timestamp", "label", "tie_key", "is_familiar", "is_repeat_target")}
    query_records, candidate_records = [], []
    for j, q in enumerate(queries):
        items = q["candidates"]
        require(len(items) == 21 and len(set(items)) == 21 and items.count(q["positive"]) == 1, "Invalid candidates/labels")
        require(all(type(q[k]) is int and q[k] >= 0 for k in ("query_id", "uid", "timestamp", "positive")) and all(type(x) is int and x >= 0 for x in items), "Invalid upstream integer IDs/time")
        require(protocol["train_end_timestamp"] <= q["timestamp"] < protocol["development_end_timestamp"], "Final/out-of-development target prohibited")
        repeat = bool(familiar[j, items.index(q["positive"])])
        ties = q.get("tie_keys")
        if ties is None:
            require(tie_function is not None, "Historical tie-key function required")
            ties = [int(tie_function(seed, q["uid"], q["timestamp"], it)) for it in items]
        require(len(ties) == 21 and all(type(x) is int and 0 <= x < 2**64 for x in ties), "Invalid historical tie keys")
        qr = {"query_id": str(q["query_id"]), "user_id": str(q["uid"]), "target_item_id": str(q["positive"]), "timestamp": q["timestamp"], "is_repeat_target": repeat}
        query_records.append(qr)
        cr = [{"item_id": str(it), "is_familiar": bool(familiar[j, k]), "tie_key": ties[k]} for k, it in enumerate(items)]
        candidate_records.append({"query_id": qr["query_id"], "candidate_set_sha256": digest(sorted(cr, key=lambda r: r["item_id"]))})
        for k, it in enumerate(items):
            values = [str(q["query_id"]), str(q["uid"]), str(it), q["timestamp"], it == q["positive"], ties[k], bool(familiar[j, k]), repeat]
            for name, value in zip(rows, values):
                rows[name].append(value)
    arrays = {k: np.asarray(v, dtype=str if k.endswith("_id") else np.uint64 if k in ("timestamp", "tie_key") else bool) for k, v in rows.items()}
    arrays["score"] = scores.reshape(-1).copy()
    identities = {"protocol_sha256": digest(protocol), "query_set_sha256": digest(sorted(query_records, key=lambda r: r["query_id"])),
                  "candidate_set_sha256": digest(sorted(candidate_records, key=lambda r: r["query_id"]))}
    return arrays, identities


def normalize(run, workspace, output, protocol, seed, expected_features):
    import types
    # Execute the verified upstream PURE core source without creating bytecode.
    # Do not duplicate its label-independent historical tie-key algorithm.
    core = types.ModuleType("m19_verified_warmup_core")
    core_path = workspace / "src/tlab_warmup/core.py"
    exec(compile(core_path.read_bytes(), str(core_path), "exec"), core.__dict__)
    ledger = json.loads((run / "ledger.json").read_text())
    predictions, coverage, costs = [], [], []
    checked_preparations = set()
    for job_id, job in sorted(ledger.items()):
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id), "Unsafe job ID")
        costs.append({"job_id": job_id, "status": job["status"], "attempts": job.get("attempts", []), "resources": job.get("resources", {})})
        req = job["request"]
        if req["kind"] not in ("fit", "search") or job["status"] != "completed":
            continue
        directory = inside(job["output"], workspace)
        complete(directory)
        prep = inside(req["protocol_result"], workspace)
        if prep not in checked_preparations:
            complete(prep)
            checked_preparations.add(prep)
        meta = json.loads((prep / "result.json").read_text())
        spec = json.loads((prep / "feature_manifest.json").read_text())
        require(spec == expected_features, "Realized feature dictionary changed", "IDENTITY")
        require(meta["protocol"] in protocol["candidates"], "Unexpected candidate protocol")
        require(meta["config"]["seed"] == seed and meta["config"]["train_end_timestamp"] == protocol["train_end_timestamp"] and meta["config"]["development_end_timestamp"] == protocol["development_end_timestamp"] and meta["test_scored"] is False, "Realized seed/cutoffs/final rules changed")
        with gzip.open(prep / "queries.json.gz", "rt") as stream:
            all_queries = json.load(stream)
        require(digest(all_queries) == meta["query_candidate_hash"], "Saved query/candidate identity mismatch", "IDENTITY")
        # Upstream rows are train then development, each with 21 candidates.
        ntrain = sum(q["split"] == "train" for q in all_queries)
        require(all(q["split"] == "train" for q in all_queries[:ntrain]) and all(q["split"] == "development" for q in all_queries[ntrain:]), "Invalid query ordering")
        queries = all_queries[ntrain:]
        matrix = np.load(prep / "features.npy", mmap_mode="r", allow_pickle=False)
        require(matrix.shape == (len(all_queries)*21, len(spec["names"])), "Feature matrix alignment mismatch")
        pair = matrix[ntrain*21:, spec["names"].index("user_item_count")]
        require(np.isfinite(pair).all() and (pair >= 0).all(), "Invalid pair counts")
        familiar = (pair > 0).reshape(-1, 21)
        result = json.loads((directory / "result.json").read_text())
        require(result["seed"] == req["seed"] and result["protocol"] == meta["protocol"] and result["query_candidate_hash"] == meta["query_candidate_hash"], "Realized method identity mismatch", "IDENTITY")
        names = result.get("feature_names", result.get("selected_features"))
        if names is not None:
            require(len(names) == len(set(names)) and set(names) <= set(spec["names"]), "Unsupported realized features")
            if req["kind"] == "fit":
                expected = spec[req["feature_set"]] if "selection_result" not in req else json.loads((inside(req["selection_result"], workspace)/"result.json").read_text())["selected_features"]
                require(names == expected, "Feature selection/order changed", "IDENTITY")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            require(np.array_equal(saved["query_ids"], [q["query_id"] for q in queries]), "Prediction query alignment mismatch")
            arrays, identity = ranking_arrays(queries, saved["scores"], familiar, seed, protocol, core.tie_key)
            require(np.array_equal(saved["repeat"], arrays["is_repeat_target"].reshape(-1, 21)[:, 0]), "Repeat flag mismatch")
        path = f"predictions/{job_id}.npz"
        np.savez_compressed(output / path, **arrays)
        predictions.append({"id": job_id, "method_id": job_id, "protocol_id": meta["protocol"], "split": "development", "format": "ranking_npz_v1", "path": path,
                            "feature_count": len(names) if names is not None else None, "identity": identity,
                            "tags": {"upstream_method": result["method"], "model_seed": result["seed"], "tie_seed": seed, "feature_names": names}})
        coverage.append({"prediction_id": job_id, "queries": len(queries), "users": len({q["uid"] for q in queries}), "familiarity": meta.get("familiarity", {})})
    require(predictions, "No completed prediction jobs", "EXECUTION")
    return predictions, coverage, costs


def execute(request, output_dir):
    refs, anchor = validate(request)
    require(Path(output_dir).is_absolute(), "Output must be absolute")
    output = Path(output_dir).resolve()
    project, source = refs["warmup_source_v1"]
    # Reject overlap in either direction, including symlink aliases and ancestors.
    for path, _ in refs.values():
        require(not output.is_relative_to(path) and not path.is_relative_to(output), "Output overlaps an input/project/raw-data/old output path")
    require(not output.exists() or output.is_dir(), "Output is not a directory")
    marker = output / "bridge-input.json"
    if output.exists() and any(output.iterdir()):
        require(marker.is_file() and json.loads(marker.read_text()) == request and not (output/"result.json").exists(), "Existing output cannot be overwritten; only identical interrupted bridge work can resume", "IDENTITY")
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    # Output lock serializes setup/normalization as well as upstream job execution.
    with (output / ".bridge.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BridgeError("EXECUTION", "Another bridge owns this output") from exc
        marker = output / "bridge-input.json"
        if any(x.name != ".bridge.lock" for x in output.iterdir()):
            require(marker.is_file() and json.loads(marker.read_text()) == request and not (output/"result.json").exists(), "Existing output cannot be overwritten; only identical interrupted bridge work can resume", "IDENTITY")
        workspace = output / "workspace"
        if not marker.exists():
            workspace.mkdir()
            for rel in source["files"]:
                target = workspace / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / rel, target)
            write(marker, request)
        require(files_for(workspace) == source["files"], "Staged source/config/lock changed", "IDENTITY")
        actual = probe(source["python"], workspace)
        require(actual == source["environment"], "Declared upstream environment changed", "IDENTITY")
        p = request["scientific"]["parameters"]
        cfg = copy.deepcopy(p["batch_config"])
        cfg["data"] = str(refs["warmup_raw_v1"][0])
        if p["batch_phase"] == "overnight":
            readiness = refs["warmup_readiness_v1"][0]
            dest = workspace / "runs/iteration1"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(readiness, dest / "readiness.json")
        config_path = output / "resolved-batch-config.json"
        write(config_path, cfg)
        run = workspace / "runs" / request["run_id"]
        require(run.name != "iteration1" or p["batch_phase"] == "iteration1", "Run ID collides with prior readiness")
        command = [source["python"], "-B", "-m", "tlab_warmup.batch", p["batch_phase"], "--config", str(config_path), "--anchor", str(refs["warmup_anchor_v1"][0]), "--data", cfg["data"], "--output", str(run)]
        if p["controls_only"]:
            command.append("--controls-only")
        provenance = {"source_project": str(project), "source_manifest": source, "anchor": {"path": str(refs["warmup_anchor_v1"][0]), "sha256": sha(refs["warmup_anchor_v1"][0]), "config": anchor},
                      "raw": {"path": cfg["data"], "sha256": sha(cfg["data"])}, "environment": actual, "command": command,
                      "config": cfg, "config_sha256": digest(cfg), "request_identity": request["identity"],
                      "adapter_files": {f.name: sha(f) for f in (Path(__file__), Path(__file__).with_name("runner.py"))},
                      "tie_keys": "preserve saved keys when present; otherwise call verified upstream core.tie_key(seed,uid,timestamp,item), without labels",
                      "adapter_environment": {"python": __import__("sys").version, "numpy": np.__version__}}
        write(output / "provenance.json", provenance)
        with (output / "upstream.log").open("a") as stream:
            process = subprocess.run(command, cwd=workspace, env=process_env(workspace), stdout=stream, stderr=subprocess.STDOUT)
        require(process.returncode == 0, "Warmup failed/stopped; inspect upstream.log; identical request/output resumes native completed jobs", "EXECUTION")
        require(files_for(project) == source["files"] and files_for(workspace) == source["files"] and probe(source["python"], workspace) == actual, "Source/environment changed during execution", "IDENTITY")
        require(sha(refs["warmup_anchor_v1"][0]) == refs["warmup_anchor_v1"][1]["sha256"] and sha(cfg["data"]) == refs["warmup_raw_v1"][1]["sha256"], "Input changed during execution", "IDENTITY")
        ledger = json.loads((run / "ledger.json").read_text())
        for job in ledger.values():
            req = job["request"]
            require(req["config"] == cfg, "Realized job configuration changed", "IDENTITY")
            allowed_seeds = {p["seed"], *(p["seed"] + offset for offset in cfg["additional_model_seed_offsets"])}
            require(req["seed"] in allowed_seeds, "Realized job seed changed", "IDENTITY")
        readiness = json.loads((run / "readiness.json").read_text())
        require(readiness["ready"] is True and readiness["test_scored"] is False, "Upstream readiness/final-data rule failed", "EXECUTION")
        (output / "predictions").mkdir(exist_ok=True)
        predictions, coverage, costs = normalize(run, workspace, output, request["scientific"]["protocol"], p["seed"], actual["features"])
        anchor_result = run / "jobs/U20_original/result.json"
        anchor_pass = p["batch_phase"] == "iteration1" and json.loads(anchor_result.read_text()).get("anchor_prediction_max_absolute_difference", float("inf")) <= 1e-8
        require(p["batch_phase"] != "iteration1" or anchor_pass, "Archived anchor prediction criterion not satisfied", "EXECUTION")
        provenance["upstream_plan"] = json.loads((run / "plan_manifest.json").read_text())
        archive = run.with_name(run.name + "-results.tar.gz")
        require(archive.is_file(), "Missing upstream archive", "EXECUTION")
        provenance["result_archive"] = {"path": str(archive.relative_to(output)), "sha256": sha(archive)}
        write(output / "provenance.json", provenance)
        result = {"schema_version": 1, "kind": "ranking", "predictions": predictions, "metrics": [], "coverage": coverage, "costs": costs,
                  "checks": {"passed": True, "items": [{"id": "upstream-anchor", "status": "passed" if anchor_pass else "unverified", "detail": ("Synthetic anchor" if anchor["source"]["synthetic"] else "Actual-data anchor") + ": upstream U20 original max absolute difference <=1e-8" if anchor_pass else "Anchor reproduction is required in iteration1, not rerun by overnight", "evidence": ["provenance.json", "upstream.log"]}]},
                  "limitations": ["Synthetic executions establish adapter behavior only; actual Mac anchor acceptance remains required.", "Sampled candidates and reused development data are not a pristine final test.", "Full feature dictionary is computed upstream; selected column count does not measure feature computation savings.", "Upstream metrics use integer item fallback; port uses lexical IDs if a 64-bit tie-key collision occurs."],
                  "artifacts": [{"path": "provenance.json", "kind": "provenance", "description": "Verified upstream source/lock/environment, anchor, effective config, command and archive identities"}, {"path": str(archive.relative_to(output)), "kind": "upstream_archive", "description": "Unmodified upstream batch export"}, {"path": "upstream.log", "kind": "log", "description": "Native runner log"}]}
        write(output / "result.json", result)
        return result


def run_warmup_bridge(request: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    from researchflow.contracts import WorkflowError
    try:
        return execute(dict(request), output_dir)
    except BridgeError as exc:
        raise WorkflowError(exc.code, str(exc), {}) from exc
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise WorkflowError("EXECUTION", str(exc), {}) from exc
