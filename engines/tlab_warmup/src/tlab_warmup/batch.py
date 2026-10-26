"""Sequential, resumable local experiments. No services or dynamic proposals."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import psutil

from .experiment_io import atomic_json, code_state, digest, now, outputs_manifest, valid_output
from .experiment_jobs import DISPATCH, Blocked, read_anchor
from .pipeline import ROOT, environment
from .representations import manifest


def run_worker(request_path, output):
    request = json.loads(Path(request_path).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}; job-level checkpoint only")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if code_state(ROOT)["tree_sha256"] != request["expected_code_hash"]:
            raise Blocked("Code changed after the run was frozen")
        import pyarrow as pa
        pa.set_cpu_count(request["config"]["threads"])
        pa.set_io_thread_count(request["config"]["threads"])
        if request["kind"] == "gate":
            calibration_gate(request, out)
        else:
            DISPATCH[request["kind"]](request, out)
        if code_state(ROOT)["tree_sha256"] != request["expected_code_hash"]:
            raise Blocked("Code changed during execution; result not committed")
        atomic_json(out / "COMPLETE.json", {"completed": now(), "request_hash": digest(request),
                                           "files": outputs_manifest(out)})
        return 0
    except BaseException as exc:
        (out / "error.log").write_text(traceback.format_exc())
        atomic_json(out / "failure.json", {"error": repr(exc), "type": type(exc).__name__,
            "blocked": isinstance(exc, Blocked), "interrupted": isinstance(exc, (InterruptedError, KeyboardInterrupt)),
            "retryable": isinstance(exc, (ConnectionError, TimeoutError)), "recorded": now()})
        return 2


def calibration_gate(request, out):
    run = Path(request["run"])
    ledger = json.loads((run / "ledger.json").read_text())
    feature_jobs = [ledger[x] for x in request["feature_jobs"]]
    peak = max(j["resources"]["peak_tree_rss_bytes"] for j in [*feature_jobs, ledger["calibration_data"], ledger[request["fit_job"]]])
    cfg = request["config"]
    small, large = cfg["panels"]["calibration"], cfg["panels"]["main"]
    scale = large["users"] / small["users"]
    matrix_bytes = (large["train"] + large["development"])*21*len(manifest()["names"])*8
    base = 200 * 1024**2
    predicted = base + max(0, peak-base)*scale + 2*matrix_bytes
    data_seconds = sum(a.get("wall_seconds", 0) for a in ledger["calibration_data"]["attempts"])
    seconds = data_seconds*max(1, scale)
    for j in feature_jobs:
        r = json.loads((Path(j["output"]) / "result.json").read_text())
        seconds += r["feature_seconds"]*scale + r["sampling_seconds"]*max(scale,
            (large["train"] + large["development"]) / (small["train"] + small["development"]))
    fit = ledger[request["fit_job"]]
    measured_fit = json.loads((Path(fit["output"]) / "result.json").read_text())["fit_seconds"]
    seconds += measured_fit * large["train"] / small["train"] * 70
    startup_per_fit = max(0, fit["attempts"][-1]["wall_seconds"]-measured_fit)
    seconds += startup_per_fit*70
    estimates = {"calibration_process_tree_peak_gib": peak/1024**3,
        "planning_peak_gib": predicted/1024**3, "main_feature_matrix_gib_per_protocol": matrix_bytes/1024**3,
        "rough_runtime_seconds_range": [max(1, seconds/3), max(5, seconds*4)],
        "calibration_data_prep_seconds": data_seconds, "included_fit_startup_seconds_per_job": startup_per_fit,
        "limitations": "Planning envelope from small local calibration, not a validated scaling law; includes sampling/materialization and repeated fits, excludes acquisition/setup; guards remain necessary",
        "search_budget_allowed": seconds*4 < cfg["wall_hours"]*3600,
        "maximum_total_wall_hours": cfg["wall_hours"]}
    atomic_json(out / "calibration.json", estimates)
    if predicted > cfg["target_rss_gib"]*1024**3:
        raise Blocked("Calibration exceeds target RSS estimate; reduce query caps in a new manifest, never truncate histories")
    atomic_json(out / "result.json", estimates)


def build_jobs(run, cfg, phase, anchor, code, *, controls_only=False):
    cache = ROOT / "data" / "batches" / digest({"run": str(run), "config": cfg, "code": code["tree_sha256"]})[:20]
    jobs = []
    def add(name, kind, priority, deps=(), *, cache_output=False, **extra):
        output = cache / name if cache_output else run / "jobs" / name
        req = {"kind": kind, "config": cfg, "anchor": str(anchor), "seed": read_anchor_seed,
               "expected_code_hash": code["tree_sha256"],
               **extra}
        item = {"id": name, "priority": priority, "depends_on": list(deps), "request": req,
                "output": str(output), "status": "pending", "attempts": []}
        jobs.append(item)
        return output
    read_anchor_seed = read_anchor(anchor)["config"]["seed"]
    source = add("source", "source", 0)
    if phase == "overnight":
        caldata = add("calibration_data", "dataset", 0, ["source"], cache_output=True,
                      panel="calibration", source_result=str(source))
        calfeatures = []
        for protocol in ["U20", "P20"]:
            p = add("calibration_"+protocol, "protocol", 0, ["calibration_data"], cache_output=True,
                    dataset_result=str(caldata), protocol=protocol)
            calfeatures.append("calibration_"+protocol)
        calfit = add("calibration_fit", "fit", 0, ["calibration_P20"],
                     protocol_result=str(p), method="manual", feature_set="manual")
        gate = add("calibration_gate", "gate", 0, [*calfeatures, "calibration_fit"],
                  run=str(run), feature_jobs=calfeatures, fit_job="calibration_fit")
        panel, data_deps = "main", ["source", "calibration_gate"]
    else:
        panel, data_deps = "iteration1", ["source"]
    data = add("data_"+panel, "dataset", 0, data_deps, cache_output=True,
               panel=panel, source_result=str(source))
    protocols = ["U20", "P20"]
    method_paths = {}
    protocol_paths = {}
    for protocol in protocols:
        protocol_id = "prepare_"+protocol
        prep = add(protocol_id, "protocol", 0, ["data_"+panel], cache_output=True,
                   dataset_result=str(data), protocol=protocol)
        protocol_paths[protocol] = prep
        method_paths[protocol] = {}
        simple = ["random", "popularity", "recency_rule"]
        if phase == "overnight":
            simple += ["personal_frequency", "decay_popularity"]
        for method in simple + ["original", "no_pair"]:
            path = add(f"{protocol}_{method}", "fit", 1, [protocol_id],
                       protocol_result=str(prep), method=method, feature_set=method,
                       verify_anchor=phase=="iteration1" and protocol=="U20" and method=="original")
            method_paths[protocol][method] = str(path)
        if phase == "overnight" and not controls_only:
            for method in ["manual", "decay", "all"]:
                path = add(f"{protocol}_{method}", "fit", 2, [protocol_id],
                           protocol_result=str(prep), method=method, feature_set=method)
                method_paths[protocol][method] = str(path)
    if phase == "overnight" and not controls_only:
        prep = protocol_paths["P20"]
        forward = add("P20_forward", "search", 3, ["P20_manual", "calibration_gate"],
                      protocol_result=str(prep), method="forward", manual_result=method_paths["P20"]["manual"])
        random = add("P20_random_search", "search", 3, ["P20_forward"], protocol_result=str(prep),
                     method="random_search", manual_result=method_paths["P20"]["manual"], forward_result=str(forward))
        method_paths["P20"].update(forward=str(forward), random_search=str(random))
        for protocol, prep in protocol_paths.items():
            if protocol == "U20":
                for selected_method, selection in [("forward", forward), ("random_search", random)]:
                    path = add(f"U20_{selected_method}_frozen", "fit", 4, ["prepare_U20", "P20_"+selected_method],
                               protocol_result=str(prep), method=selected_method,
                               selection_result=str(selection), feature_set=selected_method)
                    method_paths[protocol][selected_method] = str(path)
            for method in ["manual", "all", "forward", "random_search"]:
                selection = forward if method == "forward" else random
                dep = "P20_"+method if method in {"forward", "random_search"} else protocol+"_"+method
                for offset in cfg["additional_model_seed_offsets"]:
                    extra = {"selection_result": str(selection)} if method in {"forward", "random_search"} else {}
                    name = f"{method}_seed{read_anchor_seed+offset}"
                    path = add(f"{protocol}_{name}", "fit", 4, ["prepare_"+protocol, dep],
                        protocol_result=str(prep), method=name, feature_set=method, seed=read_anchor_seed+offset, **extra)
                    method_paths[protocol][name] = str(path)
    for name, reason in [("artist_family", cfg["optional_artist"]), ("secondary_3000_users", cfg["optional_3000_user_panel"]),
                         ("candidate_count_100", cfg["optional_100_alternatives"]), ("novel_enriched_sample", "not enabled; natural-prevalence novel breakdown retained")]:
        add(name, "blocked", 5, reason=reason)
        jobs[-1].update(status="blocked", reason=reason)
    for protocol, prep in protocol_paths.items():
        add("summary_"+protocol, "summary", 6, ["prepare_"+protocol],
            protocol_result=str(prep), protocol=protocol, methods=method_paths[protocol])
    return jobs, cache


def observe_tree():
    try:
        root = psutil.Process()
    except psutil.NoSuchProcess:
        # Some worker containers expose an outer /proc namespace. Mac uses the normal path.
        if sys.platform != "linux":
            raise
        root = psutil.Process(int(Path("/proc/self/stat").read_text().split()[0]))
    processes = [root] + root.children(recursive=True)
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rss, psutil.virtual_memory().available


def execute_job(job, run, cfg, deadline, stop_state, heartbeat=lambda: None):
    final = Path(job["output"])
    final.parent.mkdir(parents=True, exist_ok=True)
    attempt_num = len(job["attempts"])+1
    temp = final.with_name(final.name+f".attempt{attempt_num}.partial")
    temp.mkdir(parents=True, exist_ok=False)
    request_path = run / "requests" / (job["id"]+".json")
    atomic_json(request_path, job["request"])
    log = run / "logs" / f"{job['id']}.attempt{attempt_num}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    attempt = {"started": now(), "number": attempt_num, "log": str(log), "partial_output": str(temp)}
    job["attempts"].append(attempt)
    current = json.loads((run/"ledger.json").read_text())
    current[job["id"]] = job
    atomic_json(run/"ledger.json", current)
    peak, minimum, low_since = 0, psutil.virtual_memory().available, None
    started = time.monotonic()
    reason = None
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-m", "tlab_warmup.batch", "worker", str(request_path), str(temp)],
                                   stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        while process.poll() is None:
            try:
                rss, available = observe_tree()
            except (OSError, psutil.Error) as exc:
                reason = f"resource monitor unavailable: {exc}"
                rss, available = peak, minimum
            peak, minimum = max(peak, rss), min(minimum, available)
            if available < cfg["minimum_available_gib"]*1024**3:
                low_since = low_since or time.monotonic()
            else:
                low_since = None
            if reason:
                pass
            elif stop_state["reason"] or (run/"STOP").exists():
                reason = stop_state["reason"] or "STOP file requested"
            elif time.monotonic() >= deadline:
                reason = "global wall budget"
            elif time.monotonic()-started >= cfg["per_job_seconds"]:
                reason = "per-job wall budget"
            elif rss > cfg["stop_rss_gib"]*1024**3:
                reason = "process-tree RSS guard"
            elif low_since and time.monotonic()-low_since >= cfg["low_memory_seconds"]:
                reason = "sustained low available system memory"
            if reason:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if reason != "per-job wall budget":
                    stop_state["reason"] = reason
                break
            attempt["wall_seconds_so_far"] = time.monotonic()-started
            if int(attempt["wall_seconds_so_far"]*4) % 20 == 0:
                heartbeat()
                current = json.loads((run/"ledger.json").read_text())
                current[job["id"]] = job
                atomic_json(run/"ledger.json", current)
            time.sleep(.25)
    attempt.update(finished=now(), wall_seconds=time.monotonic()-started, returncode=process.returncode,
                   stop_reason=reason)
    job["resources"] = {"peak_tree_rss_bytes": peak, "minimum_system_available_bytes": minimum,
        "target_rss_exceeded": peak > cfg["target_rss_gib"]*1024**3,
        "poll_seconds": .25, "limitations": "sampled sum of runner/descendant RSS; shared pages may be double counted and spikes missed; system available memory is OS estimate"}
    if process.returncode == 0 and valid_output(temp):
        if final.exists():
            shutil.rmtree(final)
        temp.replace(final)
        job.update(status="completed", finished=now(), request_hash=digest(job["request"]))
        return
    failure = json.loads((temp/"failure.json").read_text()) if (temp/"failure.json").exists() else {"error": "worker exited without committed result"}
    job.update(status="failed" if reason == "per-job wall budget" else "interrupted" if reason else "blocked" if failure.get("blocked") else "failed",
               reason=reason or failure["error"], retryable=failure.get("retryable", False))


def export_bundle(run, ledger, plan):
    bundle = run.with_name(run.name+"-results.tar.gz")
    tmp = bundle.with_name(bundle.name+".partial")
    with tarfile.open(tmp, "w:gz") as tar:
        for p in sorted(run.rglob("*")):
            if p.is_file() and not p.name.endswith(".partial") and "requests" not in p.relative_to(run).parts:
                tar.add(p, arcname=run.name+"/"+str(p.relative_to(run)))
        for job in ledger.values():
            for attempt in job.get("attempts", []):
                for name in ["failure.json", "error.log", "trials.json", "calibration.json"]:
                    p = Path(attempt["partial_output"])/name
                    if p.exists():
                        tar.add(p, arcname=run.name+f"/failed_attempts/{job['id']}/{attempt['number']}/"+name)
            if job["request"]["kind"] in {"dataset", "protocol"} and job["status"] == "completed":
                directory = Path(job["output"])
                for name in ["result.json", "audit.json", "source_and_extraction.json", "targets.json.gz", "queries.json.gz", "feature_manifest.json"]:
                    p = directory/name
                    if p.exists():
                        tar.add(p, arcname=run.name+"/identities/"+job["id"]+"/"+name)
    tmp.replace(bundle)
    print(f"RESULTS_BUNDLE={bundle}", flush=True)
    return bundle


def validate_config(cfg):
    if not 0 < cfg["wall_hours"] <= 8 or not 1 <= cfg["threads"] <= 14:
        raise ValueError("Wall budget must be <=8 hours and model threads 1..14")
    if cfg["search_fit_cap"] > 24 or cfg["search_rounds"] > 3 or cfg["max_retries"] > 1:
        raise ValueError("Search/retry budget exceeds frozen plan")
    if cfg["target_rss_gib"] >= cfg["stop_rss_gib"] or cfg["stop_rss_gib"] > 10:
        raise ValueError("Invalid conservative memory guards")


def run_batch(phase, cfg, anchor, run, *, check=True, controls_only=False):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run/".runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner owns this output directory") from exc
        return _run_batch(phase, cfg, anchor, run, check=check, controls_only=controls_only)


def _run_batch(phase, cfg, anchor, run, *, check=True, controls_only=False):
    invocation_start = time.monotonic()
    run, anchor = Path(run).resolve(), Path(anchor).resolve()
    validate_config(cfg)
    run.mkdir(parents=True, exist_ok=True)
    state = code_state(ROOT)
    env = environment()
    import importlib.metadata
    env["packages"]["psutil"] = importlib.metadata.version("psutil")
    plan = {"schema": "tlab-batch-v1", "phase": phase, "config": cfg, "code": state,
            "environment": env, "anchor_sha256": sha256_local(anchor), "feature_manifest": manifest(),
            "test_scoring": "PROHIBITED", "controls_only": controls_only}
    plan_hash = digest(plan)
    plan_path = run/"plan_manifest.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["identity"] != plan_hash:
        raise ValueError("Resume refused: code, environment, anchor or config changed. Use a new --output path; old run preserved")
    atomic_json(plan_path, {"identity": plan_hash, **plan})
    prior_usage = json.loads((run/"usage.json").read_text())["total_active_seconds"] if (run/"usage.json").exists() else 0.0
    def heartbeat():
        atomic_json(run/"usage.json", {"total_active_seconds": prior_usage+time.monotonic()-invocation_start,
            "last_observed": now(), "limitation": "abrupt parent termination can lose up to one heartbeat interval; paused time is excluded"})
    heartbeat()
    if check and not (run/"CHECKS_PASSED.json").exists():
        checks = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                                capture_output=True, text=True, timeout=240)
        (run/"checks.log").write_text(checks.stdout+checks.stderr)
        if checks.returncode:
            atomic_json(run/"readiness.json", {"ready": False, "reason": "preflight failed", "test_scored": False})
            export_bundle(run, {}, plan)
            return False
        atomic_json(run/"CHECKS_PASSED.json", {"code": state["tree_sha256"], "passed_at": now()})
    jobs, cache = build_jobs(run, cfg, phase, anchor, state, controls_only=controls_only)
    old = json.loads((run/"ledger.json").read_text()) if (run/"ledger.json").exists() else {}
    ledger = {}
    for job in jobs:
        saved = old.get(job["id"])
        if saved:
            job["attempts"] = saved["attempts"]
            if saved["status"] == "completed" and valid_output(saved["output"]) and all(ledger.get(d, {}).get("status")=="completed" for d in job["depends_on"]):
                job = saved
            elif saved["status"] == "failed":
                job = saved
            elif saved["status"] == "blocked" and saved["request"]["kind"] == "blocked":
                job = saved
        ledger[job["id"]] = job
    # Summaries have soft method dependencies: failures are reported, never replaced by zeros.
    for job in ledger.values():
        if job["request"]["kind"] == "summary" and job["status"] == "completed":
            wanted = set(job["request"]["methods"].values())
            if any(j["output"] in wanted and j["status"] != "completed" for j in ledger.values()):
                job.update(status="pending")
    stop = {"reason": None}
    previous = {}
    def signal_stop(signum, frame):
        stop["reason"] = f"signal {signum}"
    for sig in [signal.SIGINT, signal.SIGTERM]:
        previous[sig] = signal.signal(sig, signal_stop)
    deadline = invocation_start+max(0, cfg["wall_hours"]*3600-prior_usage)
    atomic_json(run/"ledger.json", ledger)
    try:
        for job in sorted(ledger.values(), key=lambda j: j["priority"]):
            if job["status"] in {"completed", "blocked", "failed"}:
                continue
            if stop["reason"] or time.monotonic() >= deadline or (run/"STOP").exists():
                stop["reason"] = stop["reason"] or "wall budget or STOP file"
                break
            failed = [d for d in job["depends_on"] if ledger[d]["status"] != "completed"]
            if failed:
                job.update(status="blocked", reason="Dependencies not completed: "+", ".join(failed))
                atomic_json(run/"ledger.json", ledger)
                continue
            if job["priority"] in {3, 4} and "calibration_gate" in ledger:
                gate = Path(ledger["calibration_gate"]["output"])/"result.json"
                if gate.exists() and not json.loads(gate.read_text())["search_budget_allowed"]:
                    job.update(status="blocked", reason="calibration reserves time for fixed comparisons/reporting")
                    atomic_json(run/"ledger.json", ledger)
                    continue
            if job["request"]["kind"] in {"fit", "search"}:
                # Freeze the realized data/candidate/cache identities before each independent branch starts fitting.
                realized = {j["id"]: json.loads((Path(j["output"])/"result.json").read_text())
                            for j in ledger.values() if j["status"]=="completed" and j["request"]["kind"] in {"source", "dataset", "protocol", "gate"}}
                frozen = {"plan_identity": plan_hash, "realized": realized, "code": state,
                            "all_features_frozen": manifest(), "model_seeds": [read_anchor(anchor)["config"]["seed"]+i for i in [0,*cfg["additional_model_seed_offsets"]]], "frozen_at": now()}
                atomic_json(run/"freezes"/(job["id"]+".json"), frozen)
                atomic_json(run/"frozen_manifest.json", frozen)
            while True:
                job.update(status="running", started=now())
                atomic_json(run/"ledger.json", ledger)
                print(f"[{now()}] {job['id']}", flush=True)
                execute_job(job, run, cfg, deadline, stop, heartbeat)
                atomic_json(run/"ledger.json", ledger)
                if job["id"] == "calibration_gate" and job["status"] == "completed":
                    estimate = json.loads((Path(job["output"])/"result.json").read_text())
                    print("LOCAL_CALIBRATION="+json.dumps(estimate), flush=True)
                if job["status"] != "failed" or not job.get("retryable") or len(job["attempts"]) > cfg["max_retries"]:
                    break
        needed = ["U20_original", "U20_no_pair", "U20_recency_rule", "U20_popularity", "U20_random", "P20_original", "P20_no_pair", "P20_recency_rule", "P20_popularity", "P20_random", "summary_U20", "summary_P20"]
        readiness = {"ready": all(ledger.get(k, {}).get("status")=="completed" for k in needed),
                     "phase": phase, "code_hash": state["tree_sha256"], "plan_identity": plan_hash,
                     "test_scored": False, "stop_reason": stop["reason"],
                     "states": dict(__import__('collections').Counter(j["status"] for j in ledger.values())),
                     "blocked": {k:j.get("reason") for k,j in ledger.items() if j["status"]=="blocked"}}
        atomic_json(run/"readiness.json", readiness)
    except BaseException:
        (run/"runner_error.log").write_text(traceback.format_exc())
        for job in ledger.values():
            if job["status"] == "running":
                job.update(status="interrupted", reason="runner exception; inspect runner_error.log")
        atomic_json(run/"ledger.json", ledger)
        readiness = {"ready": False, "phase": phase, "test_scored": False,
                     "reason": "runner exception; see runner_error.log"}
        atomic_json(run/"readiness.json", readiness)
    finally:
        heartbeat()
        # Publish the authoritative final state before summaries/bundle export.
        atomic_json(run/"ledger.json", ledger)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_summary(run, ledger, stop)
        export_bundle(run, ledger, plan)
    return readiness["ready"] and not stop["reason"] and not any(j["status"]=="failed" for j in ledger.values())


def write_summary(run, ledger, stop):
    source = Path(ledger["source"]["output"])/"result.json"
    synthetic = json.loads(source.read_text())["source_config"]["source"]["synthetic"] if source.exists() else None
    lines = ["# Development batch", "", f"Data: {'SYNTHETIC FIXTURE ONLY' if synthetic else 'Yambda, real data' if synthetic is False else 'source not completed'}.",
             "Final test: not scored. All displayed scores are sampled-candidate development evidence.",
             "V_search and V_check reuse the development interval; V_check is not a pristine test.", "",
             "| Protocol | Method | NDCG@10 | Recall@10 | Repeat NDCG | Novel NDCG |", "|---|---|---:|---:|---:|---:|"]
    for job in ledger.values():
        if job["status"] == "completed" and job["request"]["kind"] in {"fit", "search"}:
            r = json.loads((Path(job["output"])/"result.json").read_text())
            m = r["metrics"]
            def number(value):
                return "missing" if value is None else f"{value:.6f}"
            lines.append(f"| {r['protocol']} | {r['method']} | {number(m['all']['ndcg10'])} | {number(m['all']['recall10'])} | {number(m['repeat']['ndcg10'])} | {number(m['novel']['ndcg10'])} |")
    lines += ["", "Statuses and limitations:", ""]
    for job in ledger.values():
        if job["status"] != "completed":
            lines.append(f"- {job['id']}: {job['status']}; {job.get('reason','not reached')}.")
    lines += ["", f"Stop reason: {stop['reason'] or 'declared queue finished'}.",
              "RSS is a sampled sum across runner and descendants, not total system/unified-memory pressure.",
              "The full dictionary is materialized once per protocol. Selected-feature application cost was not separately measured; do not claim deployment feature-cost savings.",
              "Timing and failed/interrupted attempts are in ledger.json. Paired user-cluster intervals and equal-user means are in summary job results."]
    (run/"batch_summary.md").write_text("\n".join(lines)+"\n")


def sha256_local(path):
    from .data import sha256
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("request")
    worker.add_argument("output")
    for phase in ["iteration1", "overnight"]:
        command = commands.add_parser(phase)
        command.add_argument("--config", type=Path, default=ROOT/"configs/experiments.json")
        command.add_argument("--anchor", type=Path, default=ROOT/"provenance/iteration0/results.tar.gz")
        command.add_argument("--data", type=Path)
        command.add_argument("--output", type=Path, default=ROOT/"runs"/phase)
        command.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        return run_worker(args.request, args.output)
    cfg = json.loads(args.config.read_text())
    if args.data:
        cfg["data"] = str(args.data.expanduser().resolve())
    if args.command == "overnight":
        readiness = ROOT/cfg["iteration1_output"]/"readiness.json"
        if not readiness.exists() or not json.loads(readiness.read_text()).get("ready"):
            parser.error("Run iteration1 successfully first; overnight needs its protocol/anchor readiness checkpoint")
        if json.loads(readiness.read_text())["code_hash"] != code_state(ROOT)["tree_sha256"]:
            parser.error("Code changed since iteration1; rerun it with a new output identity before overnight")
    return 0 if run_batch(args.command, cfg, args.anchor, args.output, controls_only=args.controls_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
