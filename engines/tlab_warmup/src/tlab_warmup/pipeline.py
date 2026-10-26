"""Single-command warmup runner. No automated search or test-set scoring."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import pyarrow.compute as pc
from .core import FEATURE_NAMES, PAIR_COLUMNS, build_features, ranked_metrics, sample_candidates
from .data import EVENT_TYPES, acquire, extract, sha256

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == "darwin" else 1024)


class Recorder:
    def __init__(self, directory):
        self.directory = directory
        self.start = time.perf_counter()
        self.stages = {}
        self.path = directory / "run.log"

    def log(self, message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        self.log(f"Starting {name}")
        try:
            yield
        finally:
            self.stages[name] = {"seconds": time.perf_counter() - start,
                                 "process_peak_rss_mib_so_far": peak_mib()}
            self.log(f"Finished {name}: {self.stages[name]['seconds']:.3f} seconds")


def environment():
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        try:
            ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2)) if sys.platform == "darwin" else None
        except (ValueError, OSError, subprocess.SubprocessError):
            ram = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": ram,
            "packages": {p: importlib.metadata.version(p) for p in [
                "numpy", "pyarrow", "scikit-learn", "scipy", "threadpoolctl"]},
            "memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; Linux KiB, macOS bytes",
            "memory_limitations": "Cumulative process high-water RSS, not per-stage increments; excludes child checks, OS file cache, download disk use, and other processes. Not unified-memory pressure.",
            "timing_limitations": "Stage wall time excludes initial uv setup and result review/handoff delay."}


def validate_config(c):
    if not 0 < c["train_fraction_of_global_time_span"] < c["development_end_fraction_of_global_time_span"] < 1:
        raise ValueError("Require 0 < train fraction < development end fraction < 1")
    for key in ["users", "training_queries", "development_queries", "alternatives", "threads",
                "window_seconds", "minimum_prior_user_events", "maximum_selected_rows", "acquisition_seconds"]:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if c["ratio_smoothing"] <= 0 or c["target_event_type"] not in EVENT_TYPES:
        raise ValueError("Invalid smoothing or target event type")
    if c["source"]["timestamp_seconds_per_unit"] != 1:
        raise ValueError("This protocol uses seconds as stored; no implicit timestamp conversion")


def prepare(table, info, c, *, return_eligible=False, select_queries=True):
    """Audit all sampled histories, then define chronological query eligibility."""
    for name in ["uid", "item_id", "timestamp", "event_type", "is_organic"]:
        if table[name].null_count:
            raise ValueError(f"Null in required field {name}")
    raw = np.column_stack([table[x].to_numpy().astype(np.int64) for x in ["uid", "item_id", "timestamp"]])
    types = np.asarray(pc.cast(table["event_type"], "string").to_pylist())
    if not set(types).issubset(EVENT_TYPES):
        raise ValueError(f"Unknown event types: {set(types) - EVENT_TYPES}")
    if np.any(raw < 0):
        raise ValueError("Negative IDs or timestamps")
    order = np.lexsort((raw[:, 1], raw[:, 0], raw[:, 2]))
    events, types = raw[order], types[order]
    if not len(events):
        raise ValueError("No sampled events")
    quant = c["source"]["timestamp_quantization_seconds"]
    if np.any(events[:, 2] % quant):
        raise ValueError("Timestamp values violate pinned quantization; inspect source before proceeding")
    lo, hi = info["source_min_timestamp"], info["source_max_timestamp"]
    if not c["source"]["synthetic"] and not 100 * 86400 < hi - lo < 400 * 86400:
        raise ValueError("Unexpected release time span in seconds; verify units before proceeding")
    train_end = c["train_end_timestamp"]
    dev_end = c["development_end_timestamp"]
    train_end = int(train_end if train_end is not None else lo + (hi - lo) * c["train_fraction_of_global_time_span"])
    dev_end = int(dev_end if dev_end is not None else lo + (hi - lo) * c["development_end_fraction_of_global_time_span"])
    if not lo < train_end < dev_end < hi:
        raise ValueError("Boundaries must lie strictly within the source time range")
    c["resolved_boundaries"] = {"source_start": lo, "train_end_exclusive": train_end,
                                "development_end_exclusive": dev_end, "reserved_test_end_inclusive": hi,
                                "derivation": "fixed fractions of global release timestamp span, or explicit configured timestamps; no metric-based selection"}
    prior = events[events[:, 2] < train_end]
    if not len(prior):
        raise ValueError("Empty training interval")
    ids, indices = np.unique(prior[:, 1], return_index=True)
    first = prior[indices, 2]
    idx = np.lexsort((ids, first))
    universe, first_seen = ids[idx], first[idx]
    first_map = dict(zip(map(int, universe), map(int, first_seen)))
    # Group once; avoid a full history scan for every user in larger panels.
    by_user = np.argsort(events[:, 0], kind="stable")
    user_ids, offsets, sizes = np.unique(events[by_user, 0], return_index=True, return_counts=True)
    user_times = events[by_user, 2]
    histories = {int(u): user_times[start:start + size] for u, start, size in zip(user_ids, offsets, sizes)}
    frozen_users = {u for u, times in histories.items() if
                    np.searchsorted(times, train_end, side="left") >= c["minimum_prior_user_events"]}
    pairs, group_counts = np.unique(events[:, [0, 2]], axis=0, return_counts=True)
    ambiguous = {tuple(map(int, row)) for row, n in zip(pairs, group_counts) if n > 1}
    type_codes = np.array([sorted(EVENT_TYPES).index(x) for x in types], dtype=np.int64)
    unique_full = len(np.unique(np.column_stack([events, type_codes]), axis=0))
    lengths = np.array(list(info["selected_counts"].values()))
    item_counts = np.unique(prior[:, 1], return_counts=True)[1]
    item_counts.sort()
    audit = {"selected_users": len(histories), "selected_rows": len(events),
             "event_types": dict(Counter(map(str, types))), "selected_items": len(np.unique(events[:, 1])),
             "training_item_universe": len(universe), "training_eligible_users": len(frozen_users),
             "history_lengths": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                 "p90": float(np.quantile(lengths, .9)), "max": int(lengths.max())},
             "selected_timestamp_min": int(events[:, 2].min()), "selected_timestamp_max": int(events[:, 2].max()),
             "timestamp_quantization_violation_rows": int(np.sum(events[:, 2] % quant != 0)),
             "source_selected_order_violations": int(np.sum((np.diff(raw[:, 0]) < 0) | (
                 (np.diff(raw[:, 0]) == 0) & (np.diff(raw[:, 2]) < 0)))),
             "ambiguous_user_timestamp_groups": len(ambiguous),
             "rows_in_ambiguous_user_timestamp_groups": int(sum(n for n in group_counts if n > 1)),
             "duplicate_uid_item_timestamp_event_type_rows": len(events) - unique_full,
             "duplicate_policy": "retain history records; exclude ambiguous target user/timestamp groups",
             "repeat_user_item_event_fraction": 1 - len(np.unique(events[:, :2], axis=0)) / len(events),
             "training_top_1pct_items_event_share": float(item_counts[-max(1, int(np.ceil(len(item_counts) * .01))):].sum() / len(prior)),
             "nulls": {name: table[name].null_count for name in table.column_names},
             "reserved_test_rows_not_scored": int(np.sum(events[:, 2] >= dev_end)),
             "history_feature_event_types": "all recorded event types; counts measure interaction, not preference"}
    eligible = {"train": [], "development": []}
    exclusions = {s: Counter() for s in eligible}
    known_dev = dev_targets = 0
    development_patterns = Counter()
    for row_id, ((u, item, t), kind) in enumerate(zip(events, types)):
        u, item, t = int(u), int(item), int(t)
        if t >= dev_end or kind != c["target_event_type"]:
            continue
        split = "train" if t < train_end else "development"
        ex = exclusions[split]
        ex["target_rows"] += 1
        if split == "development":
            dev_targets += 1
            known_dev += int(item in first_map)
            flags = []
            if (u, t) in ambiguous:
                flags.append("ambiguous_timestamp")
            if item not in first_map:
                flags.append("outside_training_catalog")
            if np.searchsorted(histories[u], t, side="left") < c["minimum_prior_user_events"]:
                flags.append("insufficient_prior_history")
            if u not in frozen_users:
                flags.append("training_ineligible_user")
            development_patterns[" & ".join(flags) if flags else "none_of_these_conditions"] += 1
        reason = None
        if (u, t) in ambiguous:
            reason = "ambiguous_user_timestamp"
        elif np.searchsorted(histories[u], t, side="left") < c["minimum_prior_user_events"]:
            reason = "insufficient_prior_user_history"
        elif split == "development" and u not in frozen_users:
            reason = "user_ineligible_at_training_boundary"
        elif item not in first_map:
            reason = "target_outside_training_universe"
        elif first_map[item] >= t:
            reason = "target_not_previously_observed"
        elif np.searchsorted(first_seen, t, side="left") < c["alternatives"] + 1:
            reason = "insufficient_candidate_pool"
        if reason:
            ex[reason] += 1
        else:
            eligible[split].append({"query_id": row_id, "uid": u, "timestamp": t, "positive": item, "split": split})
    queries = []
    rng = np.random.default_rng(c["seed"] + 1)
    for split, cap in [("train", c["training_queries"]), ("development", c["development_queries"])]:
        pool = eligible[split]
        exclusions[split]["eligible_before_query_cap"] = len(pool)
        selected = np.sort(rng.choice(len(pool), min(cap, len(pool)) if select_queries else 0, replace=False))
        for j in selected:
            q = pool[int(j)]
            q["candidates"] = sample_candidates(universe, first_seen, q["timestamp"], q["positive"],
                                                c["alternatives"], c["seed"], q["uid"])
            queries.append(q)
        exclusions[split]["selected_queries"] = len(selected)
        exclusions[split]["eligible_not_sampled"] = len(pool) - len(selected)
        if select_queries and not len(selected):
            raise ValueError(f"No eligible {split} queries; exclusion counts: {dict(exclusions[split])}; audit data and explicit limits")
    audit["exclusions"] = {s: dict(x) for s, x in exclusions.items()}
    audit["development_training_item_target_coverage"] = known_dev / dev_targets if dev_targets else None
    audit["coverage_denominator"] = f"all logged {c['target_event_type']} targets in development before eligibility filters and query cap"
    audit["development_joint_exclusion_patterns"] = dict(development_patterns)
    if return_eligible:
        return events, queries, audit, eligible
    return events, queries, audit


def export_candidates(path, queries):
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "uid", "timestamp", "split", "item_id", "label"])
        for q in queries:
            for item in q["candidates"]:
                writer.writerow([q["query_id"], q["uid"], q["timestamp"], q["split"], int(item), int(item == q["positive"])])


def fit_evaluate(queries, features, c, recorder):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import joblib
    train = [i for i, q in enumerate(queries) if q["split"] == "train"]
    dev = [i for i, q in enumerate(queries) if q["split"] == "development"]
    x = np.concatenate([features[i] for i in train])
    y = np.concatenate([(queries[i]["candidates"] == queries[i]["positive"]).astype(int) for i in train])
    variants = {"boosting": list(range(len(FEATURE_NAMES)))}
    if c["run_no_pair_ablation"]:
        variants["boosting_no_pair"] = [i for i in range(len(FEATURE_NAMES)) if i not in PAIR_COLUMNS]
    scores = {"popularity": [features[i][:, 2] for i in dev]}
    with threadpool_limits(limits=c["threads"]):
        for name, columns in variants.items():
            with recorder.stage(f"fit_{name}"):
                model = HistGradientBoostingClassifier(random_state=c["seed"], **c["model"])
                model.fit(x[:, columns], y)
            with recorder.stage(f"apply_{name}"):
                all_dev = np.concatenate([features[i][:, columns] for i in dev])
                flat_scores = model.predict_proba(all_dev)[:, 1]
                scores[name] = list(flat_scores.reshape(len(dev), c["alternatives"] + 1))
            joblib.dump(model, recorder.directory / f"{name}.joblib")
    metrics, per_query = {}, []
    for name, prediction_list in scores.items():
        rows = []
        for j, i in enumerate(dev):
            q = queries[i]
            result = ranked_metrics(q["candidates"], prediction_list[j], q["positive"],
                                    c["seed"], q["uid"], q["timestamp"])
            pos_idx = int(np.flatnonzero(q["candidates"] == q["positive"])[0])
            repeat = bool(features[i][pos_idx, 5] > 0)
            rows.append({**result, "uid": q["uid"], "query_id": q["query_id"], "repeat": repeat})
            per_query.append({"model": name, **rows[-1]})
        users = sorted({r["uid"] for r in rows})
        metrics[name] = {"queries": len(rows), "users": len(users),
                         "ndcg10_query_mean": float(np.mean([r["ndcg10"] for r in rows])),
                         "recall10_query_mean": float(np.mean([r["recall10"] for r in rows])),
                         "ndcg10_user_macro": float(np.mean([np.mean([r["ndcg10"] for r in rows if r["uid"] == u]) for u in users]))}
        metrics[name]["repeat_strata"] = {}
        for repeat in [False, True]:
            subset = [r for r in rows if r["repeat"] == repeat]
            metrics[name]["repeat_strata"]["repeat" if repeat else "novel"] = {
                "queries": len(subset), "ndcg10": float(np.mean([r["ndcg10"] for r in subset])) if subset else None}
    with gzip.open(recorder.directory / "development_predictions.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "uid", "timestamp", "item_id", "label", *scores, *FEATURE_NAMES])
        for j, i in enumerate(dev):
            q = queries[i]
            for k, item in enumerate(q["candidates"]):
                writer.writerow([q["query_id"], q["uid"], q["timestamp"], int(item), int(item == q["positive"]),
                                 *[float(scores[name][j][k]) for name in scores], *features[i][k].tolist()])
    write_json(recorder.directory / "per_query_metrics.json", per_query)
    return metrics


def run(config, data_path=None, output_root=None, skip_checks=False):
    output_root = Path(output_root or ROOT / "runs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("warmup-%Y%m%dT%H%M%S-%fZ")
    directory = output_root / name
    directory.mkdir()
    recorder = Recorder(directory)
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "data_kind": "synthetic fixture" if config["source"]["synthetic"] else "Yambda real data",
              "target_event_type": config["target_event_type"], "alternatives": config["alternatives"],
              "local_handoff_delay_seconds": None, "user_setup_seconds": None,
              "source_code_sha256": {p.name: sha256(p) for p in sorted((ROOT / "src/tlab_warmup").glob("*.py"))}}
    warnings = [
        "Event-conditioned ranking with 20 uniform sampled alternatives; not CTR, full-catalog ranking, exposure correction, or verified negative preference.",
        "Logged timestamps are treated as event-identity availability times. Ingestion/availability timestamps are absent; live serving and start-of-play validity are unverified. Playback outcomes are not predictors.",
        "Dataset user selection itself uses future activity; within-release user sampling cannot remove that source selection bias.",
        "Item popularity is measured only within sampled users. All recorded event types count as interactions, including negative feedback actions.",
        "Historical records update through development strictly before each query; model weights and item universe remain frozen at training boundary.",
        "Histories are left-censored by the released observation interval. Uniform alternatives are often easier than production retrieval candidates.",
        "Fixed modest classifier is a warmup baseline, not the final strong manual-feature baseline. No search implemented; no test-set scoring.",
        "Synthetic checks establish implementation behavior, not recommendation quality or Mac compatibility."]
    if config["source"]["synthetic"]:
        warnings.insert(0, "SYNTHETIC DATA: all quality/timing here are fixture diagnostics, not real-data findings.")
    try:
        write_json(directory / "resolved_config.json", config)
        with recorder.stage("preflight"):
            validate_config(config)
            import pyarrow as pa
            pa.set_cpu_count(config["threads"])
            pa.set_io_thread_count(config["threads"])
            write_json(directory / "environment.json", environment())
            if skip_checks:
                write_json(directory / "checks.json", {"skipped": "internal fixture integration run; parent test suite verifies checks"})
            else:
                checks = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                (directory / "checks.log").write_text(checks.stdout + checks.stderr)
                write_json(directory / "checks.json", {"command": "python -m unittest discover -s tests -v", "returncode": checks.returncode})
                if checks.returncode:
                    raise RuntimeError("Preflight correctness checks failed; see checks.log")
        cache = ROOT / "data" / config["source"]["revision"]
        cache.mkdir(parents=True, exist_ok=True)
        with recorder.stage("acquisition_and_source_fingerprint"):
            path = acquire(config, cache, data_path, recorder.log)
        record["source_file_bytes"] = path.stat().st_size
        record["source_file_path"] = str(path)
        with recorder.stage("complete_history_extraction"):
            table, info = extract(path, config, cache, recorder.log)
            write_json(directory / "source_and_extraction.json", info)
            write_json(directory / "selected_users.json", info["selected_users"])
        with recorder.stage("audit_and_queries"):
            events, queries, audit = prepare(table, info, config)
            write_json(directory / "audit.json", audit)
            write_json(directory / "resolved_config.json", config)
            export_candidates(directory / "candidates.csv.gz", queries)
        with recorder.stage("feature_building"):
            features = build_features(events, queries, config["window_seconds"], config["ratio_smoothing"])
        from .core import brute_features
        with recorder.stage("selected_history_reference_check"):
            checked = np.linspace(0, len(queries) - 1, min(6, len(queries)), dtype=int)
            for j in checked:
                np.testing.assert_allclose(features[j], brute_features(events, queries[j], config["window_seconds"], config["ratio_smoothing"]), rtol=1e-12, atol=1e-12)
            write_json(directory / "feature_reference_check.json", {"passed": True, "query_ids": [queries[j]["query_id"] for j in checked], "independent_method": "direct brute-force historical masks"})
        with recorder.stage("model_comparison_total"):
            metrics = fit_evaluate(queries, features, config, recorder)
        write_json(directory / "metrics.json", metrics)
        record.update(status="success", feature_count=len(FEATURE_NAMES), feature_names=FEATURE_NAMES,
                      training_candidate_rows=sum(q["split"] == "train" for q in queries) * (config["alternatives"] + 1),
                      training_queries=sum(q["split"] == "train" for q in queries),
                      development_queries=sum(q["split"] == "development" for q in queries), metrics=metrics)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        (directory / "error.log").write_text(traceback.format_exc())
        recorder.log(f"FAILED: {exc}")
    finally:
        record.update(stages=recorder.stages, local_pipeline_seconds=time.perf_counter() - recorder.start,
                      process_peak_rss_mib=peak_mib(), finished_utc=datetime.now(timezone.utc).isoformat(),
                      warnings=warnings, next_iteration_decision="Pending review of the returned real-data bundle; no automatic method search authorized.")
        write_json(directory / "run_record.json", record)
        (directory / "warmup_report.md").write_text(report(record))
        files = {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}
        write_json(directory / "artifact_checksums.json", files)
        bundle = output_root / f"{name}-results.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(directory, arcname=name)
        print(f"RESULTS_BUNDLE={bundle}", flush=True)
    return record, bundle


def report(r):
    lines = ["# Iteration 0 run", "", f"Status: **{r['status']}**. Data: **{r['data_kind']}**.", "",
             f"Task: rank the recorded {r['target_event_type']} item among {r['alternatives'] + 1} candidates using strictly earlier recorded interactions.",
             "Training precedes development globally; the later test interval is reserved and not scored.", "",
             f"Pipeline wall time (excluding uv setup and handoff): {r['local_pipeline_seconds']:.3f} s.",
             f"Process peak RSS: {r['process_peak_rss_mib']:.1f} MiB; see environment.json for limitations.", ""]
    if r["status"] == "success":
        lines += [f"Training queries: {r['training_queries']}; development queries: {r['development_queries']}.", "",
                  "| Method | NDCG@10 | Recall@10 | Users |", "|---|---:|---:|---:|"]
        for name, m in r["metrics"].items():
            lines.append(f"| {name} | {m['ndcg10_query_mean']:.6f} | {m['recall10_query_mean']:.6f} | {m['users']} |")
        lines += ["", "Checks: preflight suite status in checks.json; sampled-history brute-force feature check passed.",
                  "Exclusions, target coverage, timestamp ties, repeat rates, and history completeness are in audit.json and source_and_extraction.json."]
    else:
        lines += [f"Blocker: {r.get('error')}. See error.log and run.log. Retained partial downloads can be resumed with the same command."]
    lines += ["", "Next decision: " + r["next_iteration_decision"], "", "Limitations:", ""]
    lines += ["- " + w for w in r["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/warmup.json")
    parser.add_argument("--data", type=Path, help="Existing exact pinned multi_event.parquet (SHA256 verified)")
    parser.add_argument("--output", type=Path, help="Results parent directory (default runs/)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    _, bundle = run(config, args.data, args.output)
    record = json.loads((bundle.parent / bundle.name.removesuffix("-results.tar.gz") / "run_record.json").read_text())
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
