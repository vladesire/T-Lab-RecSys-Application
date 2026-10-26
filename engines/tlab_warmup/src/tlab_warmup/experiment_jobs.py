"""Concrete experiment jobs; called by the small sequential batch runner."""
from collections import Counter
import copy
import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np

from .core import FEATURE_NAMES, brute_features
from .data import acquire, extract, sha256
from .evaluation import per_query, aggregate, paired_interval
from .experiment_io import atomic_json, save_queries, load_queries, digest
from .pipeline import prepare, ROOT
from .representations import manifest, build_dictionary
from .sampling import candidate_queries


def read_anchor(path):
    with tarfile.open(path) as tar:
        names = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        def raw(name):
            return tar.extractfile(names[name]).read()
        import hashlib
        for name, h in json.loads(raw("artifact_checksums.json")).items():
            if hashlib.sha256(raw(name)).hexdigest() != h:
                raise ValueError("Anchor artifact checksum mismatch")
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw("candidates.csv.gz")).decode())))
        groups = {}
        for row in rows:
            key = (row["split"], int(row["query_id"]))
            groups.setdefault(key, []).append(row)
        queries = []
        for (split, qid), rows in sorted(groups.items(), key=lambda p: (p[0][0] != "train", p[0][1])):
            positives = [r for r in rows if int(r["label"]) == 1]
            if len(positives) != 1 or len(rows) != 21:
                raise ValueError("Invalid original anchor candidates")
            r = positives[0]
            queries.append({"query_id": qid, "uid": int(r["uid"]), "timestamp": int(r["timestamp"]),
                            "positive": int(r["item_id"]), "split": split,
                            "candidates": [int(r["item_id"]) for r in rows], "anchor": split == "development"})
        return {"queries": queries, "config": json.loads(raw("resolved_config.json")),
                "users": json.loads(raw("selected_users.json")), "source": json.loads(raw("source_and_extraction.json")),
                "predictions": list(csv.DictReader(io.StringIO(gzip.decompress(raw("development_predictions.csv.gz")).decode()))),
                "sha256": sha256(path)}


def target_key(q):
    return q["uid"], q["timestamp"], q["positive"]


def source_job(request, out):
    cfg = request["config"]
    anchor = read_anchor(request["anchor"])
    base = anchor["config"]
    if not base["source"]["synthetic"] and (base["resolved_boundaries"]["train_end_exclusive"] != 20800000 or
                                            base["resolved_boundaries"]["development_end_exclusive"] != 23400000):
        raise ValueError("Unexpected original chronology")
    cache = ROOT / "data" / base["source"]["revision"]
    cache.mkdir(parents=True, exist_ok=True)
    path = acquire(base, cache, cfg.get("data"), print)
    atomic_json(out / "result.json", {"source_path": str(path), "source_sha256": base["source"]["sha256"],
                "anchor_sha256": anchor["sha256"], "source_config": base, "users": anchor["users"]})


def dataset_job(request, out):
    cfg = request["config"]
    source = json.loads((Path(request["source_result"]) / "result.json").read_text())
    anchor = read_anchor(request["anchor"])
    c = copy.deepcopy(source["source_config"])
    limits = cfg["panels"][request["panel"]]
    c.update(users=limits["users"], training_queries=limits["train"], development_queries=limits["development"],
             maximum_selected_rows=cfg["maximum_selected_rows"], threads=cfg["threads"],
             train_end_timestamp=c["resolved_boundaries"]["train_end_exclusive"],
             development_end_timestamp=c["resolved_boundaries"]["development_end_exclusive"])
    if limits["users"] > len(anchor["users"]):
        c["retain_user_ids"] = anchor["users"]
    elif limits["users"] != len(anchor["users"]):
        raise ValueError("Diagnostic/calibration panel must preserve the entire original cohort")
    cache = ROOT / "data" / c["source"]["revision"]
    table, info = extract(source["source_path"], c, cache, print)
    if not set(anchor["users"]).issubset(info["selected_users"]):
        raise ValueError("Nested cohort preservation failed")
    events, _, audit, pools = prepare(table, info, c, return_eligible=True, select_queries=False)
    anchor_dev = {target_key(q): q for q in anchor["queries"] if q["split"] == "development"}
    rng = np.random.default_rng(c["seed"] + 4401)
    selected = []
    for split, cap in [("train", limits["train"]), ("development", limits["development"])]:
        pool = pools[split]
        if request["panel"] == "iteration1":
            old = [q for q in anchor["queries"] if q["split"] == split]
            by_key = {target_key(q): q for q in pool}
            if any(target_key(q) not in by_key for q in old):
                raise ValueError("Original query no longer eligible; refuse silent anchor replacement")
            if split == "train":
                if cap != len(old):
                    raise ValueError("Iteration 1 training cap must equal original anchor")
                chosen = [dict(q) for q in old]
            else:
                if cap < len(old):
                    raise ValueError("Expanded development cap cannot remove anchor queries")
                remaining = [q for q in pool if target_key(q) not in anchor_dev]
                indices = np.sort(rng.choice(len(remaining), min(cap - len(old), len(remaining)), replace=False))
                chosen = [dict(q) for q in old] + [dict(remaining[int(j)]) for j in indices]
        else:
            indices = np.sort(rng.choice(len(pool), min(cap, len(pool)), replace=False))
            chosen = [dict(pool[int(j)]) for j in indices]
        for q in chosen:
            q.pop("candidates", None)
            q["anchor"] = split == "development" and target_key(q) in anchor_dev
        chosen.sort(key=lambda q: q["query_id"])
        selected.extend(chosen)
        audit["exclusions"][split].update(selected_queries=len(chosen), eligible_not_sampled=len(pool)-len(chosen))
    if not all(any(q["split"] == s for q in selected) for s in ["train", "development"]):
        raise ValueError("Empty fitting or development population")
    # Descriptive audit of a deterministic bounded sample of tied groups.
    sample = table.slice(0, min(20000, len(table))).select(["uid", "timestamp", "item_id", "event_type"]).to_pydict()
    ties = {}
    for u, t, it, kind in zip(sample["uid"], sample["timestamp"], sample["item_id"], sample["event_type"]):
        key = int(u), int(t)
        if key not in ties and len(ties) >= 5000:
            continue
        ties.setdefault(key, []).append((int(it), str(kind)))
    tied = [rows for rows in ties.values() if len(rows) > 1][:100]
    audit["tied_group_descriptive_sample"] = {"selection": "first up to 100 tied groups among first 5000 groups in first 20000 source rows; prefix boundary can truncate the last group; not representative",
        "groups": len(tied), "multiplicities": dict(Counter(str(len(r)) for r in tied)),
        "event_types": dict(Counter(kind for rows in tied for _, kind in rows)),
        "groups_with_repeated_item_ids": sum(len({it for it, _ in rows}) < len(rows) for rows in tied)}
    midpoint = (c["train_end_timestamp"] + c["development_end_timestamp"]) // 2
    np.save(out / "events.npy", events, allow_pickle=False)
    save_queries(out / "targets.json.gz", selected)
    atomic_json(out / "audit.json", audit)
    atomic_json(out / "source_and_extraction.json", info)
    atomic_json(out / "result.json", {"config": c, "midpoint": midpoint, "selected_rows": len(events),
        "train_queries": sum(q["split"] == "train" for q in selected),
        "development_queries": sum(q["split"] == "development" for q in selected),
        "selected_user_ids": info["selected_users"], "target_hash": digest(selected),
        "panel": request["panel"], "test_scored": False})


def protocol_job(request, out):
    data = Path(request["dataset_result"])
    meta = json.loads((data / "result.json").read_text())
    events = np.load(data / "events.npy", mmap_mode="r")
    targets = load_queries(data / "targets.json.gz")
    c = meta["config"]
    started = time.perf_counter()
    queries = candidate_queries(events, targets, c["train_end_timestamp"], c["seed"], request["protocol"])
    sampling_seconds = time.perf_counter() - started
    anchor = read_anchor(request["anchor"])
    if meta["panel"] == "iteration1" and request["protocol"] == "U20":
        original = {target_key(q): q for q in anchor["queries"]}
        for q in queries:
            if target_key(q) in original and q["candidates"] != original[target_key(q)]["candidates"]:
                raise ValueError("Original U20 candidates changed; preserve anchor and block reproduction")
    save_queries(out / "queries.json.gz", queries)
    started = time.perf_counter()
    spec = build_dictionary(events[events[:, 2] < c["development_end_timestamp"]], queries,
                            out / "features.npy", day_seconds=c["window_seconds"] / 7)
    feature_seconds = time.perf_counter() - started
    matrix = np.load(out / "features.npy", mmap_mode="r")
    columns = [spec["names"].index(name) for name in spec["original"]]
    # Independent direct masks check a bounded selection of original features.
    for j in np.linspace(0, len(queries)-1, min(5, len(queries)), dtype=int):
        q = queries[int(j)]
        actual = matrix[j*21:(j+1)*21, columns]
        np.testing.assert_allclose(actual, brute_features(events, q, c["window_seconds"]), rtol=1e-12, atol=1e-12)
    if meta["panel"] == "iteration1" and request["protocol"] == "U20":
        lookup = {target_key(q): j for j, q in enumerate(queries)}
        originals = {q["query_id"]: q for q in anchor["queries"] if q["split"] == "development"}
        for row in anchor["predictions"]:
            q = originals[int(row["query_id"])]
            j = lookup[target_key(q)]
            candidate = queries[j]["candidates"].index(int(row["item_id"]))
            np.testing.assert_allclose(matrix[j*21+candidate, columns], [float(row[n]) for n in FEATURE_NAMES], rtol=1e-12, atol=1e-12)
    dev = [j for j, q in enumerate(queries) if q["split"] == "development"]
    candidate_matrix = np.array([q["candidates"] for q in queries])
    positive = np.array([q["positive"] for q in queries])
    pair = matrix[:, spec["names"].index("user_item_count")].reshape(-1, 21)
    repeated = pair > 0
    labels = candidate_matrix == positive[:, None]
    familiarity = {"positive_repeat_fraction": float(repeated[dev][labels[dev]].mean()),
        "alternative_repeat_fraction": float(repeated[dev][~labels[dev]].mean()),
        "distinct_development_candidate_items": len(np.unique(candidate_matrix[dev])),
        "unique_alternative_fraction": len(np.unique(candidate_matrix[dev][~labels[dev]])) / (len(dev)*20),
        "queries": len(dev)}
    _, alternative_counts = np.unique(candidate_matrix[dev][~labels[dev]], return_counts=True)
    probabilities = alternative_counts / alternative_counts.sum()
    familiarity["effective_alternative_item_count_entropy"] = float(np.exp(-np.sum(probabilities*np.log(probabilities))))
    familiarity["alternative_top1pct_item_share"] = float(np.sort(alternative_counts)[-max(1,int(np.ceil(len(alternative_counts)*.01))):].sum()/alternative_counts.sum())
    atomic_json(out / "feature_manifest.json", spec)
    atomic_json(out / "result.json", {**meta, "protocol": request["protocol"],
        "query_candidate_hash": digest(queries), "feature_manifest_hash": digest(spec),
        "sampler": "original U20 exact RNG" if request["protocol"] == "U20" else "exact sequential PPS without replacement, c_i(t)+1, exponent 1; strict past, no snapshots",
        "sampling_seconds": sampling_seconds, "feature_seconds": feature_seconds,
        "feature_matrix_bytes": (out / "features.npy").stat().st_size,
        "independent_feature_checks": min(5, len(queries)), "familiarity": familiarity})


def data_for_model(request):
    directory = Path(request["protocol_result"])
    meta = json.loads((directory / "result.json").read_text())
    queries = load_queries(directory / "queries.json.gz")
    features = np.load(directory / "features.npy", mmap_mode="r")
    ntrain = sum(q["split"] == "train" for q in queries)
    dev = queries[ntrain:]
    if not all(q["split"] == "development" for q in dev):
        raise ValueError("Queries must have train then development ordering")
    if any(q["timestamp"] >= meta["config"]["development_end_timestamp"] for q in dev):
        raise ValueError("Final-test scoring is prohibited")
    spec = manifest()
    pair_col = spec["names"].index("user_item_count")
    devpair = features[ntrain*21:, pair_col].reshape(-1, 21)
    items = np.array([q["candidates"] for q in dev])
    positive = np.array([q["positive"] for q in dev])
    repeat = devpair[items == positive[:, None]] > 0
    return meta, queries, features, ntrain, dev, repeat


def train_model(features, queries, ntrain, names, c, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    cols = [manifest()["names"].index(name) for name in names]
    y = np.array([int(it == q["positive"]) for q in queries[:ntrain] for it in q["candidates"]])
    started = time.perf_counter()
    with threadpool_limits(limits=c["threads"]):
        model = HistGradientBoostingClassifier(random_state=seed, **c["model"])
        model.fit(features[:ntrain*21, cols], y)
    return model, time.perf_counter() - started


def model_scores(model, features, ntrain, names, ndev, threads, indices=None):
    from threadpoolctl import threadpool_limits
    cols = [manifest()["names"].index(name) for name in names]
    selected = np.arange(ndev) if indices is None else np.asarray(indices)
    predictions = np.empty((len(selected), 21))
    with threadpool_limits(limits=threads):
        for start in range(0, len(selected), 512):
            qs = selected[start:start+512]
            rows = ((qs[:, None] + ntrain)*21 + np.arange(21)).reshape(-1)
            predictions[start:start+len(qs)] = model.predict_proba(features[np.ix_(rows, cols)])[:, 1].reshape(-1, 21)
    return predictions


def finish_scores(request, out, meta, dev, repeat, scores, extra):
    values = per_query(dev, scores, meta["config"]["seed"])
    metrics = aggregate(dev, values, repeat, meta["midpoint"])
    if request["method"] == "random":
        expected = 10/21
        tolerance = max(.05, 4*np.sqrt(expected*(1-expected)/len(dev)))
        extra["random_sanity"] = {"expected_recall10": expected,
            "expected_ndcg10": float(sum(1/np.log2(r+1) for r in range(1,11))/21),
            "recall_deviation_warning": bool(abs(metrics["all"]["recall10"]-expected)>tolerance),
            "tolerance": float(tolerance), "interpretation": "finite-query random-score check; not a quality gate"}
    np.savez_compressed(out / "predictions.npz", scores=scores, per_query_metrics=values, repeat=repeat,
                        query_ids=np.array([q["query_id"] for q in dev]))
    atomic_json(out / "result.json", {"method": request["method"], "seed": request["seed"],
        "protocol": meta["protocol"], "query_candidate_hash": meta["query_candidate_hash"],
        "metrics": metrics, **extra})


def fit_job(request, out):
    meta, queries, features, ntrain, dev, repeat = data_for_model(request)
    c = meta["config"]
    c["threads"] = request["config"]["threads"]
    spec = manifest()
    method = request["method"]
    started = time.perf_counter()
    if method in {"popularity", "recency_rule", "random", "personal_frequency", "decay_popularity"}:
        x = features[ntrain*21:]
        col = {name: j for j, name in enumerate(spec["names"])}
        pop = x[:, col["item_count"]].reshape(-1, 21)
        pair = x[:, col["user_item_count"]].reshape(-1, 21)
        if method == "popularity":
            scores = np.array(pop)
        elif method == "personal_frequency":
            scores = np.array(pair)
        elif method == "decay_popularity":
            scores = np.array(x[:, col["item_decay_7d"]].reshape(-1, 21))
        elif method == "random":
            scores = np.array([np.random.default_rng(np.random.SeedSequence([request["seed"], q["uid"], q["timestamp"], 81731])).random(21) for q in dev])
        else:
            recency = x[:, col["user_item_recency_seconds"]].reshape(-1, 21)
            scores = np.array(pop)
            for j in range(len(dev)):
                seen = pair[j] > 0
                distinct = np.unique(recency[j, seen])
                scores[j, seen] = pop[j].max() + 1 + len(distinct) - np.searchsorted(distinct, recency[j, seen])
        extra = {"fit_seconds": 0.0, "application_seconds": time.perf_counter()-started}
    else:
        feature_set = request.get("feature_set", method)
        if "selection_result" in request:
            names = json.loads((Path(request["selection_result"]) / "result.json").read_text())["selected_features"]
        else:
            names = spec[feature_set]
        import joblib
        reused = None
        # Reuse identical committed finalist fits within this immutable run only.
        # Search trials remain independently timed for the matched-cost control.
        for result_path in sorted(out.parent.glob("*/result.json")):
            if result_path.parent.name.endswith(".partial") or not (result_path.parent/"COMPLETE.json").exists():
                continue
            previous = json.loads(result_path.read_text())
            if (previous.get("seed") == request["seed"] and
                previous.get("query_candidate_hash") == meta["query_candidate_hash"] and
                previous.get("feature_names", previous.get("selected_features")) == names and
                (result_path.parent/"model.joblib").exists()):
                reused = result_path.parent
                break
        if reused is None:
            model, fit_seconds = train_model(features, queries, ntrain, names, c, request["seed"])
        else:
            model, fit_seconds = joblib.load(reused/"model.joblib"), 0.0
        started = time.perf_counter()
        scores = model_scores(model, features, ntrain, names, len(dev), c["threads"])
        application_seconds = time.perf_counter()-started
        serialization_start = time.perf_counter()
        joblib.dump(model, out / "model.joblib")
        extra = {"fit_seconds": fit_seconds, "application_seconds": application_seconds,
                 "model_serialization_seconds": time.perf_counter()-serialization_start,
                 "reused_identical_model_from": str(reused) if reused else None,
                 "feature_names": names, "feature_count": len(names)}
    if request.get("verify_anchor"):
        anchor = read_anchor(request["anchor"])
        expected = {(int(r["query_id"]), int(r["item_id"])): float(r["boosting"]) for r in anchor["predictions"]}
        discrepancies = [abs(scores[j, k] - expected[q["query_id"], it]) for j, q in enumerate(dev) if q.get("anchor")
                         for k, it in enumerate(q["candidates"])]
        if not discrepancies or max(discrepancies) > 1e-8:
            raise ValueError("U20 original-model anchor prediction reproduction failed")
        extra["anchor_prediction_max_absolute_difference"] = max(discrepancies)
    finish_scores(request, out, meta, dev, repeat, scores, extra)


def search_job(request, out):
    meta, queries, features, ntrain, dev, repeat = data_for_model(request)
    settings = request["config"]
    c = meta["config"]
    c["threads"] = settings["threads"]
    search_idx = np.flatnonzero(np.array([q["timestamp"] for q in dev]) < meta["midpoint"])
    check_idx = np.flatnonzero(np.array([q["timestamp"] for q in dev]) >= meta["midpoint"])
    for label, idx in [("V_search", search_idx), ("V_check", check_idx)]:
        if len(idx) < settings["search_min_queries"] or len({dev[i]["uid"] for i in idx}) < settings["search_min_users"]:
            raise Blocked(f"Insufficient {label} queries/users for declared search gate")
        if sum(~repeat[idx]) < settings["search_min_novel"]:
            raise Blocked(f"Insufficient {label} novel targets for declared search gate")
    spec = manifest()
    base = spec["manual"]
    group_names = sorted(spec["groups"])
    baseline = json.loads((Path(request["manual_result"]) / "result.json").read_text())["metrics"]["V_search"]["ndcg10"]
    best, chosen, selected_names = baseline, [], list(base)
    trials = []
    def evaluate(additions):
        names = sorted(set(base).union(*(spec["groups"][g] for g in additions)), key=spec["names"].index)
        trial = {"number": len(trials)+1, "groups": list(additions), "feature_names": names, "status": "running"}
        trials.append(trial)
        atomic_json(out / "trials.json", trials)
        model, fit_seconds = train_model(features, queries, ntrain, names, c, request["seed"])
        application_start = time.perf_counter()
        scores = model_scores(model, features, ntrain, names, len(dev), c["threads"], search_idx)
        value = float(per_query([dev[i] for i in search_idx], scores, c["seed"])[:, 0].mean())
        trial.update(status="completed", V_search_ndcg10=value, fit_seconds=fit_seconds,
                     application_and_scoring_seconds=time.perf_counter()-application_start)
        atomic_json(out / "trials.json", trials)
        return value, names, model
    selected_model = None
    if request["method"] == "forward":
        for _ in range(settings["search_rounds"]):
            remaining = [g for g in group_names if g not in chosen]
            if not remaining or len(trials)+len(remaining) > settings["search_fit_cap"]:
                break
            candidates = []
            for g in remaining:
                score, names, model = evaluate(chosen + [g])
                candidates.append((score, g, names, model))
            winner = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            if winner[0] <= best + settings["search_min_gain"]:
                break
            best, g, selected_names, selected_model = winner
            chosen.append(g)
    else:
        forward = json.loads((Path(request["forward_result"]) / "result.json").read_text())
        schedule = forward["trial_addition_counts"]
        rng = np.random.default_rng(request["seed"] + 901)
        pools = {k: list(itertools.combinations(group_names, k)) for k in set(schedule)}
        for k in pools:
            rng.shuffle(pools[k])
        for k in schedule:
            if not pools[k]:
                raise ValueError("Insufficient unique matched random subsets")
            additions = list(pools[k].pop())
            value, names, model = evaluate(additions)
            if value > best + settings["search_min_gain"]:
                best, chosen, selected_names, selected_model = value, additions, names, model
    # Selected baseline needs no duplicate fit: load the already fitted manual control.
    import joblib
    if selected_model is None:
        selected_model = joblib.load(Path(request["manual_result"]) / "model.joblib")
    scores = model_scores(selected_model, features, ntrain, selected_names, len(dev), c["threads"])
    joblib.dump(selected_model, out / "model.joblib")
    finish_scores(request, out, meta, dev, repeat, scores, {"selected_features": selected_names,
        "selected_groups": chosen, "candidate_fits": len(trials), "V_search_best": best,
        "trial_addition_counts": [len(t["groups"]) for t in trials],
        "candidate_fit_seconds": sum(t.get("fit_seconds", 0) for t in trials),
        "selection_boundary": meta["midpoint"], "V_check_used_for_selection": False})


class Blocked(RuntimeError):
    pass


def summary_job(request, out):
    protocol_path = Path(request["protocol_result"])
    meta = json.loads((protocol_path / "result.json").read_text())
    queries = [q for q in load_queries(protocol_path / "queries.json.gz") if q["split"] == "development"]
    rows, values = {}, {}
    for name, directory in request["methods"].items():
        path = Path(directory)
        if not (path / "COMPLETE.json").exists():
            continue
        rows[name] = json.loads((path / "result.json").read_text())
        with np.load(path / "predictions.npz") as predictions:
            values[name] = predictions["per_query_metrics"].copy()
            repeat = predictions["repeat"].copy()
    comparisons = [("original", "recency_rule"), ("original", "popularity"), ("no_pair", "original"),
                   ("manual", "recency_rule"), ("decay", "manual"), ("all", "manual"),
                   ("forward", "manual"), ("random_search", "forward")]
    intervals = {}
    for left, right in comparisons:
        if left not in values or right not in values:
            continue
        for label, mask in [("all", np.ones(len(queries), dtype=bool)), ("repeat", repeat), ("novel", ~repeat)]:
            intervals[f"{left}-minus-{right}/{label}"] = paired_interval(queries, values[left], values[right], mask,
                request["seed"], request["config"]["bootstrap_replicates"])
    atomic_json(out / "result.json", {"protocol": request["protocol"], "methods": rows, "paired_intervals": intervals,
                "test_scored": False, "conditional_development_only": True})


DISPATCH = {"source": source_job, "dataset": dataset_job, "protocol": protocol_job,
            "fit": fit_job, "search": search_job, "summary": summary_job}
