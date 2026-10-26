#!/usr/bin/env python3
"""Pin actual exported IDs for the existing analyzer; no metrics are implemented."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path, PurePosixPath
import shlex
import sys

from prepare_completion_study import bridge_for, identifier, load, validate


FIXED = ("random", "popularity", "recency_rule", "personal_frequency", "decay_popularity",
         "original", "no_pair", "manual", "decay", "all")
PAIRS = (("manual", "all"), ("decay", "manual"), ("no_pair", "original"))


def member(bundle, manifest, relative, sha):
    """Check a selected inventoried file. Full bundle acceptance belongs to researchflow."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or "\\" in relative:
        raise ValueError("Unsafe manifest member: " + relative)
    selected = [f for f in manifest["files"] if f["path"] == relative]
    local = bundle / relative
    if (len(selected) != 1 or not local.is_file() or local.is_symlink()
            or not local.resolve().is_relative_to(bundle.resolve())):
        raise ValueError("Selected file is missing, aliased or not uniquely inventoried: " + relative)
    entry = selected[0]
    if local.stat().st_size != entry["size_bytes"] or sha(local) != entry["sha256"]:
        raise ValueError("Selected file fingerprint mismatch: " + relative)
    return local


def configure(root, run, analysis_id):
    root, run = root.expanduser().resolve(), run.expanduser().resolve()
    identifier(analysis_id)
    if not analysis_id.startswith("A-finish-"):
        raise ValueError("Use a fresh A-finish-* analysis ID")
    output = root / "configs/completion/science" / analysis_id
    if (output.exists() or output.is_symlink()
            or any((root / "analyses").glob(analysis_id + "-*"))
            or any((root / "analyses/.staging").glob(analysis_id + "*"))):
        raise ValueError("Analysis identity already used; choose a new --analysis-id")
    b = bridge_for(root)
    features = load(root / "reproducibility/feature-manifest.json")
    seed = load(root / "reproducibility/anchor-config.json")["seed"]
    manifest_path = run / "manifest.json"
    manifest = load(manifest_path)
    validate(root, "run_manifest", manifest)
    if manifest["status"] != "succeeded" or manifest["metadata"]["phase"] != "development":
        raise ValueError("Requires a successful development run; preserve failed staging separately")
    result_path = member(run, manifest, manifest["metadata"]["result_path"], b.sha)
    result = load(result_path)
    validate(root, "runner_result", result)
    if result["kind"] != "ranking" or result["checks"]["passed"] is not True:
        raise ValueError("Runner ranking checks have not passed")
    selected, main_ids, unavailable, excluded = [], {}, [], []
    for protocol in ("U20", "P20"):
        descriptors = [p for p in result["predictions"]
                       if p["protocol_id"] == protocol and p["split"] == "development"]
        originals = [p for p in descriptors if p.get("tags", {}).get("upstream_method") == "original"]
        if len(originals) != 1:
            raise ValueError(f"{protocol}: cannot identify a unique original main-panel export")
        identity = originals[0]["identity"]
        family = [p for p in descriptors if p["identity"] == identity]
        excluded.extend(p["id"] for p in descriptors if p not in family)
        for p in family:
            member(run, manifest, str(PurePosixPath(manifest["metadata"]["result_path"]).parent / p["path"]), b.sha)
        selected.extend(family)  # retain every main-panel method, including secondary searches
        for method in FIXED:
            matches = [p for p in family if p.get("tags", {}).get("upstream_method") == method]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous {protocol}/{method}; refuse to choose a favorable export")
            if matches:
                prediction = matches[0]
                if prediction.get("tags", {}).get("model_seed") != seed:
                    raise ValueError(f"{protocol}/{method}: base seed differs from the frozen anchor")
                if method in ("original", "no_pair", "manual", "decay", "all"):
                    names = features[method]
                    if (prediction.get("tags", {}).get("feature_names") != names
                            or prediction["feature_count"] != len(names)):
                        raise ValueError(f"{protocol}/{method}: exported features/count differ from the frozen representation")
                main_ids[protocol, method] = matches[0]["id"]
            else:
                unavailable.append(f"{protocol}/{method}: no matching main-panel export")
    if len({p["id"] for p in selected}) != len(selected):
        raise ValueError("Duplicate exported prediction IDs")
    comparisons = []
    for protocol, pair, metric, subgroup, weighting in product(
            ("U20", "P20"), PAIRS, ("ndcg", "recall"),
            ("all", "repeat", "novel"), ("event", "user")):
        left, right = pair
        cid = f"{protocol}-{left}-minus-{right}-{metric}-{subgroup}-{weighting}"
        if (protocol, left) not in main_ids or (protocol, right) not in main_ids:
            unavailable.append(cid + ": unavailable (required fixed method not exported)")
            continue
        comparisons.append({"id": cid, "left": main_ids[protocol, left],
                            "right": main_ids[protocol, right], "metric": metric,
                            "subgroup": subgroup, "weighting": weighting})
    config = {"schema_version": 1, "analysis_id": analysis_id, "engine": "ranking",
              "inputs": [{"path": str(run), "reason": "Pilot-informed development evidence; within-protocol fixed comparisons"}],
              "series": [{"id": p["id"], "prediction_id": p["id"], "run_id": manifest["bundle_id"]}
                         for p in sorted(selected, key=lambda p: p["id"])],
              "k": 10, "comparisons": comparisons,
              "bootstrap": {"replicates": 1000, "seed": 20261025, "confidence": 0.95}}
    validate(root, "analysis_config", config)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "analysis.json"
    b.write(path, config)
    print("Prepared analyzer configuration only: " + str(path))
    print("Run manifest SHA256: " + b.sha(manifest_path))
    print("Runner result SHA256: " + b.sha(result_path))
    print("Excluded other-panel exports: " + json.dumps(excluded))
    print("Unavailable fixed methods/comparisons: " + json.dumps(unavailable))
    print("Search whole-development scores are secondary; selection used V_search. "
          "Only native V_check avoids direct selection reuse; it is still reused development data.")
    print("Full inventory/array verification remains the built-in analyzer's responsibility.")
    print("Analyze: " + shlex.join(["uv", "run", "--frozen", "researchflow", "analyze",
                                   "--root", str(root), "--config", str(path)]))
    print("The analyzer prints its actual committed output directory; pass that to --analysis.")
    return config


def describe(root, analysis):
    """Print existing analysis rows and exact pins, without deriving new metrics or CIs."""
    root, analysis = root.expanduser().resolve(), analysis.expanduser().resolve()
    b = bridge_for(root)
    manifest = load(analysis / "manifest.json")
    validate(root, "analysis_manifest", manifest)
    if manifest["status"] != "succeeded":
        raise ValueError("Selected analysis did not succeed")
    path = member(analysis, manifest, manifest["metadata"]["analysis_path"], b.sha)
    result = load(path)
    validate(root, "analysis_result", result)
    if (result["analysis_id"] != manifest["metadata"]["analysis_id"]
            or result["checks"]["passed"] is not True or result["phases"] != ["development"]):
        raise ValueError("Analysis identity/checks/phase mismatch")
    print(json.dumps({"selected_analysis_path": str(analysis),
                      "analysis_id": result["analysis_id"], "bundle_id": manifest["bundle_id"],
                      "manifest_sha256": b.sha(analysis / "manifest.json"),
                      "analysis_sha256": b.sha(path), "metadata": manifest["metadata"],
                      "inputs": result["inputs"], "comparisons": result["comparisons"],
                      "metrics": result["metrics"], "coverage": result["coverage"],
                      "costs": result["costs"], "limitations": result["limitations"]},
                     indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="Full committed run directory, never a compact evidence export")
    mode.add_argument("--analysis", type=Path, help="Actual committed analysis directory printed by the CLI")
    parser.add_argument("--analysis-id", help="Fresh A-finish-* ID when preparing --run")
    args = parser.parse_args()
    try:
        if args.run:
            if not args.analysis_id:
                parser.error("--run requires --analysis-id")
            configure(args.root, args.run, args.analysis_id)
        else:
            if args.analysis_id:
                parser.error("--analysis-id is only used with --run")
            describe(args.root, args.analysis)
    except Exception as exc:
        print(f"Analysis selection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
