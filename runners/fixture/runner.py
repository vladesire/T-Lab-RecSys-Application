#!/usr/bin/env python3
"""Standalone synthetic ranking fixture. Requires Python 3.12+ and NumPy 2.x.

Invocation: python runner.py --request REQUEST.json --output EMPTY_PAYLOAD_DIR
No researchflow import, data access, model calls, or network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

PROTOCOL = {
    "protocol_id": "P-fixture", "split": "development",
    "history": "synthetic_prior_only", "timestamp_unit": "synthetic_tick",
    "tie_break": "preassigned_candidate_position",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} requires exactly {sorted(keys)}")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Invalid identifier")


def validate(request):
    exact(request, {"schema_version", "run_id", "scientific", "data", "identity"}, "request")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    identifier(request["run_id"])
    sci = request["scientific"]
    exact(sci, {"experiment_id", "branch_id", "hypothesis_ids", "phase", "runner_id",
                "protocol", "parameters"}, "scientific")
    for name in ("experiment_id", "branch_id", "runner_id"):
        identifier(sci[name])
    if not isinstance(sci["hypothesis_ids"], list):
        raise ValueError("hypothesis_ids must be a list")
    for value in sci["hypothesis_ids"]:
        identifier(value)
    if sci["phase"] != "fixture" or sci["runner_id"] != "fixture":
        raise ValueError("This synthetic runner only accepts phase=fixture and runner_id=fixture")
    if sci["protocol"] != PROTOCOL:
        raise ValueError("Fixture requires its explicit synthetic protocol")
    exact(sci["parameters"], {"score_scale"}, "parameters")
    scale = sci["parameters"]["score_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1e100:
        raise ValueError("score_scale must be finite and in (0, 1e100]")
    if request["data"] != []:
        raise ValueError("Fixture has no external data; data must be []")
    exact(request["identity"], {"scientific_sha256", "code_sha256", "data_sha256"}, "identity")
    for value in request["identity"].values():
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("Invalid SHA256 identity")
    return scale


def run(request, output):
    scale = validate(request)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Output must be an existing empty payload directory")
    # Exact supplied ranking_rows.json, embedded so the runner is self-contained.
    common = {
        "query_id": np.array(["q1"] * 3 + ["q2"] * 3 + ["q3"] * 3),
        "user_id": np.array(["u1"] * 6 + ["u2"] * 3),
        "item_id": np.array(list("abcdefghi")),
        "timestamp": np.array([100] * 3 + [200] * 3 + [300] * 3, dtype=np.int64),
        "label": np.array([1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=bool),
        "tie_key": np.array([1, 2, 3] * 3, dtype=np.uint64),
        "is_familiar": np.array([1] + [0] * 8, dtype=bool),
        "is_repeat_target": np.array([1] * 3 + [0] * 6, dtype=bool),
    }
    queries, candidates = [], []
    for start in (0, 3, 6):
        positive = next(i for i in range(start, start + 3) if common["label"][i])
        query_id = str(common["query_id"][start])
        queries.append({
            "query_id": query_id, "user_id": str(common["user_id"][start]),
            "target_item_id": str(common["item_id"][positive]),
            "timestamp": int(common["timestamp"][start]),
            "is_repeat_target": bool(common["is_repeat_target"][start]),
        })
        records = [{"item_id": str(common["item_id"][i]),
                    "is_familiar": bool(common["is_familiar"][i]),
                    "tie_key": int(common["tie_key"][i])}
                   for i in range(start, start + 3)]
        candidates.append({"query_id": query_id,
                           "candidate_set_sha256": digest(sorted(records, key=lambda x: x["item_id"]))})
    identity = {"protocol_sha256": digest(request["scientific"]["protocol"]),
                "query_set_sha256": digest(sorted(queries, key=lambda x: x["query_id"])),
                "candidate_set_sha256": digest(sorted(candidates, key=lambda x: x["query_id"]))}
    predictions = []
    scores = {"A": [3, 2, 1, 3, 2, 1, 1, 3, 2], "B": [2, 3, 1, 1, 0, 3, 3, 2, 1]}
    for method, values in scores.items():
        path = f"{method}.npz"
        np.savez(output / path, **common, score=np.asarray(values, dtype=np.float64) * scale)
        predictions.append({"id": method, "method_id": method, "protocol_id": "P-fixture",
                            "split": "development", "format": "ranking_npz_v1", "path": path,
                            "feature_count": 0, "identity": identity,
                            "tags": {"synthetic": True}})
    result = {
        "schema_version": 1, "kind": "ranking", "predictions": predictions, "metrics": [],
        "coverage": [{"n_queries": 3, "n_users": 2, "n_rows_per_method": 9, "synthetic": True}],
        "costs": [{"kind": "fixture", "model_calls": 0, "network_calls": 0}],
        "checks": {"passed": True, "items": [{"id": "synthetic_fixture", "status": "passed",
            "detail": "Infrastructure fixture only. Tie keys are preassigned candidate positions, independent of labels.",
            "evidence": ["A.npz", "B.npz"]}]},
        "limitations": ["Synthetic infrastructure fixture, not recommendation research evidence.",
                        "History flags are constructed; no temporal feature correctness is established.",
                        "Request identities are supplied by the executor, not independently verified here."],
        "artifacts": [],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(json.loads(args.request.read_text(encoding="utf-8")), args.output)
    except (ValueError, OSError, OverflowError) as exc:
        print(f"fixture runner: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
