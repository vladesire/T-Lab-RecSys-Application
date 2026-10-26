"""M08 isolation tests: real schemas/NPZ/subprocesses, documented dependency fakes.

Ranking fakes supply the independent tiny fixture's known ranks; they do not
implement M07's ranking or bootstrap algorithm. Full numerical integration is M18.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jsonschema
import numpy as np
import pytest

from researchflow import analysis as mod
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "tests/fixtures/schema_examples"
FIXTURE = json.loads((ROOT / "tests/fixtures/ranking_rows.json").read_text())
EXPECTED = json.loads((ROOT / "tests/fixtures/ranking_expected.json").read_text())
SCHEMAS = ROOT / "src/researchflow/schemas"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def validate(kind, value):
    try:
        jsonschema.Draft202012Validator(json.loads((SCHEMAS / (kind + ".json")).read_text()),
                                       format_checker=jsonschema.FormatChecker()).validate(value)
    except jsonschema.ValidationError as exc:
        raise WorkflowError("CONTRACT", f"{kind}: {exc.message}") from exc


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
        raise WorkflowError("CONTRACT", "Unsafe relative path")
    return root / path


def inventory(bundle):
    return [{"path": p.relative_to(bundle).as_posix(), "sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in sorted(bundle.rglob("*")) if p.is_file() and p != bundle / "manifest.json"]


def seal(bundle, manifest):
    manifest["files"] = inventory(bundle)
    write(bundle / "manifest.json", manifest)


def verify(bundle_dir: Path, expected_kind: str | None = None) -> dict:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    validate("run_manifest" if manifest["bundle_kind"] == "run" else "analysis_manifest", manifest)
    if expected_kind and manifest["bundle_kind"] != expected_kind:
        raise WorkflowError("CONTRACT", "Bundle kind mismatch")
    if manifest["files"] != inventory(bundle_dir):
        raise WorkflowError("IDENTITY", "Inventory mismatch")
    return manifest


def canned_rows(arrays, *, k):
    assert k == 2
    assert not any(a.dtype.hasobject for a in arrays.values())
    assert set(arrays) == {"query_id", "user_id", "item_id", "timestamp", "label", "score", "tie_key",
                           "is_familiar", "is_repeat_target"}
    method = "A" if arrays["score"].tolist() == FIXTURE["score_A"] else "B"
    rows = []
    for q, rank in zip(("q1", "q2", "q3"), EXPECTED[f"{method}_ranks"]):
        indices = [i for i, val in enumerate(arrays["query_id"]) if str(val) == q]
        positive = next(i for i in indices if arrays["label"][i])
        rec = dict(query_id=q, user_id=str(arrays["user_id"][positive]),
                   target_item_id=str(arrays["item_id"][positive]), timestamp=int(arrays["timestamp"][positive]),
                   is_repeat_target=bool(arrays["is_repeat_target"][positive]))
        candidates = sorted([dict(item_id=str(arrays["item_id"][i]), is_familiar=bool(arrays["is_familiar"][i]),
                                  tie_key=int(arrays["tie_key"][i])) for i in indices], key=lambda c: c["item_id"])
        rows.append({**rec, "query_identity_sha256": digest(rec), "candidate_set_sha256": digest(candidates),
                     "ndcg": float(rank <= 2) / np.log2(rank + 1), "recall": float(rank <= 2),
                     "n_candidates": len(indices), "n_familiar_alternatives": 0})
    return rows


def fake_aggregate(rows, *, metric, weighting, subgroup="all"):
    selected = [r for r in rows if subgroup == "all" or r["is_repeat_target"] == (subgroup == "repeat")]
    if not selected:
        return {"value": None, "n_queries": 0, "n_users": 0}
    users = {r["user_id"] for r in selected}
    if weighting == "event":
        value = sum(r[metric] for r in selected) / len(selected)
    else:
        value = sum(sum(r[metric] for r in selected if r["user_id"] == u) /
                    sum(r["user_id"] == u for r in selected) for u in users) / len(users)
    return {"value": value, "n_queries": len(selected), "n_users": len(users)}


@pytest.fixture
def isolated(monkeypatch):
    calls = {"pair": [], "load": [], "process": []}
    def begin_bundle(output_root: Path, bundle_id: str) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        if (output_root / bundle_id).exists():
            raise WorkflowError("IDENTITY", "Bundle collision")
        stage = output_root / ("stage-" + bundle_id)
        stage.mkdir()
        return stage
    def commit_bundle(stage_dir: Path, final_dir: Path, manifest) -> Path:
        assert manifest["files"] == inventory(stage_dir)
        validate("analysis_manifest", manifest)
        write(stage_dir / "manifest.json", manifest)
        if final_dir.exists():
            raise WorkflowError("IDENTITY", "Bundle collision")
        stage_dir.rename(final_dir)
        return final_dir
    def paired_interval(left_rows, right_rows, *, metric, weighting, subgroup, replicates, seed, confidence=0.95):
        calls["pair"].append((replicates, seed, confidence))
        if not any(subgroup == "all" or r["is_repeat_target"] == (subgroup == "repeat") for r in left_rows):
            raise WorkflowError("CONTRACT", "Empty paired subgroup")
        left = fake_aggregate(left_rows, metric=metric, weighting=weighting, subgroup=subgroup)
        right = fake_aggregate(right_rows, metric=metric, weighting=weighting, subgroup=subgroup)
        estimate = left["value"] - right["value"]
        # Deliberately canned interval: M08 must forward it, never calculate it.
        return dict(estimate=estimate, ci_low=-0.9, ci_high=0.8,
                    n_queries=left["n_queries"], n_users=left["n_users"])
    def run_process(argv, cwd: Path, timeout_seconds: float, stdout_path: Path, stderr_path: Path,
                    env=None, *, stop_file=None, memory_limit_bytes=None):
        calls["process"].append(list(argv))
        start = datetime.now(timezone.utc).isoformat()
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            process = subprocess.run(argv, cwd=cwd, timeout=timeout_seconds, stdout=out, stderr=err)
        return dict(schema_version=1, argv=list(argv), status="succeeded" if process.returncode == 0 else "failed",
                    exit_code=process.returncode, started_at=start, finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=0.01, peak_rss_bytes=None, warnings=[])
    for name, fn in {"load_document": lambda p: json.loads(p.read_text()), "canonical_bytes": canonical,
                     "digest_document": digest, "sha256_file": sha, "atomic_write_json": write,
                     "validate_document": validate, "contained_path": contained,
                     "begin_bundle": begin_bundle, "commit_bundle": commit_bundle, "verify_bundle": verify,
                     "bundle_fingerprint": digest, "per_query_metrics": canned_rows,
                     "aggregate": fake_aggregate, "paired_interval": paired_interval, "run_process": run_process}.items():
        monkeypatch.setattr(mod, name, fn)
    original_load = np.load
    def observed_load(path, **kwargs):
        calls["load"].append((Path(path), kwargs))
        return original_load(path, **kwargs)
    monkeypatch.setattr(mod.np, "load", observed_load)
    return calls


def make_run(root, rid="R1", phase="fixture", origin="fresh_execution", method="A", status="succeeded",
             candidate_change=False, query_change=False, empty_repeat=False, protocol_change=False):
    bundle = root / "runs" / rid
    (bundle / "payload").mkdir(parents=True)
    manifest = json.loads((EXAMPLES / "run_manifest.json").read_text())
    manifest.update(bundle_id=rid, status=status)
    meta = manifest["metadata"]
    meta.update(phase=phase, origin=origin, result_path="payload/result.json" if status == "succeeded" else None,
                process_status=status)
    # Real, nonzero identities; fixture timestamps remain labelled by phase.
    meta["identity"] = {key: digest([rid, key]) for key in meta["identity"]}
    if status == "succeeded":
        resolved = json.loads((EXAMPLES / "resolved_experiment.json").read_text())
        resolved["scientific"]["phase"] = phase
        resolved["identity"] = meta["identity"]
        if protocol_change:
            resolved["scientific"]["protocol"]["description"] = "Different history rules"
        write(bundle / "resolved_experiment.json", resolved)
        arrays = {key: np.array(value) for key, value in FIXTURE.items() if not key.startswith("score_")}
        arrays["tie_key"] = arrays["tie_key"].astype("uint64")
        arrays["score"] = np.array(FIXTURE[f"score_{method}"], dtype=float)
        if candidate_change:
            arrays["tie_key"][1] = 99
        if query_change:
            arrays["timestamp"][0:3] += 1
        if empty_repeat:
            arrays["is_repeat_target"][:] = False
            arrays["is_familiar"][:] = False
        np.savez(bundle / "payload/predictions.npz", **arrays)
        rows = canned_rows(arrays, k=2)
        query_records = [{key: row[key] for key in ("query_id", "user_id", "target_item_id", "timestamp", "is_repeat_target")}
                         for row in rows]
        ids = dict(protocol_sha256=digest(resolved["scientific"]["protocol"]), query_set_sha256=digest(query_records),
                   candidate_set_sha256=digest([dict(query_id=r["query_id"], candidate_set_sha256=r["candidate_set_sha256"])
                                                for r in rows]))
        runner = json.loads((EXAMPLES / "runner_result.json").read_text())
        runner["predictions"] = [dict(id="p", method_id=method, protocol_id="P-fixture", split="development",
                                      format="ranking_npz_v1", path="predictions.npz", identity=ids, feature_count=0)]
        runner["artifacts"][0]["path"] = "predictions.npz"
        runner["coverage"] = [{"n_events": 3}]
        runner["costs"] = [{"seconds": 1.5}]
        write(bundle / "payload/result.json", runner)
    seal(bundle, manifest)
    return bundle


def configure(root, runs=("R1",), *, comparisons=None, external=None):
    config = json.loads((EXAMPLES / "analysis_config.json").read_text())
    config.update(analysis_id="test", k=2,
                  inputs=[{"path": "runs/" + rid, "reason": "Explicit synthetic acceptance fixture"} for rid in runs],
                  series=[dict(id="s" + str(i), run_id=rid, prediction_id="p") for i, rid in enumerate(runs)],
                  comparisons=comparisons if comparisons is not None else [],
                  bootstrap=dict(replicates=17, seed=42, confidence=0.9))
    if external:
        config.update(engine="external", external=external)
    write(root / "config.json", config)
    return config


def pair(**changes):
    return dict(id="named-difference", left="s0", right="s1", metric="ndcg", weighting="event", subgroup="all", **changes)


def run(root):
    output = mod.analyze(root, Path("config.json"))
    return output, json.loads((output / "analysis.json").read_text())


def mutate_run(bundle, relative, mutate):
    doc = json.loads((bundle / relative).read_text())
    mutate(doc)
    write(bundle / relative, doc)
    manifest = json.loads((bundle / "manifest.json").read_text())
    seal(bundle, manifest)


def set_protocol_family(bundle, member, allowed):
    protocol = {"id": "warmup-development-v1", "candidates": allowed,
                "history": "strict-past-complete"}
    mutate_run(bundle, "resolved_experiment.json",
               lambda doc: doc["scientific"].update(protocol=protocol))
    def descriptor(doc):
        doc["predictions"][0]["protocol_id"] = member
        doc["predictions"][0]["identity"]["protocol_sha256"] = digest(protocol)
    mutate_run(bundle, "payload/result.json", descriptor)


def test_declared_family_member_is_analyzed(tmp_path, isolated):
    bundle = make_run(tmp_path)
    set_protocol_family(bundle, "U20", ["U20", "P20"])
    configure(tmp_path)
    _, result = run(tmp_path)
    assert result["checks"]["passed"]
    assert len(result["metrics"]) == 12
    assert {row["protocol_id"] for row in result["metrics"]} == {"U20"}


def test_undeclared_family_member_is_rejected(tmp_path, isolated):
    bundle = make_run(tmp_path)
    set_protocol_family(bundle, "U100", ["U20", "P20"])
    configure(tmp_path)
    _, result = run(tmp_path)
    assert not result["checks"]["passed"]
    assert not result["metrics"]
    assert any("Undeclared candidate protocol" in item["detail"] for item in result["checks"]["items"])


def test_distinct_family_members_are_not_paired(tmp_path, isolated):
    left = make_run(tmp_path)
    right = make_run(tmp_path, "R2", method="B")
    set_protocol_family(left, "U20", ["U20", "P20"])
    set_protocol_family(right, "P20", ["U20", "P20"])
    configure(tmp_path, ("R1", "R2"), comparisons=[pair()])
    _, result = run(tmp_path)
    assert len(result["metrics"]) == 24
    assert result["comparisons"][0]["status"] == "unavailable"
    assert "identity mismatch" in result["comparisons"][0]["reason"].lower()


def test_known_metrics_named_comparison_and_inventory(tmp_path, isolated, monkeypatch):
    make_run(tmp_path)
    make_run(tmp_path, "R2", method="B")
    configure(tmp_path, ("R1", "R2"), comparisons=[pair()])
    # Resolving a config/input must not depend on the process working directory.
    monkeypatch.chdir(tmp_path.parent)
    output, result = run(tmp_path)
    assert len(result["metrics"]) == 24
    for method, sid in (("A", "s0"), ("B", "s1")):
        for metric in ("ndcg", "recall"):
            for weighting in ("event", "user"):
                cell = next(c for c in result["metrics"] if c["series_id"] == sid and
                            c["metric"] == metric + "@2" and c["weighting"] == weighting and c["subgroup"] == "all")
                assert cell["value"] == pytest.approx(EXPECTED[f"{method}_{weighting}_{metric}"])
                assert (cell["n_queries"], cell["n_users"]) == (3, 2)
    comp = result["comparisons"][0]
    assert comp["id"] == "named-difference" and comp["status"] == "computed"
    assert comp["estimate"] == pytest.approx(EXPECTED["A_event_ndcg"] - EXPECTED["B_event_ndcg"])
    assert (comp["ci_low"], comp["ci_high"]) == (-0.9, 0.8)
    assert isolated["pair"] == [(17, 42, 0.9)]
    assert all(kwargs == {"allow_pickle": False} for _, kwargs in isolated["load"])
    assert len((output / "per_query/s0.jsonl").read_text().splitlines()) == 3
    assert json.loads((output / "tables/metrics.json").read_text()) == result["metrics"]
    assert json.loads((output / "tables/comparisons.json").read_text()) == result["comparisons"]
    assert (output / "inputs/R1/manifest.json").read_bytes() == (tmp_path / "runs/R1/manifest.json").read_bytes()
    assert {c["run_id"] for c in result["costs"]} == {"R1", "R2"}
    manifest = verify(output, "analysis")
    assert manifest["metadata"]["analyzer_identity"]["code_sha256"]
    assert len(manifest["metadata"]["input_fingerprints"]) == 2
    before = sha(output / "manifest.json")
    with pytest.raises(WorkflowError):
        run(tmp_path)
    assert sha(output / "manifest.json") == before


@pytest.mark.parametrize("change", ["query_change", "candidate_change", "protocol_change"])
def test_identity_mismatch_prevents_pairing(tmp_path, isolated, change):
    make_run(tmp_path)
    make_run(tmp_path, "R2", **{change: True})
    configure(tmp_path, ("R1", "R2"), comparisons=[pair()])
    _, result = run(tmp_path)
    assert len(result["metrics"]) == 24
    assert result["comparisons"][0]["status"] == "unavailable"
    assert "identity mismatch" in result["comparisons"][0]["reason"]
    assert result["comparisons"][0]["estimate"] is None
    assert not isolated["pair"]


def test_failed_input_visible_not_zero(tmp_path, isolated):
    make_run(tmp_path)
    make_run(tmp_path, "R2", status="failed")
    configure(tmp_path, ("R1", "R2"), comparisons=[pair()])
    _, result = run(tmp_path)
    assert len(result["inputs"]) == 2
    assert {m["series_id"] for m in result["metrics"]} == {"s0"}
    excluded = next(c for c in result["coverage"] if c.get("series_id") == "s1")
    assert excluded["status"] == "excluded" and "failed" in excluded["reason"]
    assert result["comparisons"][0]["estimate"] is None
    assert not result["checks"]["passed"]


def test_reproductions_and_phases_preserved_without_pooling(tmp_path, isolated):
    for rid, phase in (("R1", "fixture"), ("R2", "development"), ("R3", "final")):
        make_run(tmp_path, rid, phase=phase, origin="reproduction")
    configure(tmp_path, ("R1", "R2", "R3"))
    _, result = run(tmp_path)
    assert result["phases"] == ["development", "final", "fixture"]
    assert {i["origin"] for i in result["inputs"]} == {"reproduction"}
    assert len(result["metrics"]) == 36
    assert all(m["n_queries"] == 3 for m in result["metrics"] if m["subgroup"] == "all")
    assert result["comparisons"] == [] and isolated["pair"] == []


def test_empty_subgroup_null_and_comparison_unavailable(tmp_path, isolated):
    make_run(tmp_path, empty_repeat=True)
    make_run(tmp_path, "R2", empty_repeat=True)
    spec = pair()
    spec["subgroup"] = "repeat"
    configure(tmp_path, ("R1", "R2"), comparisons=[spec])
    _, result = run(tmp_path)
    empty = [m for m in result["metrics"] if m["subgroup"] == "repeat"]
    assert len(empty) == 8
    assert all(m["value"] is None and m["n_queries"] == m["n_users"] == 0 for m in empty)
    assert result["comparisons"][0]["status"] == "unavailable"
    assert "Empty" in result["comparisons"][0]["reason"]


@pytest.mark.parametrize("field", ["protocol_sha256", "query_set_sha256", "candidate_set_sha256"])
def test_forged_descriptor_excludes_series(tmp_path, isolated, field):
    bundle = make_run(tmp_path)
    mutate_run(bundle, "payload/result.json", lambda r: r["predictions"][0]["identity"].update({field: "a" * 64}))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert result["metrics"] == []
    assert any(field.split("_sha256")[0] in c["detail"] for c in result["checks"]["items"] if c["status"] == "failed")


def test_duplicate_series_unknown_selection_and_blank_reason(tmp_path, isolated):
    make_run(tmp_path)
    config = configure(tmp_path)
    config["series"].append(deepcopy(config["series"][0]))
    write(tmp_path / "config.json", config)
    with pytest.raises(WorkflowError, match="Duplicate series"):
        run(tmp_path)
    config["series"] = [dict(id="x", run_id="not-selected", prediction_id="p")]
    write(tmp_path / "config.json", config)
    with pytest.raises(WorkflowError, match="unselected"):
        run(tmp_path)
    config["inputs"][0]["reason"] = " "
    write(tmp_path / "config.json", config)
    with pytest.raises(WorkflowError, match="explicit"):
        run(tmp_path)
    assert not (tmp_path / "analyses").exists()


def test_corrupt_bundle_is_rejected_before_staging(tmp_path, isolated):
    make_run(tmp_path)
    configure(tmp_path)
    (tmp_path / "runs/R1/payload/predictions.npz").write_bytes(b"corruption")
    with pytest.raises(WorkflowError, match="Inventory"):
        run(tmp_path)
    assert not (tmp_path / "analyses").exists()


def test_only_declared_safe_numeric_predictions_are_loaded(tmp_path, isolated):
    bundle = make_run(tmp_path)
    configure(tmp_path)
    # An object array must fail on NumPy load, before reaching the ranking fake.
    path = bundle / "payload/predictions.npz"
    np.savez(path, query_id=np.array([object()], dtype=object))
    seal(bundle, json.loads((bundle / "manifest.json").read_text()))
    _, result = run(tmp_path)
    assert result["metrics"] == []
    assert any("Invalid ranking NPZ" in c["detail"] for c in result["checks"]["items"])
    assert isolated["load"][0][1] == {"allow_pickle": False}


def test_unlisted_and_traversing_prediction_paths_excluded(tmp_path, isolated):
    bundle = make_run(tmp_path)
    mutate_run(bundle, "payload/result.json", lambda r: r["predictions"][0].update(path="../resolved_experiment.json"))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert not isolated["load"] and not result["metrics"]


def external_script(root, mutation="", *, fail=False):
    script = root / "analyzer.py"
    script.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
request = json.loads(Path(a.request).read_text())
assert Path(a.request).is_absolute() and Path(a.output).is_absolute()
output = Path(a.output)
inputs = []
for item in request['inputs']:
    meta = json.loads((Path(item['bundle_path']) / 'manifest.json').read_text())['metadata']
    inputs.append(dict(run_id=item['run_id'], manifest_sha256=item['manifest_sha256'], phase=meta['phase'], origin=meta['origin']))
result = dict(schema_version=1, analysis_id=request['analysis_id'], inputs=inputs,
              phases=sorted({i['phase'] for i in inputs}), metrics=[], comparisons=[], coverage=[], costs=[],
              checks=dict(passed=True, items=[dict(id='synthetic', status='unverified', detail='Synthetic external fixture', evidence=['evidence.json'])]),
              limitations=['Fixture method only'])
(output / 'evidence.json').write_text('{"synthetic":true}')
''' + mutation + "\n(output / 'analysis.json').write_text(json.dumps(result))\n" + ("raise SystemExit(3)\n" if fail else ""))
    return dict(command=[sys.executable, "analyzer.py"], code_paths=["analyzer.py"], timeout_seconds=10)


def test_external_real_file_protocol_and_identity(tmp_path, isolated):
    make_run(tmp_path, phase="development", origin="reproduction")
    config = configure(tmp_path, external=external_script(tmp_path))
    output, result = run(tmp_path)
    argv = isolated["process"][0]
    assert argv[2] == "--request" and argv[4] == "--output"
    request = json.loads((output / "analysis_request.json").read_text())
    assert request["config"] == config and Path(request["inputs"][0]["bundle_path"]).is_absolute()
    assert result["inputs"][0]["phase"] == "development"
    assert result["inputs"][0]["origin"] == "reproduction"
    assert result["checks"]["items"][0]["evidence"] == ["external/evidence.json"]
    assert any("not independently verified" in x for x in result["limitations"])
    manifest = verify(output, "analysis")
    identities = manifest["metadata"]["analyzer_identity"]["files"]
    assert any(x["sha256"] == sha(tmp_path / "analyzer.py") for x in identities)


@pytest.mark.parametrize("mutation", [
    "result['inputs'][0]['manifest_sha256'] = 'f' * 64",
    "result['inputs'][0]['origin'] = 'legacy_import'",
    "result['inputs'][0]['phase'] = 'final'",
    "result['phases'] = ['final']",
    "result['analysis_id'] = 'wrong'",
    "result['unexpected'] = True",
    "result['checks']['items'][0]['evidence'] = ['../analysis_config.json']",
    "result['checks']['items'][0]['evidence'] = ['missing.json']",
    "(output / 'escape').symlink_to(Path(a.request))",
    "result['checks']['items'][0]['status'] = 'failed'",
])
def test_invalid_external_output_rejected(tmp_path, isolated, mutation):
    make_run(tmp_path)
    configure(tmp_path, external=external_script(tmp_path, mutation))
    with pytest.raises(WorkflowError):
        run(tmp_path)
    assert not list((tmp_path / "analyses").iterdir())


def test_external_process_failure_not_committed(tmp_path, isolated):
    make_run(tmp_path)
    configure(tmp_path, external=external_script(tmp_path, fail=True))
    with pytest.raises(WorkflowError) as err:
        run(tmp_path)
    assert err.value.code == "EXECUTION"
    assert not list((tmp_path / "analyses").iterdir())


def test_failed_source_checks_costs_and_limits_survive(tmp_path, isolated):
    bundle = make_run(tmp_path)
    def change(r):
        r["checks"]["passed"] = False
        r["checks"]["items"][0].update(status="failed", detail="Synthetic leakage reference test failed")
    mutate_run(bundle, "payload/result.json", change)
    configure(tmp_path)
    _, result = run(tmp_path)
    assert not result["checks"]["passed"]
    assert any("leakage" in c["detail"] for c in result["checks"]["items"])
    assert result["costs"][0]["seconds"] == 1.5
    assert any(x.startswith("R1:") for x in result["limitations"])


def test_non_document_json_artifacts_do_not_hide_resolved_protocol(tmp_path, isolated):
    bundle = make_run(tmp_path)
    write(bundle / "payload/table.json", [1, 2, 3])
    seal(bundle, json.loads((bundle / "manifest.json").read_text()))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert len(result["metrics"]) == 12


def test_request_snapshot_also_supplies_protocol(tmp_path, isolated):
    bundle = make_run(tmp_path)
    resolved = json.loads((bundle / "resolved_experiment.json").read_text())
    request = dict(schema_version=1, run_id="R1", scientific=resolved["scientific"], identity=resolved["identity"], data=[])
    write(bundle / "runner_request.json", request)
    (bundle / "resolved_experiment.json").unlink()
    seal(bundle, json.loads((bundle / "manifest.json").read_text()))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert len(result["metrics"]) == 12


def test_conflicting_resolved_snapshots_are_excluded(tmp_path, isolated):
    bundle = make_run(tmp_path)
    resolved = json.loads((bundle / "resolved_experiment.json").read_text())
    resolved["scientific"]["protocol"]["description"] = "Contradiction"
    write(bundle / "another_snapshot.json", resolved)
    seal(bundle, json.loads((bundle / "manifest.json").read_text()))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert not result["metrics"]
    assert any("Ambiguous" in c["detail"] for c in result["checks"]["items"])


def test_aggregate_failure_cannot_publish_partial_series(tmp_path, isolated, monkeypatch):
    make_run(tmp_path)
    configure(tmp_path)
    def fake(rows, *, metric, weighting, subgroup="all"):
        if subgroup == "novel":
            raise WorkflowError("CONTRACT", "Synthetic aggregation rejection")
        return fake_aggregate(rows, metric=metric, weighting=weighting, subgroup=subgroup)
    monkeypatch.setattr(mod, "aggregate", fake)
    _, result = run(tmp_path)
    assert result["metrics"] == []
    assert any(c.get("status") == "excluded" for c in result["coverage"])


def test_invalid_runner_envelope_visible_as_exclusion(tmp_path, isolated):
    bundle = make_run(tmp_path)
    mutate_run(bundle, "payload/result.json", lambda r: r.update(invented_field=True))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert result["metrics"] == []
    assert not result["checks"]["passed"]
    assert any("Excluded invalid result" in c["detail"] for c in result["checks"]["items"])


def test_external_failed_run_cannot_contribute_metric(tmp_path, isolated):
    make_run(tmp_path, status="failed")
    cell = json.loads((EXAMPLES / "analysis_result.json").read_text())["metrics"][0]
    cell["series_id"] = "s0"
    mutation = "result['metrics'] = " + repr([cell])
    configure(tmp_path, external=external_script(tmp_path, mutation))
    with pytest.raises(WorkflowError, match="failed run"):
        run(tmp_path)


def test_external_analyzer_code_directory_is_fingerprinted(tmp_path, isolated):
    make_run(tmp_path)
    ext = external_script(tmp_path)
    code = tmp_path / "scientific_code"
    code.mkdir()
    (code / "method.txt").write_text("Synthetic method identity fixture")
    ext["code_paths"].append("scientific_code")
    configure(tmp_path, external=ext)
    output, _ = run(tmp_path)
    files = verify(output, "analysis")["metadata"]["analyzer_identity"]["files"]
    assert any(f["sha256"] == sha(code / "method.txt") for f in files)


def test_code_mutation_during_external_execution_is_rejected(tmp_path, isolated):
    make_run(tmp_path)
    mutation = "Path(__file__).write_text(Path(__file__).read_text() + '\\n# mutated\\n')"
    configure(tmp_path, external=external_script(tmp_path, mutation))
    with pytest.raises(WorkflowError, match="code changed"):
        run(tmp_path)
    assert not list((tmp_path / "analyses").iterdir())


def test_unselected_bundle_and_undeclared_npz_never_loaded(tmp_path, isolated):
    make_run(tmp_path)
    make_run(tmp_path, "unselected", status="failed")
    # An unrelated archive in an explicitly selected bundle must not be loaded.
    bundle = tmp_path / "runs/R1"
    np.savez(bundle / "payload/unused.npz", unsafe=np.array([object()], dtype=object))
    seal(bundle, json.loads((bundle / "manifest.json").read_text()))
    configure(tmp_path)
    _, result = run(tmp_path)
    assert [x["run_id"] for x in result["inputs"]] == ["R1"]
    assert [path.name for path, _ in isolated["load"]] == ["predictions.npz"]


def test_cross_phase_comparison_is_not_computed(tmp_path, isolated):
    make_run(tmp_path)
    make_run(tmp_path, "R2", phase="final")
    configure(tmp_path, ("R1", "R2"), comparisons=[pair()])
    _, result = run(tmp_path)
    assert len(result["metrics"]) == 24
    assert result["comparisons"][0]["reason"] == "Input phases differ"
    assert not isolated["pair"]
