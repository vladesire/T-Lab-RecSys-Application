"""Isolated M18 checks: real standalone runner, explicitly faked service orchestration."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import jsonschema
import numpy as np
import pytest
import yaml

from researchflow.contracts import WorkflowError

sm = importlib.import_module("researchflow.smoke")
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(kind, value):
    jsonschema.Draft202012Validator(read(ROOT / f"src/researchflow/schemas/{kind}.json"),
                                   format_checker=jsonschema.FormatChecker()).validate(value)


def request():
    config = yaml.safe_load((ROOT / "configs/examples/fixture_experiment.yaml").read_text())
    result = {"schema_version": 1, "run_id": "R-fixture", "scientific": config["scientific"],
              "data": [], "identity": {key: canonical_hash(value) for key, value in {
                  "scientific_sha256": config["scientific"], "code_sha256": "synthetic-test-code",
                  "data_sha256": []}.items()}}
    validate("runner_request", result)
    return result


def launch(tmp_path, value, name="payload"):
    payload = tmp_path / name
    payload.mkdir()
    req = write(tmp_path / f"{name}.json", value)
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "runners/fixture/runner.py"),
                           "--request", str(req), "--output", str(payload)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=30)
    return proc, payload


def test_standalone_matches_supplied_arrays_ranks_means_and_identities(tmp_path):
    req = request()
    proc, payload = launch(tmp_path, req)
    assert proc.returncode == 0, proc.stderr
    result = read(payload / "result.json")
    validate("runner_result", result)
    assert result["kind"] == "ranking" and result["checks"]["passed"]
    assert "researchflow" not in (ROOT / "runners/fixture/runner.py").read_text().split("import numpy")[1]
    rows = read(ROOT / "tests/fixtures/ranking_rows.json")
    expected = read(ROOT / "tests/fixtures/ranking_expected.json")
    query_records, candidate_records = [], []
    for pred in result["predictions"]:
        with np.load(payload / pred["path"], allow_pickle=False) as archive:
            arrays = dict(archive)
        assert all(arr.ndim == 1 and len(arr) == 9 and arr.dtype.kind != "O" for arr in arrays.values())
        assert arrays["tie_key"].dtype == np.uint64
        for key, array in arrays.items():
            np.testing.assert_array_equal(array, rows["score_" + pred["id"] if key == "score" else key])
        ranks, ndcg, recall = [], [], []
        for start in (0, 3, 6):
            indices = list(range(start, start + 3))
            ordered = sorted(indices, key=lambda i: (-float(arrays["score"][i]),
                                                     int(arrays["tie_key"][i]), str(arrays["item_id"][i])))
            target = next(i for i in indices if arrays["label"][i])
            rank = ordered.index(target) + 1
            ranks.append(rank)
            ndcg.append(1 / math.log2(rank + 1) if rank <= 2 else 0.)
            recall.append(float(rank <= 2))
            if pred["id"] == "A":
                query_records.append({"query_id": str(arrays["query_id"][start]),
                    "user_id": str(arrays["user_id"][start]), "target_item_id": str(arrays["item_id"][target]),
                    "timestamp": int(arrays["timestamp"][start]),
                    "is_repeat_target": bool(arrays["is_repeat_target"][start])})
                candidates = [{"item_id": str(arrays["item_id"][i]),
                               "is_familiar": bool(arrays["is_familiar"][i]),
                               "tie_key": int(arrays["tie_key"][i])} for i in indices]
                candidate_records.append({"query_id": str(arrays["query_id"][start]),
                                          "candidate_set_sha256": canonical_hash(candidates)})
        assert ranks == expected[pred["id"] + "_ranks"]
        for metric, values in [("ndcg", ndcg), ("recall", recall)]:
            assert sum(values) / 3 == pytest.approx(expected[f"{pred['id']}_event_{metric}"])
            assert ((values[0] + values[1]) / 2 + values[2]) / 2 == pytest.approx(expected[f"{pred['id']}_user_{metric}"])
    identities = [p["identity"] for p in result["predictions"]]
    assert identities[0] == identities[1] == {
        "protocol_sha256": canonical_hash(req["scientific"]["protocol"]),
        "query_set_sha256": canonical_hash(query_records),
        "candidate_set_sha256": canonical_hash(candidate_records)}


def test_runner_parameter_changes_scores_but_preserves_candidates_and_previous_payload(tmp_path):
    req = request()
    first, a = launch(tmp_path, req, "one")
    before = {p.name: p.read_bytes() for p in a.iterdir()}
    req["scientific"]["parameters"]["score_scale"] = 2
    second, b = launch(tmp_path, req, "two")
    assert first.returncode == second.returncode == 0
    assert read(a / "result.json")["predictions"] == read(b / "result.json")["predictions"]
    with np.load(a / "A.npz") as x, np.load(b / "A.npz") as y:
        np.testing.assert_array_equal(y["score"], 2 * x["score"])
    assert before == {p.name: p.read_bytes() for p in a.iterdir()}


@pytest.mark.parametrize("change", ["unknown", "zero", "negative", "nan", "bool", "phase", "protocol", "data", "envelope"])
def test_runner_rejects_invalid_or_unexposed_inputs_before_writing(tmp_path, change):
    req = request()
    if change == "unknown":
        req["scientific"]["parameters"]["hidden_choice"] = 1
    elif change in {"zero", "negative", "nan", "bool"}:
        req["scientific"]["parameters"]["score_scale"] = {"zero": 0, "negative": -1, "nan": float("nan"), "bool": True}[change]
    elif change == "phase":
        req["scientific"]["phase"] = "final"
    elif change == "protocol":
        req["scientific"]["protocol"]["history"] = "unexposed"
    elif change == "data":
        req["data"] = [{}]
    else:
        req["extra"] = True
    proc, payload = launch(tmp_path, req)
    assert proc.returncode != 0 and proc.stderr
    assert list(payload.iterdir()) == []


def test_runner_refuses_existing_output(tmp_path):
    proc, payload = launch(tmp_path, request())
    assert proc.returncode == 0
    before = {p.name: p.read_bytes() for p in payload.iterdir()}
    proc = subprocess.run([sys.executable, str(ROOT / "runners/fixture/runner.py"),
                           "--request", str(tmp_path / "payload.json"), "--output", str(payload)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert before == {p.name: p.read_bytes() for p in payload.iterdir()}


@pytest.fixture
def project(tmp_path):
    for rel in sm._FIXTURES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return tmp_path


@pytest.fixture
def services(monkeypatch):
    """Only documented dependency APIs are patched. Outputs use shared schemas."""
    calls = []

    def config(kind, path):
        value = read(path)
        validate(kind, value)
        return value

    def example(kind):
        return copy.deepcopy(read(ROOT / f"tests/fixtures/schema_examples/{kind}.json"))

    def execute(project_root, config_path, *, run_id=None, freeze_path=None):
        cfg = config("experiment_config", config_path)
        assert freeze_path is None and cfg["scientific"]["phase"] == "fixture"
        for kind, rel in [("hypothesis", "hypotheses/H-fixture.yaml"), ("branch", "branches/B-fixture.yaml"),
                          ("decision", "decisions/D-fixture.yaml")]:
            validate(kind, yaml.safe_load((project_root / "research" / rel).read_text()))
        manifest = example("run_manifest")
        manifest["bundle_id"] = run_id
        manifest["metadata"]["identity"]["scientific_sha256"] = canonical_hash(cfg["scientific"])
        bundle = project_root / "runs" / run_id
        validate("run_manifest", manifest)
        write(bundle / "manifest.json", manifest)
        calls.append(("execute", project_root, cfg))
        return bundle

    def analyze(project_root, config_path):
        cfg = config("analysis_config", config_path)
        assert cfg["series"][0]["run_id"] == "R-smoke-" + project_root.parent.name
        assert (project_root / cfg["inputs"][0]["path"] / "manifest.json").is_file()
        bundle = project_root / "analyses/A-fixture-fake"
        result = example("analysis_result")
        validate("analysis_result", result)
        write(bundle / "analysis.json", result)
        manifest = example("analysis_manifest")
        validate("analysis_manifest", manifest)
        write(bundle / "manifest.json", manifest)
        calls.append(("analyze", project_root, cfg))
        return bundle

    def build_view(project_root, config_path):
        cfg = config("view_config", config_path)
        calls.append(("view", project_root, cfg))
        state = example("state_view")
        validate("state_view", state)
        return state

    def write_view(state, output_dir):
        write(output_dir / "view.json", state)
        (output_dir / "view.md").write_text("Synthetic mocked view\n")
        return output_dir

    def packet(project_root, config_path, output_dir):
        cfg = config("context_config", config_path)
        assert all((project_root / x["path"]).is_file() for x in cfg["sources"])
        assert cfg["profile"] == "results_reviewer"
        manifest = example("context_manifest")
        validate("context_manifest", manifest)
        write(output_dir / "manifest.json", manifest)
        calls.append(("packet", project_root, cfg))
        return output_dir

    def publication(project_root, config_path):
        cfg = config("publication_config", config_path)
        assert cfg["mode"] == "draft"
        assert cfg["analyses"][0]["sha256"] == canonical_hash(read(project_root / cfg["analyses"][0]["path"] / "manifest.json"))
        bundle = project_root / "build/paper/P-fixture-fake"
        manifest = example("publication_manifest")
        validate("publication_manifest", manifest)
        write(bundle / "manifest.json", manifest)
        calls.append(("publication", project_root, cfg))
        return bundle

    def doctor(project_root):
        calls.append(("doctor", project_root, None))
        return {"fake": True}

    def compile(snapshot_dir, output_root, *, timeout_seconds=120.):
        assert (snapshot_dir / "manifest.json").is_file()
        bundle = output_root / "fake-compiled"
        manifest = example("compiled_manifest")
        validate("compiled_manifest", manifest)
        write(bundle / "manifest.json", manifest)
        pdf = bundle / manifest["metadata"]["pdf_path"]
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"mock PDF, not TeX acceptance")
        calls.append(("compile", snapshot_dir, timeout_seconds))
        return bundle

    def inspect(compiled_dir):
        calls.append(("inspect", compiled_dir, None))
        return {"checks": {"passed": True}, "fake": True}

    for name, value in {"execute_experiment": execute, "analyze": analyze, "build_view": build_view,
                        "write_view": write_view, "build_packet": packet, "assemble_publication": publication,
                        "doctor": doctor, "compile_snapshot": compile, "inspect_build": inspect}.items():
        monkeypatch.setattr(sm, name, value)
    return calls


@pytest.mark.parametrize("compile_pdf", [False, True])
def test_orchestration_isolated_paths_contracts_and_compile_switch(project, services, compile_pdf, monkeypatch, tmp_path):
    sentinel = write(project / "research/real-study.json", {"do_not_modify": True})
    templates = {rel: (project / rel).read_bytes() for rel in sm._FIXTURES}
    monkeypatch.chdir(project.parent)
    result = sm.smoke(project, compile_pdf=compile_pdf)
    workspace = Path(result["workspace"])
    assert workspace.parent.parent == project / "build/smoke"
    assert all(call[1] == workspace for call in services if call[0] in {"execute", "analyze", "view", "packet", "publication", "doctor"})
    assert [c[0] for c in services] == ["execute", "analyze", "view", "packet", "publication"] + (
        ["doctor", "compile", "inspect"] if compile_pdf else [])
    for key in ("run", "analysis", "view", "packet", "snapshot", "experiment_config", "hypothesis"):
        assert Path(result[key]).is_relative_to(workspace) and Path(result[key]).exists()
    assert result["is_fixture"] and "pending" in result["visual_inspection"]
    assert bool(result["compiled"]) == compile_pdf
    assert bool(result["pdf"]) == compile_pdf
    assert read(Path(result["summary"])) == result
    assert read(sentinel) == {"do_not_modify": True}
    assert all((project / rel).read_bytes() == data for rel, data in templates.items())
    assert not (project / "runs").exists() and not (project / "analyses").exists()
    assert all(Path(p).is_relative_to(workspace) for p in result["files"])


def test_parameter_change_retains_prior_workspace_and_changes_scientific_identity(project, services):
    first = sm.smoke(project, compile_pdf=False)
    before = {p: p.read_bytes() for p in Path(first["smoke_root"]).rglob("*") if p.is_file()}
    path = project / "configs/examples/fixture_experiment.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["scientific"]["parameters"]["score_scale"] = 2
    path.write_text(yaml.safe_dump(cfg))
    second = sm.smoke(project, compile_pdf=False)
    assert first["workspace"] != second["workspace"]
    a = read(Path(first["run"]) / "manifest.json")["metadata"]["identity"]
    b = read(Path(second["run"]) / "manifest.json")["metadata"]["identity"]
    assert a["scientific_sha256"] != b["scientific_sha256"]
    assert all(path.read_bytes() == data for path, data in before.items())


def test_failed_execution_retains_evidence_and_stops_following_services(project, services, monkeypatch):
    def fail(project_root, config_path, *, run_id=None, freeze_path=None):
        raise WorkflowError("EXECUTION", "intentional fixture failure")
    monkeypatch.setattr(sm, "execute_experiment", fail)
    with pytest.raises(WorkflowError) as err:
        sm.smoke(project, compile_pdf=False)
    assert services == []
    workspace = Path(err.value.details["smoke_workspace"])
    assert workspace.is_dir()
    assert read(workspace.parent / "failure.json")["exception"] == "WorkflowError"


def test_failed_bundle_or_inspection_is_not_success(project, services, monkeypatch):
    monkeypatch.setattr(sm, "inspect_build", lambda compiled_dir: {"checks": {"passed": False}})
    with pytest.raises(WorkflowError, match="inspection failed"):
        sm.smoke(project)
    assert list((project / "build/smoke").glob("*/failure.json"))
    assert not list((project / "build/smoke").glob("*/smoke.json"))


def test_missing_template_is_actionable_and_does_not_create_partial_study(project):
    (project / "paper/main.tex").unlink()
    with pytest.raises(WorkflowError) as err:
        sm.smoke(project, compile_pdf=False)
    assert err.value.code == "DEPENDENCY"
    assert not (project / "build").exists()


def test_build_symlink_cannot_create_smoke_files_outside_project(project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside-smoke")
    (project / "build").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkflowError, match="escapes project root"):
        sm.smoke(project, compile_pdf=False)
    assert list(outside.iterdir()) == []


def test_executor_failed_terminal_bundle_stops_analysis(project, services, monkeypatch):
    original = sm.execute_experiment
    def failed(project_root, config_path, *, run_id=None, freeze_path=None):
        bundle = original(project_root, config_path, run_id=run_id, freeze_path=freeze_path)
        manifest = read(bundle / "manifest.json")
        manifest["status"] = "failed"
        write(bundle / "manifest.json", manifest)
        return bundle
    monkeypatch.setattr(sm, "execute_experiment", failed)
    with pytest.raises(WorkflowError, match="did not succeed"):
        sm.smoke(project, compile_pdf=False)
    assert [call[0] for call in services] == ["execute"]
