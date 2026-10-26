"""M09 isolation tests: real schemas, synthetic records, dependency API fakes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import jsonschema
import pytest
import yaml

from researchflow import bundles, documents, research, views
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "tests/fixtures/schema_examples"


def example(kind):
    return json.loads((EXAMPLES / f"{kind}.json").read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    calls = {"loads": [], "verified": []}
    index = {"schema_version": 1, "branches": [], "hypotheses": [],
             "decisions": [], "journal": [], "unresolved": []}

    def validate_document(kind: str, value):
        schema = json.loads((ROOT / f"src/researchflow/schemas/{kind}.json").read_text())
        try:
            jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError("CONTRACT", str(exc)) from exc

    def load_document(path: Path) -> dict:
        calls["loads"].append(path)
        assert path.suffix in {".json", ".yaml"}, "Prediction payload must not be decoded"
        return yaml.safe_load(path.read_text())

    def contained_path(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
            raise WorkflowError("CONTRACT", "Escaping relative path")
        return path

    def verify_bundle(bundle_dir: Path, expected_kind: str | None = None) -> dict:
        calls["verified"].append((bundle_dir, expected_kind))
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        assert manifest["bundle_kind"] == expected_kind
        # Deliberately a fake: M03 integrity algorithms are not duplicated here.
        return manifest

    def resolve_pointer(document, pointer: str):
        try:
            value = document
            if pointer == "":
                return value
            if not pointer.startswith("/"):
                raise KeyError(pointer)
            for part in pointer[1:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                value = value[int(key)] if isinstance(value, list) else value[key]
            return value
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            raise WorkflowError("CONTRACT", "Missing JSON pointer") from exc

    def atomic_write_json(path: Path, value) -> None:
        dump(path, value)

    def load_research(project_root: Path) -> dict:
        assert project_root == tmp_path
        return deepcopy(index)

    def bundle_fingerprint(manifest) -> str:
        return hashlib.sha256(canonical(manifest)).hexdigest()

    monkeypatch.setattr(documents, "load_document", load_document)
    monkeypatch.setattr(documents, "validate_document", validate_document)
    monkeypatch.setattr(documents, "canonical_bytes", canonical)
    monkeypatch.setattr(documents, "sha256_file", sha)
    monkeypatch.setattr(documents, "contained_path", contained_path)
    monkeypatch.setattr(documents, "resolve_pointer", resolve_pointer)
    monkeypatch.setattr(documents, "atomic_write_json", atomic_write_json)
    monkeypatch.setattr(bundles, "verify_bundle", verify_bundle)
    monkeypatch.setattr(bundles, "bundle_fingerprint", bundle_fingerprint)
    monkeypatch.setattr(research, "load_research", load_research)

    def no_runner(*args, **kwargs):
        raise AssertionError("A view must not execute a runner")
    monkeypatch.setattr(subprocess, "run", no_runner)

    def record(kind, identifier, **changes):
        rec = example(kind)
        rec.update(id=identifier, **changes)
        path = dump(tmp_path / f"research/{kind}/{identifier}.json", rec)
        entry = {"record": rec, "source_path": path.relative_to(tmp_path).as_posix(),
                 "source_sha256": sha(path)}
        index[{"branch": "branches", "hypothesis": "hypotheses", "decision": "decisions"}[kind]].append(entry)
        return rec

    def run(identifier, experiment_id, branch_id="B-main", status="succeeded", checks=True,
            phase="development", ranking=False):
        path = tmp_path / "runs" / identifier
        manifest = example("run_manifest")
        manifest.update(bundle_id=identifier, status=status)
        manifest["metadata"].update(experiment_id=experiment_id, branch_id=branch_id,
            hypothesis_ids=["H-main"], phase=phase,
            result_path="payload/result.json" if status == "succeeded" else None,
            process_status=status)
        if status == "succeeded":
            result = example("runner_result")
            result["checks"]["passed"] = checks
            if not ranking:
                result.update(kind="metrics", predictions=[])
            dump(path / "payload/result.json", result)
            # Ensure large data existence is irrelevant to view loading.
            if ranking:
                payload = path / "payload" / result["predictions"][0]["path"]
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(b"not a real NPZ; never parse this")
        dump(path / "manifest.json", manifest)
        return f"runs/{identifier}"

    def experiment(identifier, branch_id="B-main", phase="development"):
        doc = example("experiment_config")
        doc["scientific"].update(experiment_id=identifier, branch_id=branch_id,
                                 hypothesis_ids=["H-main"], phase=phase)
        dump(tmp_path / f"configs/{identifier}.json", doc)
        return f"configs/{identifier}.json"

    def analysis(identifier="A-main", run_ids=("R-completed",), phase="development", checks=True, status="succeeded"):
        path = tmp_path / "analyses" / identifier
        manifest = example("analysis_manifest")
        manifest.update(bundle_id=identifier + "-bundle", status=status)
        manifest["metadata"].update(analysis_id=identifier, phases=[phase])
        result = example("analysis_result")
        result.update(analysis_id=identifier, phases=[phase])
        result["checks"]["passed"] = checks
        result["inputs"] = [{"run_id": rid,
            "manifest_sha256": sha(tmp_path / f"runs/{rid}/manifest.json"),
            "phase": phase, "origin": "fresh_execution"} for rid in run_ids]
        manifest["metadata"]["input_fingerprints"] = [
            bundle_fingerprint(json.loads((tmp_path / f"runs/{rid}/manifest.json").read_text())) for rid in run_ids]
        dump(path / "analysis.json", result)
        dump(path / "manifest.json", manifest)
        return f"analyses/{identifier}"

    record("hypothesis", "H-main", branch_ids=["B-main", "B-blocked"])
    record("branch", "B-main", hypothesis_ids=["H-main"],
           planned_experiment_ids=["E-planned", "E-failed", "E-completed"])
    record("branch", "B-blocked", status="blocked", hypothesis_ids=["H-main"],
           planned_experiment_ids=["E-blocked"])
    config = example("view_config")
    config.update(view_id="V-test", branch_ids=[], hypothesis_ids=[], phases=[],
                  experiment_paths=[], run_paths=[], analysis_paths=[])

    def build(**changes):
        config.update(changes)
        dump(tmp_path / "configs/view.json", config)
        return views.build_view(tmp_path, Path("configs/view.json"))

    return {"root": tmp_path, "index": index, "config": config, "calls": calls,
            "record": record, "run": run, "experiment": experiment, "analysis": analysis, "build": build}


def test_planned_failed_completed_blocked_and_missing_run(sandbox):
    s = sandbox
    state = s["build"](run_paths=[s["run"]("R-failed", "E-failed", status="failed"),
        s["run"]("R-completed", "E-completed"), "runs/R-missing"],
        experiment_paths=[s["experiment"]("E-staged")])
    rows = {r["id"]: r for r in state["experiments"]}
    assert {r["status"] for r in rows.values()} >= {"planned", "failed", "completed", "blocked", "staged", "unresolved"}
    assert rows["E-planned"]["evidence"] is None
    assert rows["runs/R-missing"]["evidence"] is None
    assert any(x["id"] == "E-planned" for x in state["unresolved"])
    assert any(x["id"] == "E-staged" for x in state["unresolved"])


def test_superseded_history_and_independent_current_decisions(sandbox):
    s = sandbox
    for identifier, supersedes in (("D-old", []), ("D-new", ["D-old"]), ("D-independent", [])):
        s["record"]("decision", identifier, branch_ids=["B-main"], hypothesis_ids=["H-main"], supersedes=supersedes)
    state = s["build"]()
    ds = {r["record"]["id"]: r for r in state["decisions"]}
    assert not ds["D-old"]["is_current"]
    assert ds["D-old"]["superseded_by"] == ["D-new"]
    assert {k for k, v in ds.items() if v["is_current"]} == {"D-new", "D-independent"}


def test_excluded_successor_does_not_resurrect_old_decision(sandbox):
    s = sandbox
    s["record"]("decision", "D-old", branch_ids=["B-main"], supersedes=[])
    s["record"]("decision", "D-new", branch_ids=["B-blocked"], supersedes=["D-old"])
    state = s["build"](branch_ids=["B-main"])
    assert state["decisions"][0]["is_current"] is False
    assert any(x["id"] == "D-new" for x in state["exclusions"])


def test_branch_filter_exclusions_and_source_hashes(sandbox):
    s = sandbox
    failed = s["run"]("R-failed", "E-failed", status="failed")
    other = s["run"]("R-other", "E-other", branch_id="B-blocked")
    state = s["build"](branch_ids=["B-main"], run_paths=[failed, other])
    assert [x["record"]["id"] for x in state["branches"]] == ["B-main"]
    assert {x["id"] for x in state["exclusions"]} >= {"B-blocked", "R-other"}
    for source in state["sources"]:
        assert source["source_sha256"] == sha(s["root"] / source["source_path"])
    assert not any(x["id"] == "R-other" for x in state["experiments"])
    assert any(x["id"] == "R-failed" for x in state["experiments"])


def test_checks_invalidate_but_do_not_remove_work(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed", checks=False)
    analysis = s["analysis"](checks=False)
    state = s["build"](run_paths=[run], analysis_paths=[analysis])
    assert next(x for x in state["experiments"] if x["id"] == "R-completed")["status"] == "invalidated"
    assert state["analyses"][0]["status"] == "invalidated"
    assert state["analyses"][0]["evidence"]["limitations"] == ["Synthetic example only."]


def test_no_prediction_decode_or_runner_execution_and_root_resolution(sandbox, monkeypatch, tmp_path):
    s = sandbox
    run = s["run"]("R-completed", "E-completed", ranking=True)
    config = s["experiment"]("E-staged", phase="final")
    monkeypatch.chdir(tmp_path.parent)
    state = s["build"](run_paths=[run], experiment_paths=[config])
    assert all(p.is_relative_to(s["root"]) for p in s["calls"]["loads"])
    assert not any(p.suffix == ".npz" for p in s["calls"]["loads"])
    assert any(x.get("phase") == "final" and x["kind"] == "staged" for x in state["experiments"])
    assert s["calls"]["verified"] == [(s["root"] / run, "run")]


def test_analysis_scope_excludes_mixed_branches_without_cherry_picking(sandbox):
    s = sandbox
    paths = [s["run"]("R-completed", "E-completed"),
             s["run"]("R-other", "E-other", branch_id="B-blocked")]
    state = s["build"](branch_ids=["B-main"], run_paths=paths,
                       analysis_paths=[s["analysis"](run_ids=("R-completed", "R-other"))])
    assert state["analyses"] == []
    assert any(x["id"] == "A-main" for x in state["exclusions"])


def test_deterministic_even_when_index_and_path_order_changes(sandbox):
    s = sandbox
    paths = [s["run"]("R-failed", "E-failed", status="failed"),
             s["run"]("R-completed", "E-completed")]
    first = s["build"](run_paths=paths, analysis_paths=[s["analysis"]()])
    for key in ("branches", "hypotheses", "decisions", "journal", "unresolved"):
        s["index"][key].reverse()
    second = s["build"](run_paths=paths)
    assert canonical(first) == canonical(second)
    assert s["index"]["branches"][0]["record"].keys() == example("branch").keys()
    output = views.write_view(first, s["root"] / "output")
    before = ((output / "view.json").read_bytes(), (output / "view.md").read_bytes())
    assert views.write_view(second, output) == output
    assert before == ((output / "view.json").read_bytes(), (output / "view.md").read_bytes())
    assert json.loads(before[0]) == first
    assert "R-failed" in before[1].decode() and "SHA256" in before[1].decode()


def test_journal_legacy_untouched_and_unresolved_references(sandbox):
    s = sandbox
    header = example("journal_header")
    header.update(id="J-main", branch_ids=["B-main"], hypothesis_ids=["H-main"], run_ids=["R-unknown"])
    entries = [(header, "Full structured body", False), (None, "Original legacy body", True)]
    for i, (h, body, legacy) in enumerate(entries):
        path = s["root"] / f"research/journal/{i}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        s["index"]["journal"].append({"header": h, "body": body, "legacy": legacy,
            "source_path": path.relative_to(s["root"]).as_posix(), "source_sha256": sha(path)})
    original = deepcopy(s["index"])
    state = s["build"](include_legacy_journal=False)
    assert len(state["journal"]) == 1
    assert any(x["id"] == "R-unknown" for x in state["unresolved"])
    assert any("legacy journal disabled" in x["reasons"] for x in state["exclusions"])
    assert s["index"] == original
    assert len(s["build"](include_legacy_journal=True)["journal"]) == 2
    assert (s["root"] / "research/journal/1.md").read_text() == "Original legacy body"


def test_bad_config_and_escaping_paths_raise_shared_error(sandbox):
    with pytest.raises(WorkflowError) as exc:
        sandbox["build"](phases=["pretend-science"])
    assert exc.value.code == "CONTRACT"
    with pytest.raises(WorkflowError):
        sandbox["build"](phases=[], run_paths=["../outside"])


def test_missing_summaries_and_inputs_are_not_zero_scores(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed")
    path = s["root"] / run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["result_path"] = None
    dump(path, manifest)
    state = s["build"](run_paths=[run], analysis_paths=["analyses/missing"])
    row = next(x for x in state["experiments"] if x["id"] == "R-completed")
    assert row["status"] == "invalidated" and row["evidence"] is None
    assert state["analyses"][0]["evidence"] is None


def test_phase_filter_keeps_unknown_plan_and_lists_final_exclusions(sandbox):
    s = sandbox
    state = s["build"](phases=["development"],
        experiment_paths=[s["experiment"]("E-final", phase="final")],
        run_paths=[s["run"]("R-final", "E-final", phase="final")])
    assert any(x["id"] == "E-planned" and x["phase"] is None for x in state["experiments"])
    assert {x["id"] for x in state["exclusions"]} >= {"R-final", "E-final"}


def test_bad_evidence_pointer_and_supersession_cycle_visible(sandbox):
    s = sandbox
    s["record"]("decision", "D-a", branch_ids=["B-main"], supersedes=["D-b"],
        evidence=["research/branch/B-main.json#/missing", "unknown/huge.npz"])
    s["record"]("decision", "D-b", branch_ids=["B-main"], supersedes=["D-a"])
    state = s["build"]()
    assert all(not x["is_current"] for x in state["decisions"])
    assert any("Cyclic" in x["reason"] for x in state["unresolved"])
    assert any("JSON pointer" in x["reason"] for x in state["unresolved"])
    assert any(x["id"] == "unknown/huge.npz" for x in state["unresolved"])


def test_mismatched_input_hash_invalidates_analysis(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed")
    analysis = s["analysis"]()
    path = s["root"] / analysis / "analysis.json"
    result = json.loads(path.read_text())
    result["inputs"][0]["manifest_sha256"] = "f" * 64
    dump(path, result)
    state = s["build"](run_paths=[run], analysis_paths=[analysis])
    assert state["analyses"][0]["status"] == "invalidated"
    assert any("manifest hash" in x["reason"] for x in state["unresolved"])


def test_duplicate_canonical_identity_rejected(sandbox):
    s = sandbox
    s["index"]["branches"].append(deepcopy(s["index"]["branches"][0]))
    with pytest.raises(WorkflowError) as exc:
        s["build"]()
    assert exc.value.code == "IDENTITY"


def test_markdown_bundle_links_point_to_actual_manifest(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed")
    state = s["build"](run_paths=[run])
    output = views.write_view(state, s["root"] / "unrelated/deep/output")
    text = (output / "view.md").read_text()
    assert f"[source](<{(s['root'] / run / 'manifest.json').as_uri()}>)" in text
    assert "[source](<runs/" not in text


def test_analysis_cannot_hide_final_input_under_development_label(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed", phase="final")
    analysis = s["analysis"](phase="final")
    for filename in ("manifest.json", "analysis.json"):
        path = s["root"] / analysis / filename
        doc = json.loads(path.read_text())
        target = doc["metadata"] if filename == "manifest.json" else doc
        target["phases"] = ["development"]
        dump(path, doc)
    with pytest.raises(WorkflowError) as exc:
        s["build"](phases=["development"], run_paths=[run], analysis_paths=[analysis])
    assert exc.value.code == "IDENTITY"


def test_validation_failures_are_not_swallowed_and_writer_checks_schema(sandbox):
    s = sandbox
    staged = s["experiment"]("E-bad")
    path = s["root"] / staged
    doc = json.loads(path.read_text())
    doc["scientific"]["phase"] = "unknown"
    dump(path, doc)
    with pytest.raises(WorkflowError):
        s["build"](experiment_paths=[staged])
    with pytest.raises(WorkflowError):
        views.write_view({"schema_version": 1}, s["root"] / "bad-output")
    assert not (s["root"] / "bad-output").exists()


def test_canonical_manifest_fingerprint_is_also_recognized(sandbox):
    s = sandbox
    run = s["run"]("R-completed", "E-completed")
    analysis = s["analysis"]()
    path = s["root"] / analysis / "analysis.json"
    doc = json.loads(path.read_text())
    manifest = json.loads((s["root"] / run / "manifest.json").read_text())
    doc["inputs"][0]["manifest_sha256"] = bundles.bundle_fingerprint(manifest)
    dump(path, doc)
    state = s["build"](run_paths=[run], analysis_paths=[analysis])
    assert state["analyses"][0]["status"] == "completed"
    assert not any("Input manifest hash" in x["reason"] for x in state["unresolved"])
