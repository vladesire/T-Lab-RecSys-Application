"""M10 isolated tests: M01/M03 are signature-compatible test doubles only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import jsonschema
import pytest
import yaml

from researchflow import context
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src/researchflow/schemas"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(name):
    return read(ROOT / "tests/fixtures/schema_examples" / (name + ".json"))


@pytest.fixture(autouse=True)
def dependencies(monkeypatch):
    """Small local fake store, validating actual schemas at dependency boundaries."""
    def canonical_bytes(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

    def load_document(path):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def validate_document(kind, value):
        validator = jsonschema.Draft202012Validator(read(SCHEMAS / (kind + ".json")),
                                                   format_checker=jsonschema.FormatChecker())
        errors = list(validator.iter_errors(value))
        if errors:
            raise WorkflowError("CONTRACT", errors[0].message)

    def contained_path(root, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise WorkflowError("CONTRACT", "Unsafe relative path")
        result = (Path(root) / path).resolve()
        if not result.is_relative_to(Path(root).resolve()):
            raise WorkflowError("CONTRACT", "Escaping path")
        return result

    def begin_bundle(output_root, bundle_id):
        output_root.mkdir(parents=True, exist_ok=True)
        if (output_root / bundle_id).exists():
            raise WorkflowError("IDENTITY", "Existing terminal bundle")
        stage = output_root / (".stage-" + bundle_id)
        stage.mkdir()
        return stage

    def commit_bundle(stage_dir, final_dir, manifest):
        if final_dir.exists():
            raise WorkflowError("IDENTITY", "Existing terminal bundle")
        dump(stage_dir / "manifest.json", manifest)
        stage_dir.rename(final_dir)
        return final_dir

    def verify_bundle(bundle_dir, expected_kind=None):
        manifest = read(bundle_dir / "manifest.json")
        if expected_kind and manifest["bundle_kind"] != expected_kind:
            raise WorkflowError("CONTRACT", "Wrong bundle kind")
        for entry in manifest["files"]:
            path = bundle_dir / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
                raise WorkflowError("IDENTITY", "Changed bundle")
        return manifest

    def bundle_fingerprint(manifest):
        return hashlib.sha256(canonical_bytes(manifest)).hexdigest()

    for name, function in locals().copy().items():
        if name in {"canonical_bytes", "load_document", "sha256_file", "validate_document", "contained_path"}:
            monkeypatch.setattr(context.documents, name, function)
        if name in {"begin_bundle", "commit_bundle", "verify_bundle", "bundle_fingerprint"}:
            monkeypatch.setattr(context.bundles, name, function)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copytree(ROOT / "interfaces", root / "interfaces")
    for kind in ["hypothesis", "decision", "analysis_result", "analysis_config", "state_view", "run_manifest"]:
        dump(root / "research" / (kind + ".json"), fixture(kind))
    dump(root / "config/science.json", {"phase": "development", "limitations": ["small sample"]})
    (root / "research/sample.csv").write_bytes(b"query,rank\n1,2\n")
    # This must never be discovered or included.
    dump(root / "research/unselected-secret.json", {"secret": "DO NOT UPLOAD"})
    return root


def source(path, role, mandatory=False):
    return {"path": path, "role": role, "mandatory": mandatory, "reason": "explicit test selection"}


def config(project, profile="runner_author", sources=None, **changes):
    if sources is None:
        sources = [source("interfaces/runner/RUNNER_PORT.md", "port"),
                   source("config/science.json", "config")]
    value = {"schema_version": 1, "packet_id": "C-test", "profile": profile,
             "task": "Assess the selected scientific intent: улучшение.", "max_bytes": 100000,
             "sources": sources} | changes
    dump(project / "context.json", value)
    return project / "context.json"


def build(project, **kwargs):
    return context.build_packet(project, config(project, **kwargs), Path("packets"))


def test_explicit_inventory_and_complete_hashes(project, monkeypatch, tmp_path):
    sources = [source("interfaces/runner/RUNNER_PORT.md", "port"),
               source("interfaces/runner/runner_result.schema.json", "schema"),
               source("interfaces/runner/examples/runner_result.json", "example"),
               source("config/science.json", "config"), source("research/hypothesis.json", "hypothesis")]
    config(project, sources=sources)
    monkeypatch.chdir(tmp_path)  # Relative project inputs must not depend on CWD.
    packet = context.build_packet(project, Path("context.json"), Path("packets"))
    manifest = read(packet / "manifest.json")
    selection = read(packet / "context_manifest.json")
    expected = {s["path"] for s in sources}
    assert {s["path"] for s in selection["sources"]} == expected
    assert {f["path"] for f in manifest["files"]} == {
        "prompt.md", "evidence.json", "context_manifest.json", "response_template.json"
    } | {"sources/" + p for p in expected}
    for entry in selection["sources"]:
        raw = (project / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["size_bytes"] == len(raw)
        assert (packet / "sources" / entry["path"]).read_bytes() == raw
    evidence = read(packet / "evidence.json")
    hypothesis = next(s for s in evidence["sources"] if s["role"] == "hypothesis")
    assert json.loads(hypothesis["content"]) == fixture("hypothesis")
    assert "DO NOT UPLOAD" not in (packet / "evidence.json").read_text()
    count = len((packet / "prompt.md").read_bytes()) + len((packet / "evidence.json").read_bytes())
    assert count == manifest["metadata"]["included_bytes"] == selection["included_bytes"]


@pytest.mark.parametrize("profile,sources", [
    ("runner_author", []),
    ("runner_author", [source("config/science.json", "config")]),
    ("runner_author", [source("interfaces/runner/RUNNER_PORT.md", "port")]),
    ("analyzer_author", [source("interfaces/analyzer/ANALYZER_PORT.md", "port")]),
    ("analyzer_author", [source("interfaces/analyzer/ANALYZER_PORT.md", "port"), source("research/run_manifest.json", "run_manifest")]),
    ("results_reviewer", [source("research/hypothesis.json", "hypothesis")]),
    ("state_reviewer", [source("research/decision.json", "decision")]),
])
def test_mandatory_profile_sources(project, profile, sources):
    with pytest.raises(WorkflowError) as error:
        build(project, profile=profile, sources=sources)
    assert error.value.code == "CONTRACT"
    assert not (project / "packets").exists()


@pytest.mark.parametrize("path,role", [
    ("interfaces/analyzer/ANALYZER_PORT.md", "port"),
    ("interfaces/runner/RUNNER_PORT.md", "config"),
    ("config/science.json", "analysis"),
    ("../secret.md", "config"),
    ("config/science.json", "port"),
])
def test_role_allowlist_and_path_escape(project, path, role):
    sources = [source("interfaces/runner/RUNNER_PORT.md", "port"),
               source("config/science.json", "config"), source(path, role)]
    with pytest.raises(WorkflowError):
        build(project, sources=sources)


def test_workflow_cannot_be_disguised_as_evidence(project):
    dump(project / "src/secrets.json", {"workflow": "private"})
    (project / "research/alias.json").symlink_to(project / "src/secrets.json")
    for path in ["src/secrets.json", "research/alias.json"]:
        with pytest.raises(WorkflowError):
            build(project, sources=[source("research/analysis_result.json", "analysis"), source(path, "config")],
                  profile="results_reviewer")


def test_mandatory_multibyte_context_exact_budget(project):
    packet = build(project)
    size = read(packet / "manifest.json")["metadata"]["included_bytes"]
    other = project / "packets" / "C-test"
    shutil.rmtree(other)
    build(project, max_bytes=size)
    shutil.rmtree(other)
    with pytest.raises(WorkflowError) as err:
        build(project, max_bytes=size - 1)
    assert err.value.details["required_bytes"] == size
    assert err.value.details["sources"]
    assert not other.exists()


def test_optional_whole_source_exclusion_is_auditable(project):
    sample = project / "interfaces/runner/examples/large.json"
    dump(sample, {"table": ["large row" * 1000] * 20})
    sources = [source("interfaces/runner/RUNNER_PORT.md", "port"), source("config/science.json", "config"),
               source("interfaces/runner/examples/large.json", "example")]
    packet = build(project, sources=sources, max_bytes=10000)
    excluded = read(packet / "manifest.json")["metadata"]["excluded"]
    assert len(excluded) == 1
    assert excluded[0]["path"].endswith("large.json")
    assert excluded[0]["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert excluded[0]["size_bytes"] == sample.stat().st_size
    assert excluded[0]["reason"] == "explicit test selection"
    assert "max_bytes" in excluded[0]["selection_reason"]
    assert not (packet / "sources/interfaces/runner/examples/large.json").exists()


def test_reviewer_retains_failed_checks_phase_origin_limits_and_history(project):
    analysis = fixture("analysis_result")
    analysis["checks"]["passed"] = False
    analysis["checks"]["items"][0]["status"] = "failed"
    analysis["checks"]["items"][0]["detail"] = "Temporal leakage not ruled out"
    analysis["limitations"] += ["Development only", "таблица не усечена"]
    dump(project / "research/analysis_result.json", analysis)
    dump(project / "research/history.json", {"phase": "development", "failed_checks": ["failure"], "history": list(range(100))})
    packet = build(project, profile="results_reviewer", sources=[
        source("research/analysis_result.json", "analysis"), source("research/hypothesis.json", "hypothesis"),
        source("research/history.json", "history")])
    included = read(packet / "evidence.json")["sources"]
    assert json.loads(included[0]["content"]) == analysis
    assert json.loads(included[2]["content"])["history"] == list(range(100))
    assert all(entry["mandatory"] for entry in included)
    assert analysis["inputs"][0]["origin"] in included[0]["content"]
    assert "fixture" in included[0]["content"]


def test_reviewer_safety_evidence_cannot_be_marked_optional(project):
    analysis = fixture("analysis_result")
    analysis["limitations"] = ["x" * 20000]
    dump(project / "research/analysis_result.json", analysis)
    with pytest.raises(WorkflowError, match="Mandatory"):
        build(project, profile="results_reviewer", sources=[source("research/analysis_result.json", "analysis", False)], max_bytes=10000)


def test_analyzer_contract_sample_and_intent(project):
    packet = build(project, profile="analyzer_author", sources=[
        source("interfaces/analyzer/ANALYZER_PORT.md", "port"),
        source("interfaces/analyzer/run_manifest.schema.json", "schema"),
        source("research/sample.csv", "sample_payload"), source("research/analysis_config.json", "analysis_config")])
    assert len(read(packet / "evidence.json")["sources"]) == 4
    assert "selected scientific intent" in (packet / "prompt.md").read_text()


def test_analyzer_binary_sample_is_explicit_and_budgeted(project):
    (project / "research/sample.npz").write_bytes(b"\x00\xff\x80binary-sample")
    packet = build(project, profile="analyzer_author", sources=[
        source("interfaces/analyzer/ANALYZER_PORT.md", "port"), source("research/run_manifest.json", "run_manifest"),
        source("research/sample.npz", "sample_payload")])
    sample = read(packet / "evidence.json")["sources"][-1]
    import base64
    assert sample["encoding"] == "base64"
    assert base64.b64decode(sample["content"]) == (project / "research/sample.npz").read_bytes()


def test_state_view_is_preserved_with_exclusions_unresolved_and_selected_records(project):
    view = fixture("state_view")
    view["unresolved"] = [{"kind": "evidence", "id": "missing", "reason": "not supplied"}]
    view["exclusions"] = [{"id": "excluded", "reason": "fixture only"}]
    dump(project / "research/state_view.json", view)
    packet = build(project, profile="state_reviewer", sources=[source("research/state_view.json", "state"),
        source("research/hypothesis.json", "hypothesis"), source("research/decision.json", "decision")])
    evidence = read(packet / "evidence.json")
    assert json.loads(evidence["sources"][0]["content"]) == view
    assert {s["role"] for s in evidence["sources"]} == {"state", "hypothesis", "decision"}


def test_response_packet_identity_exact_bytes_unknown_metadata_and_immutability(project, tmp_path):
    packet = build(project)
    raw = b"proposal: do not run\r\n\xff\x00"
    response = tmp_path / "response.txt"
    response.write_bytes(raw)
    before = {p.relative_to(packet): p.read_bytes() for p in packet.rglob("*") if p.is_file()}
    first = context.capture_response(packet, response, {"author": "Tester", "model": "unknown-family", "settings": {"temperature": 0}}, tmp_path / "captures")
    record = read(first / "response.json")
    assert record["packet_fingerprint"] == hashlib.sha256(context.documents.canonical_bytes(read(packet / "manifest.json"))).hexdigest()
    assert record["packet_id"] == "C-test"
    assert record["model_version"] is None
    assert record["settings"] == {"temperature": 0}
    assert (first / record["raw_response_path"]).read_bytes() == raw
    response.write_bytes(b"a second response")
    second = context.capture_response(packet, response, {}, tmp_path / "captures")
    assert first != second
    assert (first / "raw_response.bin").read_bytes() == raw
    assert read(second / "response.json")["model"] is None
    assert before == {p.relative_to(packet): p.read_bytes() for p in packet.rglob("*") if p.is_file()}
    with pytest.raises(WorkflowError, match="original context packet"):
        context.capture_response(first, response, {}, tmp_path / "captures")


def test_capture_rejects_tampered_packet(project, tmp_path):
    packet = build(project)
    response = tmp_path / "response.txt"
    response.write_text("test")
    (packet / "prompt.md").write_text("tampered")
    with pytest.raises(WorkflowError) as err:
        context.capture_response(packet, response, {}, tmp_path / "captures")
    assert err.value.code == "IDENTITY"
    assert not (tmp_path / "captures").exists()


def test_packet_collision_does_not_overwrite(project):
    packet = build(project)
    original = (packet / "manifest.json").read_bytes()
    with pytest.raises(WorkflowError) as error:
        build(project)
    assert error.value.code == "IDENTITY"
    assert (packet / "manifest.json").read_bytes() == original


def test_malformed_review_evidence_rejected(project):
    dump(project / "research/analysis_result.json", {"metrics": []})
    with pytest.raises(WorkflowError):
        build(project, profile="results_reviewer", sources=[source("research/analysis_result.json", "analysis")])


def test_response_collision_cannot_overwrite_previous_capture(project, tmp_path, monkeypatch):
    from types import SimpleNamespace
    packet = build(project)
    response = tmp_path / "response.txt"
    response.write_bytes(b"first")
    monkeypatch.setattr(context, "uuid4", lambda: SimpleNamespace(hex="collision"))
    first = context.capture_response(packet, response, {}, tmp_path / "captures")
    response.write_bytes(b"second")
    with pytest.raises(WorkflowError) as err:
        context.capture_response(packet, response, {}, tmp_path / "captures")
    assert err.value.code == "IDENTITY"
    assert (first / "raw_response.bin").read_bytes() == b"first"


def test_capture_output_cannot_mutate_original_packet(project, tmp_path):
    packet = build(project)
    response = tmp_path / "response.txt"
    response.write_text("proposal")
    with pytest.raises(WorkflowError, match="outside the immutable packet"):
        context.capture_response(packet, response, {}, packet / "captures")
    assert not (packet / "captures").exists()
    capture = context.capture_response(packet, response, {}, Path("captures"))
    assert capture.parent == packet.parent / "captures"
