"""M15 isolation tests; M01 I/O/schema and M02 resolution are explicit fakes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from threading import Barrier
from typing import Any, Mapping

import jsonschema
import pytest

from researchflow import freeze as subject
from researchflow.contracts import WorkflowError


ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src/researchflow/schemas"
EXAMPLES = ROOT / "tests/fixtures/schema_examples"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _resolved(experiment_id: str = "E-final") -> dict[str, Any]:
    value = json.loads((EXAMPLES / "resolved_experiment.json").read_text())
    value["scientific"].update(experiment_id=experiment_id, phase="final")
    value["scientific"]["protocol"]["split"] = "final"
    # Nonzero synthetic identities supplied by the resolver fake, not real
    # provenance or an implementation of M02's identity algorithm.
    value["identity"] = {
        key: _hash(experiment_id + key)
        for key in ("scientific_sha256", "code_sha256", "data_sha256")
    }
    return value


@pytest.fixture
def ports(monkeypatch):
    configs: dict[Path, dict[str, Any]] = {}
    calls: list[tuple[Path, Path]] = []

    def validate_document(kind: str, value: Mapping[str, Any]) -> None:
        schema = json.loads((SCHEMAS / (kind + ".json")).read_text())
        try:
            jsonschema.Draft202012Validator(
                schema, format_checker=jsonschema.FormatChecker(),
            ).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError("CONTRACT", exc.message) from exc

    def load_document(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise WorkflowError("CONTRACT", "Cannot read test document.") from exc

    def atomic_write_json(path: Path, value: Any) -> None:
        # M15 supplies a private path. Atomicity of M01's writer is not under
        # test here; final-name publication must be atomic independently.
        path.write_text(json.dumps(value) + "\n")

    def resolve_experiment(project_root: Path, config_path: Path) -> dict[str, Any]:
        calls.append((project_root, config_path))
        return deepcopy(configs[config_path])

    monkeypatch.setattr(subject, "validate_document", validate_document)
    monkeypatch.setattr(subject, "load_document", load_document)
    monkeypatch.setattr(subject, "atomic_write_json", atomic_write_json)
    monkeypatch.setattr(subject, "resolve_experiment", resolve_experiment)
    return configs, calls


def _register(ports, root: Path, name: str = "final.json", experiment_id: str = "E-final"):
    resolved = _resolved(experiment_id)
    ports[0][root / name] = resolved
    return resolved


def test_matching_freeze_actual_time_paths_and_no_mutation(tmp_path, monkeypatch, ports):
    project = tmp_path / "project"
    project.mkdir()
    resolved = _register(ports, project)
    before_resolved = deepcopy(resolved)
    # Relative paths must be rooted in the supplied project, not the cwd.
    monkeypatch.chdir(tmp_path)
    before = datetime.now(timezone.utc)
    result = subject.freeze_study(project, [Path("final.json")], Path("design/freeze.json"))
    after = datetime.now(timezone.utc)
    created_at = datetime.fromisoformat(result["created_at"])
    assert created_at.tzinfo is not None and created_at.utcoffset() is not None
    assert before <= created_at <= after
    assert ports[1] == [(project, project / "final.json")]
    path = project / "design/freeze.json"
    assert json.loads(path.read_text()) == result
    assert result["configs"] == [{"experiment_id": "E-final", "identity": resolved["identity"]}]
    assert subject.check_final_gate(path, resolved) == result
    assert resolved == before_resolved
    assert sorted(p.name for p in path.parent.iterdir()) == ["freeze.json"]


@pytest.mark.parametrize("key", ["scientific_sha256", "code_sha256", "data_sha256"])
def test_changed_identity_rejected(tmp_path, ports, key):
    resolved = _register(ports, tmp_path)
    path = tmp_path / "freeze.json"
    subject.freeze_study(tmp_path, [Path("final.json")], path)
    changed = deepcopy(resolved)
    changed["identity"][key] = _hash("changed " + key)
    with pytest.raises(WorkflowError) as error:
        subject.check_final_gate(path, changed)
    assert error.value.code == "IDENTITY"
    assert error.value.details["changed"] == [key]


def test_declared_model_manifest_identity_is_preserved(tmp_path, ports):
    resolved = _register(ports, tmp_path)
    resolved["runtime"]["data_refs"] = [{
        "id": "model-weights", "manifest": "manifests/model.json", "local_path": None,
    }]
    resolved["scientific"]["parameters"]["model_config_manifest"] = "manifests/config.json"
    # Model input manifests are fingerprinted by M02. M15 must preserve and
    # compare the resulting identity without scoring or reading model weights.
    resolved["identity"]["data_sha256"] = _hash("declared weights manifest v1")
    path = tmp_path / "freeze.json"
    result = subject.freeze_study(tmp_path, [Path("final.json")], path)
    assert result["configs"][0]["identity"] == resolved["identity"]
    changed = deepcopy(resolved)
    changed["identity"]["data_sha256"] = _hash("declared weights manifest v2")
    with pytest.raises(WorkflowError, match="differs"):
        subject.check_final_gate(path, changed)


@pytest.mark.parametrize("phase", ["development", "fixture"])
def test_nonfinal_cannot_be_frozen_or_gated(tmp_path, ports, phase):
    resolved = _register(ports, tmp_path)
    resolved["scientific"]["phase"] = phase
    before = deepcopy(resolved)
    path = tmp_path / "freeze.json"
    with pytest.raises(WorkflowError) as error:
        subject.freeze_study(tmp_path, [Path("final.json")], path)
    assert error.value.code == "CONTRACT"
    assert not path.exists()
    # No freeze read is necessary to reject non-final input.
    with pytest.raises(WorkflowError, match="final phase"):
        subject.check_final_gate(path, resolved)
    assert resolved == before


def test_multiple_configs_and_unknown_experiment(tmp_path, ports):
    a = _register(ports, tmp_path, "a.json", "E-a")
    b = _register(ports, tmp_path, "b.json", "E-b")
    path = tmp_path / "freeze.json"
    result = subject.freeze_study(tmp_path, [Path("a.json"), tmp_path / "b.json"], path)
    assert [entry["experiment_id"] for entry in result["configs"]] == ["E-a", "E-b"]
    for value in (a, b):
        assert subject.check_final_gate(path, value) == result
    with pytest.raises(WorkflowError) as error:
        subject.check_final_gate(path, _resolved("E-unknown"))
    assert error.value.code == "IDENTITY"


def test_no_configs_or_duplicate_experiment_ids(tmp_path, ports):
    _register(ports, tmp_path, "a.json")
    _register(ports, tmp_path, "b.json")
    path = tmp_path / "freeze.json"
    for paths in ([], [Path("a.json"), Path("b.json")]):
        with pytest.raises(WorkflowError) as error:
            subject.freeze_study(tmp_path, paths, path)
        assert error.value.code == "CONTRACT"
        assert not path.exists()


def test_mixed_phases_never_publish_partial_freeze(tmp_path, ports):
    _register(ports, tmp_path, "a.json", "E-a")
    b = _register(ports, tmp_path, "b.json", "E-b")
    b["scientific"]["phase"] = "development"
    with pytest.raises(WorkflowError):
        subject.freeze_study(tmp_path, [Path("a.json"), Path("b.json")], Path("freeze.json"))
    assert not (tmp_path / "freeze.json").exists()


def test_second_freeze_cannot_overwrite_first(tmp_path, ports):
    _register(ports, tmp_path)
    path = tmp_path / "freeze.json"
    subject.freeze_study(tmp_path, [Path("final.json")], path)
    original = path.read_bytes()
    calls = len(ports[1])
    with pytest.raises(WorkflowError) as error:
        subject.freeze_study(tmp_path, [Path("final.json")], path)
    assert error.value.code == "IDENTITY"
    assert path.read_bytes() == original
    assert len(ports[1]) == calls


@pytest.mark.parametrize("existing", ["file", "directory", "symlink", "dangling"])
def test_existing_output_is_always_a_collision(tmp_path, ports, existing):
    _register(ports, tmp_path)
    path = tmp_path / "freeze.json"
    target = tmp_path / "target"
    if existing == "file":
        path.write_text("original")
    elif existing == "directory":
        path.mkdir()
    else:
        if existing == "symlink":
            target.write_text("original")
        path.symlink_to(target)
    with pytest.raises(WorkflowError) as error:
        subject.freeze_study(tmp_path, [Path("final.json")], path)
    assert error.value.code == "IDENTITY"
    if existing == "file":
        assert path.read_text() == "original"
    elif existing == "symlink":
        assert target.read_text() == "original"
    elif existing == "dangling":
        assert path.is_symlink() and not target.exists()


def test_concurrent_create_has_exactly_one_winner(tmp_path, monkeypatch, ports):
    _register(ports, tmp_path)
    barrier = Barrier(2)
    original = subject.atomic_write_json

    def atomic_write_json(path: Path, value: Any) -> None:
        original(path, value)
        barrier.wait(timeout=10)

    monkeypatch.setattr(subject, "atomic_write_json", atomic_write_json)
    path = tmp_path / "freeze.json"

    def attempt():
        try:
            return subject.freeze_study(tmp_path, [Path("final.json")], path)
        except WorkflowError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    winners = [result for result in results if isinstance(result, dict)]
    losers = [result for result in results if isinstance(result, WorkflowError)]
    assert len(winners) == len(losers) == 1
    assert losers[0].code == "IDENTITY"
    assert json.loads(path.read_text()) == winners[0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["freeze.json"]


def test_writer_failure_leaves_no_final_record(tmp_path, monkeypatch, ports):
    _register(ports, tmp_path)

    def atomic_write_json(path: Path, value: Any) -> None:
        path.write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(subject, "atomic_write_json", atomic_write_json)
    with pytest.raises(WorkflowError) as error:
        subject.freeze_study(tmp_path, [Path("final.json")], Path("freeze.json"))
    assert error.value.code == "EXECUTION"
    assert list(tmp_path.iterdir()) == []


def test_gate_has_no_execution_resolution_or_writing(tmp_path, monkeypatch, ports):
    resolved = _register(ports, tmp_path)
    path = tmp_path / "freeze.json"
    result = subject.freeze_study(tmp_path, [Path("final.json")], path)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("gate must only read and compare the supplied identities")

    monkeypatch.setattr(subject, "resolve_experiment", forbidden)
    monkeypatch.setattr(subject, "atomic_write_json", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert subject.check_final_gate(path, resolved) == result
    assert path.read_bytes() == before


@pytest.mark.parametrize("problem", ["missing_identity", "bad_digest", "unknown_field", "invalid_time", "duplicate"])
def test_gate_rejects_invalid_or_ambiguous_freezes(tmp_path, ports, problem):
    resolved = _register(ports, tmp_path)
    path = tmp_path / "freeze.json"
    result = subject.freeze_study(tmp_path, [Path("final.json")], path)
    if problem == "missing_identity":
        del result["configs"][0]["identity"]
    elif problem == "bad_digest":
        result["configs"][0]["identity"]["code_sha256"] = "not a digest"
    elif problem == "unknown_field":
        result["preregistered"] = True
    elif problem == "invalid_time":
        result["created_at"] = "2026-10-25T12:00:00"  # Missing timezone.
    else:
        result["configs"].append(deepcopy(result["configs"][0]))
    path.write_text(json.dumps(result))
    with pytest.raises(WorkflowError) as error:
        subject.check_final_gate(path, resolved)
    assert error.value.code == "CONTRACT"


def test_malformed_resolved_config_fails_before_publication(tmp_path, ports):
    resolved = _register(ports, tmp_path)
    del resolved["identity"]["data_sha256"]
    with pytest.raises(WorkflowError) as error:
        subject.freeze_study(tmp_path, [Path("final.json")], Path("freeze.json"))
    assert error.value.code == "CONTRACT"
    assert not (tmp_path / "freeze.json").exists()
