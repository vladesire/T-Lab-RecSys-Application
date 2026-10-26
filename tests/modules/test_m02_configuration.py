"""Isolated M02 tests: explicit M01 fakes, immutable real JSON schemas."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import jsonschema
import pytest
import yaml

from researchflow import configuration as module
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def documents_fakes(monkeypatch):
    # These stand-ins are injected into M02 only; M01's stub is untouched.
    def load_document(path: Path) -> dict:
        try:
            value = yaml.safe_load(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("document must be an object")
            return value
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise WorkflowError("CONTRACT", str(exc)) from exc

    def digest_document(value) -> str:
        try:
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
        except (ValueError, TypeError) as exc:
            raise WorkflowError("CONTRACT", str(exc)) from exc

    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate_document(kind: str, value) -> None:
        schema = json.loads((ROOT / "src/researchflow/schemas" / (kind + ".json")).read_text())
        try:
            jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)
        except jsonschema.ValidationError as exc:
            raise WorkflowError("CONTRACT", exc.message) from exc

    def contained_path(root: Path, relative: str) -> Path:
        rel = Path(relative)
        result = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not result.is_relative_to(root.resolve()):
            raise WorkflowError("CONTRACT", "unsafe path")
        return result

    def atomic_write_json(path: Path, value) -> None:
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    for name, fake in locals().copy().items():
        if name in {"load_document", "digest_document", "sha256_file", "validate_document",
                    "contained_path", "atomic_write_json"}:
            monkeypatch.setattr(module, name, fake)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "runners/fixture").mkdir(parents=True)
    (root / "data").mkdir()
    config = json.loads((ROOT / "tests/fixtures/schema_examples/experiment_config.json").read_text())
    config["scientific"]["protocol"] = "configs/protocol.yaml"
    config["scientific"]["parameters"] = {"model": {"rank": 8}, "search": {"seed": 17}}
    config["runtime"]["code_paths"].append("runners/fixture/helper.py")
    config["runtime"]["data_refs"] = [{"id": "D-fixture", "manifest": "data/manifest.json",
                                        "local_path": "data/raw-not-present"}]
    config["runtime"]["research_paths"] = ["research"]
    (root / "configs/experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "configs/protocol.yaml").write_bytes(b"# preserve source comment\r\nid: fixture\r\nquery: held-in\r\n")
    (root / "runners/fixture/runner.py").write_text("print('fixture only')\n")
    (root / "runners/fixture/helper.py").write_text("RANK = 8\n")
    (root / "data/manifest.json").write_text(json.dumps({
        "source": {"version": "synthetic-v1", "checksum": "a" * 64},
        "cohort_id": "C-fixture", "query_id": "Q-fixture",
    }))
    (root / "pyproject.toml").write_text("# synthetic dependency declaration\n")
    return root


def resolve(root, path=Path("configs/experiment.json")):
    return module.resolve_experiment(root, path)


def edit_config(root, edit):
    path = root / "configs/experiment.json"
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_relocation_absolute_inputs_and_unrelated_cwd(project, tmp_path, monkeypatch):
    first = resolve(project)
    moved = tmp_path / "relocated"
    shutil.copytree(project, moved)
    monkeypatch.chdir(tmp_path)
    # Both relative root/config and absolute root/config use project semantics.
    assert resolve(Path("project")) == first
    second = resolve(moved, moved / "configs/experiment.json")
    assert first["identity"] == second["identity"]
    assert first["scientific"] == second["scientific"]
    assert first["runtime"]["project_root"] != second["runtime"]["project_root"]
    assert first["runtime"]["data_refs"][0]["local_path"] != second["runtime"]["data_refs"][0]["local_path"]
    assert first["runtime"]["research_paths"] != second["runtime"]["research_paths"]
    edit_config(moved, lambda c: c["runtime"].update(
        code_paths=[str(moved / p) for p in c["runtime"]["code_paths"]]))
    assert resolve(moved)["identity"] == first["identity"]


@pytest.mark.parametrize("change,identity", [
    ("parameter", "scientific_sha256"), ("protocol", "scientific_sha256"),
    ("runner", "code_sha256"), ("helper", "code_sha256"),
    ("pyproject", "code_sha256"), ("lock", "code_sha256"), ("data", "data_sha256"),
])
def test_independent_identities(project, change, identity):
    before = resolve(project)["identity"]
    if change == "parameter":
        edit_config(project, lambda c: c["scientific"]["parameters"]["model"].update(rank=9))
    else:
        paths = {"protocol": "configs/protocol.yaml", "runner": "runners/fixture/runner.py",
                 "helper": "runners/fixture/helper.py", "pyproject": "pyproject.toml",
                 "lock": "uv.lock", "data": "data/manifest.json"}
        path = project / paths[change]
        if change == "protocol":
            path.write_text("id: fixture\nquery: other\n")
        elif change == "data":
            path.write_text('{"source_version":"synthetic-v2","cohort_id":"C2","query_id":"Q2"}')
        else:
            path.write_text("# changed bytes\n")
    after = resolve(project)["identity"]
    assert {key for key in before if before[key] != after[key]} == {identity}


def test_runtime_and_order_do_not_contaminate_identities(project):
    before = resolve(project)
    edit_config(project, lambda c: c["runtime"].update(
        code_paths=list(reversed(c["runtime"]["code_paths"])),
        command=["other-python", "runners/fixture/runner.py"],
        resources={"timeout_seconds": 100, "memory_limit_bytes": 5000}))
    assert resolve(project)["identity"] == before["identity"]
    assert resolve(project)["runtime"] != before["runtime"]
    assert resolve(project)["scientific"]["parameters"]["search"] == {"seed": 17}


def test_digest_includes_file_names_and_explicit_missing_dependencies(project):
    resolved = resolve(project)
    code, deps, data = module._inventory(project, resolved["runtime"])
    assert deps["uv.lock"] is None
    assert [item["path"] for item in code] == sorted(item["path"] for item in code)
    assert resolved["identity"]["code_sha256"] == module.digest_document({"files": code, "dependencies": deps})
    assert data[0]["sha256"] == hashlib.sha256((project / "data/manifest.json").read_bytes()).hexdigest()
    old = project / "runners/fixture/helper.py"
    old.rename(old.with_name("renamed.py"))
    edit_config(project, lambda c: c["runtime"].update(code_paths=[
        "runners/fixture/runner.py", "runners/fixture/renamed.py"]))
    assert resolve(project)["identity"]["code_sha256"] != resolved["identity"]["code_sha256"]


@pytest.mark.parametrize("location", ["envelope", "scientific", "runtime", "resources", "data"])
def test_unknown_envelope_keys_rejected(project, location):
    def corrupt(config):
        target = {"envelope": config, "scientific": config["scientific"],
                  "runtime": config["runtime"], "resources": config["runtime"]["resources"],
                  "data": config["runtime"]["data_refs"][0]}[location]
        target["invented"] = True
    edit_config(project, corrupt)
    with pytest.raises(WorkflowError) as err:
        resolve(project)
    assert err.value.code == "CONTRACT"


@pytest.mark.parametrize("field,value", [
    ("timeout_seconds", 0), ("timeout_seconds", -1), ("timeout_seconds", True),
    ("timeout_seconds", float("inf")), ("timeout_seconds", float("nan")),
    ("memory_limit_bytes", 0), ("memory_limit_bytes", 1.5), ("memory_limit_bytes", True),
])
def test_resource_limits_rejected(project, field, value):
    edit_config(project, lambda c: c["runtime"]["resources"].update({field: value}))
    with pytest.raises(WorkflowError):
        resolve(project)


def test_missing_parameters_and_arbitrary_runner_settings(project):
    edit_config(project, lambda c: c["scientific"].pop("parameters"))
    with pytest.raises(WorkflowError):
        resolve(project)
    edit_config(project, lambda c: c["scientific"].update(parameters={"runner_specific_unknown_to_m02": 5}))
    assert resolve(project)["scientific"]["parameters"] == {"runner_specific_unknown_to_m02": 5}


@pytest.mark.parametrize("name", ["runners/fixture/runner.py", "data/manifest.json", "configs/protocol.yaml"])
def test_missing_inputs_fail_without_launch(project, name, monkeypatch):
    (project / name).unlink()
    monkeypatch.setattr(module, "_run", lambda *args: pytest.fail("must not launch a process"))
    with pytest.raises(WorkflowError) as err:
        resolve(project)
    assert err.value.code == "CONTRACT"


@pytest.mark.parametrize("kind", ["code", "protocol", "manifest", "symlink"])
def test_unsafe_input_paths(project, tmp_path, kind):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    if kind == "code":
        edit_config(project, lambda c: c["runtime"].update(code_paths=[str(outside)]))
    elif kind == "protocol":
        edit_config(project, lambda c: c["scientific"].update(protocol="../outside.json"))
    elif kind == "manifest":
        edit_config(project, lambda c: c["runtime"]["data_refs"][0].update(manifest="../outside.json"))
    else:
        code = project / "runners/fixture/helper.py"
        code.unlink()
        code.symlink_to(outside)
    with pytest.raises(WorkflowError):
        resolve(project)


def test_duplicate_data_reference_ids_rejected(project):
    edit_config(project, lambda c: c["runtime"]["data_refs"].append(deepcopy(c["runtime"]["data_refs"][0])))
    with pytest.raises(WorkflowError):
        resolve(project)


def test_protocol_object_and_source_bytes_captured(project, tmp_path):
    resolved = resolve(project)
    assert resolved["scientific"]["protocol"] == {"id": "fixture", "query": "held-in"}
    assert resolved["source_files"] == ["configs/experiment.json", "configs/protocol.yaml"]
    output = tmp_path / "provenance"
    record = module.capture_provenance(project, resolved, output)
    assert record["identity"] == resolved["identity"]
    for name in ["configs/experiment.json", "configs/protocol.yaml", "runners/fixture/helper.py",
                 "runners/fixture/runner.py", "data/manifest.json", "pyproject.toml"]:
        assert (output / "inputs" / name).read_bytes() == (project / name).read_bytes()
    for entry in record["files"]:
        assert not Path(entry["path"]).is_absolute()
        assert (output / entry["path"]).stat().st_size == entry["size_bytes"]
        assert hashlib.sha256((output / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    env = record["environment"]
    assert env["python"]["version"] and env["platform"]["system"]
    assert env["model"]["status"] == "unknown"
    assert env["offline_reproduction"] == "unverified"
    assert env["dependency_files"]["uv.lock"] is None
    assert env["git"]["patch_status"] == "captured" or env["git"]["patch_status"] == "unavailable"
    with pytest.raises(WorkflowError):
        module.capture_provenance(project, resolved, output)


@pytest.mark.parametrize("change", ["code", "data", "config", "lock"])
def test_stale_resolution_rejected_before_output(project, tmp_path, change):
    resolved = resolve(project)
    if change == "code":
        (project / "runners/fixture/helper.py").write_text("changed\n")
    elif change == "data":
        (project / "data/manifest.json").write_text('{"changed":true}')
    elif change == "lock":
        (project / "uv.lock").write_text("version = 1\n")
    else:
        edit_config(project, lambda c: c["runtime"].update(command=["changed"]))
    output = tmp_path / "not-created"
    with pytest.raises(WorkflowError) as err:
        module.capture_provenance(project, resolved, output)
    assert err.value.code == "IDENTITY"
    assert not output.exists()


def test_absent_git_is_explicit(project, tmp_path, monkeypatch):
    def unavailable(*args):
        raise FileNotFoundError("git/uv unavailable")
    monkeypatch.setattr(module, "_run", unavailable)
    record = module.capture_provenance(project, resolve(project), tmp_path / "no-git")
    assert record["environment"]["git"]["status"] == "unavailable"
    assert record["environment"]["git"]["commit"] == "unknown"
    assert record["environment"]["git"]["dirty"] == "unknown"
    assert record["environment"]["uv"]["status"] == "unavailable"


def test_real_git_captures_staged_unstaged_and_reports_untracked(project, tmp_path):
    if not shutil.which("git"):
        pytest.skip("git unavailable")
    def git(*args):
        return subprocess.run(["git", *args], cwd=project, capture_output=True, check=True)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=M02 fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "synthetic test only")
    helper = project / "runners/fixture/helper.py"
    helper.write_text("RANK = 9\n")
    git("add", str(helper))
    (project / "runners/fixture/runner.py").write_text("print('unstaged synthetic change')\n")
    (project / "untracked.txt").write_text("fixture")
    output = tmp_path / "git-provenance"
    record = module.capture_provenance(project, resolve(project), output)
    info = record["environment"]["git"]
    assert info["dirty"] is True
    assert info["commit"] == git("rev-parse", "HEAD").stdout.decode().strip()
    assert "untracked.txt" in info["porcelain"]
    patch = (output / info["patch_path"]).read_text()
    assert "+RANK = 9" in patch and "+print('unstaged synthetic change')" in patch
    assert info["untracked_contents_captured"] is False


def test_inline_protocol_and_invalid_document(project):
    edit_config(project, lambda c: c["scientific"].update(protocol={"id": "inline"}))
    assert resolve(project)["source_files"] == ["configs/experiment.json"]
    (project / "data/manifest.json").write_text("[]")
    with pytest.raises(WorkflowError):
        resolve(project)


def test_small_input_bound_does_not_read_raw_data(project, monkeypatch):
    # Simulate a low bound rather than creating a large fixture.
    monkeypatch.setattr(module, "_MAX_INPUT_BYTES", 4)
    with pytest.raises(WorkflowError):
        resolve(project)


def test_capture_refuses_symlink_escape(project, tmp_path):
    output = tmp_path / "provenance"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "inputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkflowError):
        module.capture_provenance(project, resolve(project), output)
    assert list(outside.iterdir()) == []


def test_identity_change_during_inventory_rejected(project, tmp_path, monkeypatch):
    resolved = resolve(project)
    original = module._inventory
    calls = 0
    def changing_inventory(root, runtime):
        nonlocal calls
        calls += 1
        # First call is re-resolution, second starts capture's inventory.
        if calls == 2:
            (root / "runners/fixture/helper.py").write_text("# racing change\n")
        return original(root, runtime)
    monkeypatch.setattr(module, "_inventory", changing_inventory)
    with pytest.raises(WorkflowError) as err:
        module.capture_provenance(project, resolved, tmp_path / "capture")
    assert err.value.code == "IDENTITY"
    assert not (tmp_path / "capture").exists()
