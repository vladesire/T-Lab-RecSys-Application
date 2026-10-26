"""Experiment resolution and bounded, explicit input provenance (M02)."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

from researchflow.contracts import WorkflowError
from researchflow.documents import (
    atomic_write_json, contained_path, digest_document, load_document,
    sha256_file, validate_document,
)

# Provenance captures declared small inputs, never raw datasets.
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_DEPENDENCIES = ("pyproject.toml", "uv.lock")


def _root(project_root: Path) -> Path:
    root = project_root.resolve()
    if not root.is_dir():
        raise WorkflowError("CONTRACT", "Project root must be an existing directory", {"path": str(root)})
    return root


def _input(root: Path, value: str | Path) -> tuple[str, Path]:
    """Accept absolute in-project inputs; keep portable project-relative names."""
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError as exc:
            raise WorkflowError("CONTRACT", "Input is outside project root", {"path": str(value)}) from exc
    path = contained_path(root, candidate.as_posix())
    if not path.is_file():
        raise WorkflowError("CONTRACT", "Declared input file is missing", {"path": str(value)})
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise WorkflowError("CONTRACT", "Declared provenance input exceeds 16 MiB", {"path": str(value)})
    return path.relative_to(root).as_posix(), path


def _limits(runtime: Mapping[str, Any]) -> None:
    # JSON Schema's numeric comparisons alone do not reliably reject NaN/Inf.
    limits = runtime["resources"]
    timeout = limits["timeout_seconds"]
    memory = limits["memory_limit_bytes"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or timeout <= 0 or (isinstance(timeout, float) and not math.isfinite(timeout))):
        raise WorkflowError("CONTRACT", "Timeout must be a finite positive number")
    if memory is not None and (type(memory) is not int or memory < 1):
        raise WorkflowError("CONTRACT", "Memory limit must be a positive integer or null")


def _inventory(root: Path, runtime: Mapping[str, Any]) -> tuple[list, dict, list]:
    paths = sorted({_input(root, name)[0] for name in runtime["code_paths"]})
    code = [{"path": name, "sha256": sha256_file(root / name)} for name in paths]
    dependencies = {}
    for name in _DEPENDENCIES:
        path = root / name
        if path.exists() or path.is_symlink():
            _, path = _input(root, name)
            dependencies[name] = sha256_file(path)
        else:
            dependencies[name] = None
    data = []
    ids = set()
    for ref in runtime["data_refs"]:
        if ref["id"] in ids:
            raise WorkflowError("CONTRACT", "Duplicate data reference id", {"id": ref["id"]})
        ids.add(ref["id"])
        name, path = _input(root, ref["manifest"])
        # A manifest is an object supplied by its data producer. No common
        # schema exists for source versions/checksums/cohort/query semantics.
        load_document(path)
        data.append({"id": ref["id"], "path": name, "sha256": sha256_file(path)})
    return code, dependencies, sorted(data, key=lambda item: (item["id"], item["path"]))


def _identity(scientific: Mapping[str, Any], code: list, dependencies: dict, data: list) -> dict:
    return {
        "scientific_sha256": digest_document(scientific),
        "code_sha256": digest_document({"files": code, "dependencies": dependencies}),
        "data_sha256": digest_document(data),
    }


def resolve_experiment(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = _root(project_root)
    config_name, path = _input(root, config_path)
    config = load_document(path)
    validate_document("experiment_config", config)
    scientific = deepcopy(config["scientific"])
    runtime = deepcopy(config["runtime"])
    _limits(runtime)
    sources = [config_name]
    if isinstance(scientific["protocol"], str):
        name, path = _input(root, scientific["protocol"])
        scientific["protocol"] = load_document(path)
        if name not in sources:
            sources.append(name)
    # M01 supplies portable JSON canonicalization, including finite values.
    digest_document(scientific)
    runtime["project_root"] = str(root)
    runtime["code_paths"] = sorted({_input(root, name)[0] for name in runtime["code_paths"]})
    for ref in runtime["data_refs"]:
        ref["manifest"] = _input(root, ref["manifest"])[0]
        if ref["local_path"] is not None:
            # Local data may live outside the project and need not be present
            # on the machine assembling provenance. It is never scanned.
            local = Path(ref["local_path"])
            ref["local_path"] = str((root / local).resolve())
    runtime["research_paths"] = [str((root / Path(p)).resolve()) for p in runtime["research_paths"]]
    code, dependencies, data = _inventory(root, runtime)
    result = {
        "schema_version": 1, "scientific": scientific, "runtime": runtime,
        "identity": _identity(scientific, code, dependencies, data), "source_files": sources,
    }
    validate_document("resolved_experiment", result)
    return result


def _run(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=root, capture_output=True, check=False, timeout=10)


def _git(root: Path) -> tuple[dict, bytes | None]:
    """Observe git without equating HEAD with the actual working tree."""
    info: dict[str, Any] = {
        "status": "unavailable", "commit": "unknown", "dirty": "unknown",
        "patch_status": "unavailable", "untracked_contents_captured": False,
    }
    try:
        probe = _run(root, ["git", "rev-parse", "--show-toplevel"])
        if probe.returncode:
            info["reason"] = "not a git work tree"
            return info, None
        info["repository_root"] = probe.stdout.decode(errors="replace").strip()
        head = _run(root, ["git", "rev-parse", "--verify", "HEAD"])
        if head.returncode == 0:
            info["commit"] = head.stdout.decode().strip()
        status = _run(root, ["git", "status", "--porcelain=v1", "--untracked-files=all"])
        if status.returncode:
            info["reason"] = "git status failed"
            return info, None
        info.update(status="observed", dirty=bool(status.stdout),
                    porcelain=status.stdout.decode(errors="replace"))
        if head.returncode:
            info["reason"] = "HEAD unavailable (possibly an unborn repository)"
            return info, None
        patch = _run(root, ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"])
        if patch.returncode:
            info["reason"] = "git diff failed"
            return info, None
        info["patch_status"] = "captured"
        info["patch_path"] = "git.patch"
        info["patch_scope"] = "tracked staged and unstaged changes relative to HEAD"
        return info, patch.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        info["reason"] = type(exc).__name__
        return info, None


def _environment(root: Path) -> tuple[dict, bytes | None]:
    versions = {}
    for name in ("tlab-researchflow", "numpy", "PyYAML", "jsonschema", "matplotlib", "psutil"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    uv = {"status": "unavailable", "version": "unknown"}
    try:
        result = _run(root, ["uv", "--version"])
        if result.returncode == 0:
            uv = {"status": "observed", "version": result.stdout.decode(errors="replace").strip()}
    except (OSError, subprocess.SubprocessError):
        pass
    git, patch = _git(root)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable,
                   "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "description": platform.platform()},
        "packages": versions, "uv": uv, "git": git,
        "model": {"status": "unknown", "reason": "not observed by configuration module"},
        "scope": "capture process environment; runner command is recorded, not executed or inspected",
        "offline_reproduction": "unverified",
    }, patch


def capture_provenance(project_root: Path, resolved: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    root = _root(project_root)
    validate_document("resolved_experiment", resolved)
    _limits(resolved["runtime"])
    if resolved["runtime"]["project_root"] != str(root):
        raise WorkflowError("IDENTITY", "Resolved project root does not match capture root")
    sources = resolved["source_files"]
    if not sources:
        raise WorkflowError("CONTRACT", "Resolved experiment must identify its source configuration")
    # Detect changed code, manifests, dependency metadata, scientific config,
    # protocol, or runtime settings between resolution and capture.
    current = resolve_experiment(root, Path(sources[0]))
    if current != resolved:
        raise WorkflowError("IDENTITY", "Inputs changed since experiment resolution")
    code, dependencies, data = _inventory(root, resolved["runtime"])
    if _identity(resolved["scientific"], code, dependencies, data) != resolved["identity"]:
        raise WorkflowError("IDENTITY", "Inputs changed while preparing provenance capture")
    expected = {item["path"]: item["sha256"] for item in code + data}
    expected.update({name: value for name, value in dependencies.items() if value is not None})
    for name in sources:
        name, path = _input(root, name)
        expected[name] = sha256_file(path)
    environment, patch = _environment(root)
    environment["dependency_files"] = dependencies
    environment["identity_inputs"] = {"code": code, "data": data}
    environment["source_byte_scope"] = "bytes observed at capture; resolution-time source byte hashes are not in the v1 schema"
    output = output_dir.resolve()
    destinations = ["inputs/" + name for name in sorted(expected)] + ["resolved.json"]
    if patch is not None:
        destinations.append("git.patch")
    # Never overwrite another capture or a project input, including symlink aliases.
    targets = [contained_path(output, name) for name in destinations]
    input_paths = {root / name for name in expected}
    if any(path.exists() or path in input_paths for path in targets):
        raise WorkflowError("CONTRACT", "Provenance destination is already occupied or aliases an input")
    files = []
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name in sorted(expected):
            _, source = _input(root, name)
            target = contained_path(output, "inputs/" + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_file(target) != expected[name]:
                raise WorkflowError("IDENTITY", "Input changed during provenance capture", {"path": name})
        atomic_write_json(output / "resolved.json", resolved)
        if patch is not None:
            with (output / "git.patch").open("xb") as stream:
                stream.write(patch)
        for name in destinations:
            path = contained_path(output, name)
            files.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    except OSError as exc:
        raise WorkflowError("EXECUTION", "Could not capture provenance files", {"reason": str(exc)}) from exc
    # Recheck scientific/code/data consistency after copying. This is a drift
    # check, not a filesystem transaction; callers must keep inputs quiescent.
    if resolve_experiment(root, Path(sources[0])) != current:
        raise WorkflowError("IDENTITY", "Inputs changed during provenance capture")
    record = {"schema_version": 1, "identity": deepcopy(resolved["identity"]),
              "files": sorted(files, key=lambda item: item["path"]), "environment": environment}
    validate_document("provenance_record", record)
    return record
