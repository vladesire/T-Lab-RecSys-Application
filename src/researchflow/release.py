"""Verified, local-only release packaging. No experiment or TeX execution."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import re
import shutil
import tarfile

from researchflow.contracts import WorkflowError
from researchflow.documents import (
    load_document, sha256_file, atomic_write_json, validate_document, contained_path,
)
from researchflow.bundles import begin_bundle, commit_bundle, verify_bundle, bundle_fingerprint
from researchflow.latex import inspect_build


_PRIVATE = re.compile(r"(?:/(?:Users|home|workspace|private|tmp)/[^\s\"']+|[A-Za-z]:[\\/]Users[\\/])")
_SECRET = re.compile(r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|\b(?:api_key|access_token|password|secret_key)\s*[=:]\s*[\"']?[^\s\"']+)", re.I)
_EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "cache", "caches", ".cache", "raw", "raw_data", "datasets", "node_modules"}


def _require(condition: bool, message: str, code: str = "CONTRACT") -> None:
    if not condition:
        raise WorkflowError(code, message)


def _relative(root: Path, rel: str) -> Path:
    p = PurePosixPath(rel)
    _require(bool(rel) and not p.is_absolute() and not any(x in {"", ".", ".."} for x in rel.split("/"))
             and "\\" not in rel and ":" not in rel, "Unsafe release path")
    result = contained_path(root, rel)
    _require(not any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts) + 1)),
             "Symlinks cannot enter release archives")
    return result


def _public(path: Path, rel: str) -> None:
    parts = PurePosixPath(rel).parts
    _require(not any(x.lower() in _EXCLUDED or x.startswith(".env") for x in parts)
             and path.suffix.lower() not in {".pem", ".key", ".p12", ".pyc"},
             "Excluded raw data/cache/credential file selected")
    # Binary artifacts are preserved byte for byte. Text is rejected, never silently redacted.
    try:
        value = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return
    _require(not _PRIVATE.search(value), "Private machine path in selected public file")
    _require(not _SECRET.search(value), "Possible credential in selected public file")


def _checks(value: dict[str, Any]) -> None:
    checks = value.get("checks", value)
    _require(checks.get("passed") is True and bool(checks.get("items")), "Missing or failed evidence checks")
    _require(all(x.get("status") == "passed" and x.get("id", "").lower() != "fixture"
                 for x in checks["items"]), "Failed, fixture or unverified evidence check")


def _bundle(path: Path, kind: str) -> tuple[dict[str, Any], str]:
    manifest = verify_bundle(path, expected_kind=kind)
    validate_document(kind + "_manifest", manifest)
    _require(manifest["status"] == "succeeded", "Input bundle did not succeed")
    return manifest, bundle_fingerprint(manifest)


def _entries(path: Path, manifest: dict[str, Any]) -> list[tuple[Path, str]]:
    entries = [(path / "manifest.json", "manifest.json")]
    seen = {"manifest.json"}
    for item in manifest["files"]:
        rel = item["path"]
        _require(rel not in seen, "Duplicate bundle inventory entry")
        seen.add(rel)
        source = _relative(path, rel)
        _require(source.is_file(), "Missing inventoried input")
        entries.append((source, rel))
    return entries


def _review(path: Path | None, pdf_hash: str, created_at: str) -> dict[str, Any]:
    _require(path is not None, "An actual visual review record is required")
    value = load_document(path)
    validate_document("visual_review", value)
    _require(bool(value["reviewer"].strip()) and value["reviewer"].strip().lower() not in
             {"fixture", "synthetic", "example"}, "A named human reviewer is required")
    _require(value["outcome"] == "passed" and value["pdf_sha256"] == pdf_hash,
             "Visual review does not approve this PDF", "IDENTITY")
    try:
        recorded = datetime.fromisoformat(value["recorded_at"].replace("Z", "+00:00"))
        built = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        _require(recorded.utcoffset() is not None and built.utcoffset() is not None,
                 "Review and build timestamps require timezones")
        _require(built <= recorded <= datetime.now(timezone.utc),
                 "Review must follow the build and cannot be future dated")
    except (ValueError, TypeError) as exc:
        raise WorkflowError("CONTRACT", "Invalid review timestamp") from exc
    _require(not re.search(r"\b(?:synthetic|fixture)\b", value["notes"], re.I),
             "Synthetic visual acceptance cannot authorize release")
    return value


def package_release(project_root: Path, snapshot_dir: Path, compiled_dir: Path, output_dir: Path, *, visual_review: Path | None = None) -> Path:
    root = project_root.resolve()
    def resolve(p: Path) -> Path:
        return (p if p.is_absolute() else root / p).resolve()
    snapshot, compiled, output = map(resolve, (snapshot_dir, compiled_dir, output_dir))
    _require(not any(output.is_relative_to(p) or p.is_relative_to(output)
                     for p in (snapshot, compiled)), "Release output overlaps an input bundle")
    publication, snapshot_hash = _bundle(snapshot, "publication")
    build, compiled_hash = _bundle(compiled, "compiled")
    meta = publication["metadata"]
    _require(meta["mode"] == "release" and meta["is_fixture"] is False
             and "fixture" not in meta["phases"], "Release requires non-fixture publication evidence")
    _require(build["metadata"]["snapshot_sha256"] == snapshot_hash,
             "Compiled PDF belongs to another snapshot", "IDENTITY")
    _checks(build["metadata"])
    _checks(inspect_build(compiled))
    pdf = _relative(compiled, build["metadata"]["pdf_path"])
    pdf_hash = sha256_file(pdf)
    _require(pdf_hash == build["metadata"]["pdf_sha256"] and pdf.read_bytes().startswith(b"%PDF-"),
             "Compiled PDF identity is invalid", "IDENTITY")
    _require(build["metadata"]["pdf_path"] in {x["path"] for x in build["files"]},
             "PDF is missing from compiled inventory")
    review_path = resolve(visual_review) if visual_review is not None else None
    review = _review(review_path, pdf_hash, build["created_at"])
    _public(review_path, "visual_review.json")

    selected: dict[str, Path] = {}
    expected: dict[str, str] = {}
    dependencies: list[dict[str, Any]] = []
    limitations: list[str] = []
    inventories: list[str] = []
    def select(source: Path, dest: str, digest: str | None = None) -> None:
        _relative(root, dest)
        _require(not source.resolve().is_relative_to(output), "Output cannot be a release input")
        _require(dest not in selected, "Archive destination collision")
        _public(source, dest)
        selected[dest] = source
        expected[dest] = digest if digest is not None else sha256_file(source)
    def include_bundle(path: Path, manifest: dict[str, Any], prefix: str) -> None:
        for source, rel in _entries(path, manifest):
            digest = next((x["sha256"] for x in manifest["files"] if x["path"] == rel), None)
            select(source, prefix + "/" + rel, digest)
    include_bundle(snapshot, publication, "snapshot")
    include_bundle(compiled, build, "compiled")
    # No filename convention is specified for the copied publication config.
    configs = []
    for dest, source in list(selected.items()):
        if dest.startswith("snapshot/") and source.suffix == ".json":
            value = load_document(source)
            if {"required_comparison_ids", "analyses", "artifacts", "source_dir"} <= value.keys():
                configs.append(value)
    _require(len(configs) == 1, "Snapshot must inventory exactly one publication config")
    config = configs[0]
    validate_document("publication_config", config)
    _require(config["mode"] == "release" and config["publication_id"] == meta["publication_id"],
             "Copied publication configuration disagrees with snapshot")
    _require(bool(config["required_comparison_ids"]), "No required main comparisons declared")
    fingerprints, comparisons = [], {}
    for ref in config["analyses"]:
        analysis_path = _relative(root, ref["path"])
        manifest, fingerprint = _bundle(analysis_path, "analysis")
        _require(fingerprint == ref["sha256"], "Selected analysis identity mismatch", "IDENTITY")
        fingerprints.append(fingerprint)
        _require("fixture" not in manifest["metadata"]["phases"], "Fixture analysis cannot release")
        result = load_document(_relative(analysis_path, manifest["metadata"]["analysis_path"]))
        validate_document("analysis_result", result)
        _checks(result)
        _require(result["inputs"] and "fixture" not in result["phases"]
                 and all(x["phase"] != "fixture" for x in result["inputs"]), "Fixture or empty analysis inputs")
        _require([x["manifest_sha256"] for x in result["inputs"]] == manifest["metadata"]["input_fingerprints"],
                 "Analysis input identities disagree", "IDENTITY")
        for comparison in result["comparisons"]:
            comparisons.setdefault(comparison["id"], []).append((comparison, result["inputs"]))
        limitations.extend(result["limitations"])
        for run in result["inputs"]:
            # No run-directory resolver is provided by v1. Explicit references are portable
            # identifiers, not assertions that omitted runs can be recreated offline.
            dependencies.append({"kind": "run_reconstruction", "run_id": run["run_id"],
                                 "manifest_sha256": run["manifest_sha256"], "included": False,
                                 "limitation": "Original verified run bundle and its data/code/config are required to reconstruct this input."})
        include_bundle(analysis_path, manifest, "analyses/" + ref["alias"])
    _require(sorted(fingerprints) == sorted(meta["analysis_fingerprints"]),
             "Publication analysis identities disagree", "IDENTITY")
    for required in config["required_comparison_ids"]:
        candidates = comparisons.get(required, [])
        _require(len(candidates) == 1, "Required main comparison is missing or ambiguous")
        comparison, inputs = candidates[0]
        _require(comparison["status"] == "computed" and comparison["estimate"] is not None
                 and all(x["phase"] == "final" for x in inputs),
                 "Required main comparison lacks computed final evidence")

    # Only explicit code/configuration trees and known reproduction files are selected.
    # Never walk the project root, data directories, or output directories.
    for directory in ("src", "configs", "config", "scripts", "interfaces"):
        base = root / directory
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                rel = path.relative_to(root).as_posix()
                if path.is_file() and not path.resolve().is_relative_to(output) and not any(
                        x.lower() in _EXCLUDED for x in path.relative_to(root).parts):
                    _relative(root, rel)
                    select(path, "reproduction/" + rel)
    for name in ("pyproject.toml", "uv.lock", "requirements.txt", "requirements.lock", "poetry.lock",
                 "Makefile", "README.md", "REPRODUCE.md", "LICENSE"):
        path = root / name
        if path.is_file():
            _relative(root, name)
            select(path, "reproduction/" + name)
    _require(any(x.startswith("reproduction/src/") for x in selected), "Reproduction source code missing")
    _require("reproduction/pyproject.toml" in selected, "Project configuration missing")
    _require(any("reproduction/" + x in selected for x in ("uv.lock", "requirements.lock", "poetry.lock")),
             "Reproduction dependency lock missing")
    _require(any("reproduction/" + x in selected for x in ("README.md", "REPRODUCE.md")),
             "Scientific reproduction instructions missing")
    # Inventory envelopes are intentionally open in v1. Preserve them and report
    # every explicit external-data declaration, without inferring data acquisition.
    for dest, source in list(selected.items()):
        if source.suffix != ".json":
            continue
        value = load_document(source)
        def collect(node: Any) -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in {"external_dependencies", "raw_data_dependencies", "data_dependencies"}:
                        inventories.append(dest)
                        declarations = child if isinstance(child, list) else [child]
                        dependencies.extend({"kind": "source_inventory_dependency", "inventory": dest,
                                             "declaration": d, "included": False} for d in declarations)
                    collect(child)
            elif isinstance(node, list):
                for child in node:
                    collect(child)
        collect(value)
    limitations.append("Raw datasets and caches are excluded. Run references are not bundled run evidence. Offline experiment reproduction is not certified.")
    if not inventories:
        limitations.append("No explicit external-data declarations were found in included inventories; data dependency completeness is unverified.")

    bundle_id = "release-" + snapshot_hash[:24] + "-" + compiled_hash[:24]
    _require(not (output / bundle_id).exists(), "Release identity already exists", "IDENTITY")
    stage = begin_bundle(output, bundle_id)
    final = output / bundle_id
    try:
        payload = stage / "payload"
        payload.mkdir()
        for dest, source in sorted(selected.items()):
            target = payload / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            _require(sha256_file(target) == expected[dest], "Input changed while packaging", "IDENTITY")
        atomic_write_json(payload / "visual_review.json", review)
        atomic_write_json(payload / "portability.json", {"offline_complete": False,
                          "external_dependencies": dependencies, "limitations": limitations,
                          "source_inventories": sorted(set(inventories))})
        (payload / "RELEASE_README.md").write_text(
            "# Release reproduction\n\nVerify CHECKSUMS.json before using the files.\n"
            "The snapshot and compiled directories preserve selected paper sources, generated artifacts, PDF and build evidence.\n"
            "Analyses are preserved under analyses/. Run fingerprints in portability.json identify omitted bundles.\n"
            "Read portability.json and analysis limitations before interpreting or reproducing results.\n"
            "Install the locked environment described by reproduction/pyproject.toml and its lock file explicitly.\n"
            "Follow reproduction/README.md and any REPRODUCE.md for scientific commands and data setup.\n"
            "No raw data are supplied and offline reproduction is not certified. This package performs no acquisition or execution.\n",
            encoding="utf-8")
        files = [{"path": p.relative_to(payload).as_posix(), "sha256": sha256_file(p),
                  "size_bytes": p.stat().st_size} for p in sorted(payload.rglob("*")) if p.is_file()]
        atomic_write_json(payload / "CHECKSUMS.json", {"files": files})
        archive = stage / "release.tar.gz"
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(payload.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(payload).as_posix()
                    info = tar.gettarinfo(str(path), arcname=rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        shutil.rmtree(payload)
        release = {"schema_version": 1, "bundle_id": bundle_id, "bundle_kind": "release",
                   "status": "succeeded", "created_at": datetime.now(timezone.utc).isoformat(),
                   "files": [{"path": "release.tar.gz", "sha256": sha256_file(archive),
                              "size_bytes": archive.stat().st_size}],
                   "metadata": {"snapshot_sha256": snapshot_hash, "compiled_sha256": compiled_hash,
                                "archive_path": "release.tar.gz", "external_dependencies": dependencies,
                                "visual_review": review}}
        validate_document("release_manifest", release)
        return commit_bundle(stage, final, release)
    except BaseException as exc:
        if stage.exists():
            shutil.rmtree(stage)
        if isinstance(exc, OSError):
            raise WorkflowError("EXECUTION", "Release packaging failed; no terminal bundle committed") from exc
        raise
