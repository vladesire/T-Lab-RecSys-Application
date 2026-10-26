"""Assemble immutable, explicitly selected paper evidence; never compile TeX."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
import re
import shutil
import sys

from researchflow import bundles, documents, figures, tables
from researchflow.contracts import WorkflowError

_PACKAGE_FILES = {"packages.txt", "packages.list", "packages.json", "package-list",
                  "package-list.txt", "package-list.json"}
_MACRO = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*"
    r"(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))"
    r"|\\(?:def|gdef|edef|xdef)\s*\\([A-Za-z@]+)")


def _fail(message: str, code: str = "CONTRACT", **details: Any) -> None:
    raise WorkflowError(code, message, details)


def _text_without_comments(text: str) -> str:
    # A percent preceded by an odd number of backslashes is escaped.
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def _unique(values: list[str], what: str) -> None:
    if len(values) != len(set(values)):
        _fail("Duplicate " + what, values=values)


def _input_path(root: Path, value: str) -> Path:
    path = Path(value)
    if "latest" in {part.lower() for part in path.parts}:
        _fail("Evidence and sources must be selected explicitly", path=value)
    return (path if path.is_absolute() else root / path).resolve()


def _pointer(analysis: dict[str, Any], pointer: str) -> Any:
    try:
        value = documents.resolve_pointer(analysis, pointer)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise WorkflowError("CONTRACT", "Unresolved evidence pointer", {"pointer": pointer}) from exc
    if value is None:
        _fail("Evidence pointer resolves to unavailable evidence", pointer=pointer)
    return value


def _inventory(directory: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            _fail("Symlinks are not permitted in a snapshot", path=str(path))
        if path.is_file():
            result.append({"path": path.relative_to(directory).as_posix(),
                           "sha256": documents.sha256_file(path),
                           "size_bytes": path.stat().st_size})
    return result


def _code_identity() -> dict[str, Any]:
    sources = {}
    for name in ("publication", "tables", "figures", "documents", "bundles", "contracts"):
        path = Path(__file__).with_name(name + ".py")
        sources[name] = documents.sha256_file(path)
    versions = {}
    for package in ("numpy", "matplotlib", "jsonschema", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"sources": sources, "packages": versions, "python": sys.version}


def _check_macros(sources: Path) -> set[str]:
    definitions: dict[str, str] = {}
    for path in sorted(sources.rglob("*.tex")):
        for match in _MACRO.finditer(_text_without_comments(path.read_text(encoding="utf-8"))):
            name = next(group for group in match.groups() if group is not None)
            if name in definitions:
                _fail("Global TeX macro collision", name=name, first=definitions[name],
                      second=path.relative_to(sources).as_posix())
            definitions[name] = path.relative_to(sources).as_posix()
    return set(definitions)


def _assemble(root: Path, config_path: Path) -> Path:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = documents.load_document(config_path)
    documents.validate_document("publication_config", config)
    aliases = [selection["alias"] for selection in config["analyses"]]
    _unique(aliases, "analysis aliases")
    _unique([claim["id"] for claim in config["claims"]], "claim IDs")
    _unique([spec["id"] for kind in ("tables", "figures")
             for spec in config["artifacts"][kind]], "artifact IDs")
    _unique([spec["name"] for spec in config["artifacts"]["numbers"]], "number macros")
    for spec in [*config["claims"], *[s for group in config["artifacts"].values() for s in group]]:
        if spec["analysis_alias"] not in aliases:
            _fail("Unknown analysis alias", alias=spec["analysis_alias"])

    authored = _input_path(root, config["source_dir"])
    if not authored.is_dir():
        _fail("Authored source directory is missing", path=str(authored))
    source_files = []
    for path in sorted(authored.rglob("*")):
        if path.is_symlink():
            _fail("Authored symlinks are not supported", path=str(path))
        # Explicit authored figures and their evidence sidecars must travel
        # with the immutable source tree, rather than depend on local paths.
        if path.is_file() and (path.suffix.lower() in {".tex", ".bib", ".pdf", ".png", ".jpg", ".jpeg", ".json"} or path.name in _PACKAGE_FILES):
            relative = path.relative_to(authored).as_posix()
            if relative.split("/")[0] == "generated":
                _fail("Authored sources occupy the reserved generated directory")
            source_files.append({"path": relative, "sha256": documents.sha256_file(path),
                                 "size_bytes": path.stat().st_size})
    main = authored / "main.tex"
    if not main.is_file():
        _fail("Authored main.tex is required")
    main_text = _text_without_comments(main.read_text(encoding="utf-8"))
    for name in ("status", "numbers", "artifacts"):
        if not re.search(r"\\(?:input|include)\s*\{\s*generated/" + name + r"(?:\.tex)?\s*\}", main_text):
            _fail("main.tex must include generated/" + name + ".tex")

    selected = {}
    phases = set()
    fingerprints = []
    resolved = {**config, "source_dir": str(authored), "analyses": []}
    for selection in config["analyses"]:
        alias = selection["alias"]
        directory = _input_path(root, selection["path"])
        manifest = bundles.verify_bundle(directory, "analysis")
        documents.validate_document("analysis_manifest", manifest)
        fingerprint = bundles.bundle_fingerprint(manifest)
        if fingerprint != selection["sha256"]:
            _fail("Selected analysis fingerprint mismatch", "IDENTITY", alias=alias)
        if manifest["status"] != "succeeded":
            _fail("Selected analysis bundle did not succeed", alias=alias)
        data_path = documents.contained_path(directory, manifest["metadata"]["analysis_path"])
        data = documents.load_document(data_path)
        documents.validate_document("analysis_result", data)
        if data["analysis_id"] != manifest["metadata"]["analysis_id"]:
            _fail("Analysis ID disagrees with its manifest", "IDENTITY", alias=alias)
        # The selected numerical document must itself be covered by verification.
        relative = data_path.relative_to(directory).as_posix()
        entries = [entry for entry in manifest["files"] if entry["path"] == relative]
        if len(entries) != 1 or documents.sha256_file(data_path) != entries[0]["sha256"]:
            _fail("Analysis document is not pinned by its inventory", "IDENTITY", alias=alias)
        phases.update(manifest["metadata"]["phases"])
        phases.update(data["phases"])
        phases.update(item["phase"] for item in data["inputs"])
        if config["mode"] == "release" and (
            not data["checks"]["passed"] or
            any(item["status"] != "passed" for item in data["checks"]["items"])
        ):
            _fail("Release requires passed analysis checks", alias=alias)
        selected[alias] = (directory, manifest, data_path, data)
        fingerprints.append(fingerprint)
        resolved["analyses"].append({**selection, "path": str(directory)})
    fixture = "fixture" in phases
    if fixture and config["mode"] == "release":
        _fail("Release cannot contain fixture evidence")
    for comparison_id in config["required_comparison_ids"]:
        matches = [comparison for _, _, _, data in selected.values()
                   for comparison in data["comparisons"] if comparison["id"] == comparison_id]
        if not matches or any(c["status"] != "computed" or c["estimate"] is None for c in matches):
            _fail("Required comparison is missing or unavailable", comparison_id=comparison_id)
    claims = []
    for claim in config["claims"]:
        data = selected[claim["analysis_alias"]][3]
        _pointer(data, claim["pointer"])
        claims.append({**claim, "analysis_id": data["analysis_id"],
                       "analysis_fingerprint": next(s["sha256"] for s in config["analyses"]
                                                    if s["alias"] == claim["analysis_alias"])})
    for spec in config["artifacts"]["numbers"]:
        _pointer(selected[spec["analysis_alias"]][3], spec["pointer"])

    identity = {"authored_files": source_files, "resolved_config": resolved,
                "implementation": _code_identity()}
    digest = documents.digest_document(identity)
    # Keep the schema's 128-character bundle ID limit even for a maximum-length ID.
    bundle_id = config["publication_id"][:63] + "-" + digest
    output_root = root / "build" / "paper"
    final = output_root / bundle_id
    metadata_record = {"publication_id": config["publication_id"], "mode": config["mode"],
                       "is_fixture": fixture, "phases": sorted(phases),
                       "analysis_fingerprints": fingerprints,
                       "source_identity": {**identity, "input_digest": digest}}
    if final.exists():
        existing = bundles.verify_bundle(final, "publication")
        documents.validate_document("publication_manifest", existing)
        if existing["status"] != "succeeded" or existing["metadata"] != metadata_record:
            _fail("Existing publication identity collision", "IDENTITY", path=str(final))
        return final

    stage = bundles.begin_bundle(output_root, bundle_id)
    try:
        sources = stage / "sources"
        for entry in source_files:
            source = documents.contained_path(authored, entry["path"])
            target = documents.contained_path(sources, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if documents.sha256_file(target) != entry["sha256"]:
                _fail("Authored input changed during assembly", "IDENTITY", path=entry["path"])
        generated = sources / "generated"
        generated.mkdir(parents=True)
        artifacts = []
        artifact_ids: set[str] = set()
        artifact_paths: set[str] = set()
        wrappers: dict[str, list[str]] = {"artifacts": [], "numbers": []}
        provenance = []
        for selection in resolved["analyses"]:
            alias = selection["alias"]
            directory, manifest, data_path, data = selected[alias]
            evidence_dir = stage / "evidence" / alias
            evidence_dir.mkdir(parents=True)
            shutil.copyfile(data_path, evidence_dir / "analysis.json")
            pinned_entry = next(entry for entry in manifest["files"]
                                if entry["path"] == data_path.relative_to(directory).as_posix())
            copied_data = documents.load_document(evidence_dir / "analysis.json")
            if (documents.sha256_file(evidence_dir / "analysis.json") != pinned_entry["sha256"] or
                documents.digest_document(copied_data) != documents.digest_document(data)):
                _fail("Analysis copy changed", "IDENTITY", alias=alias)
            documents.atomic_write_json(evidence_dir / "manifest.json", manifest)
            provenance.append({**selection, "analysis_id": data["analysis_id"],
                               "snapshot_analysis_path": f"evidence/{alias}/analysis.json"})
            for kind, renderer, suffix in (
                ("tables", tables.render_tables, "tables"),
                ("numbers", tables.render_numbers, ""),
                ("figures", figures.render_figures, "figures"),
            ):
                specs = [spec for spec in config["artifacts"][kind] if spec["analysis_alias"] == alias]
                if not specs:
                    continue
                out = generated / alias / suffix
                out.mkdir(parents=True, exist_ok=True)
                descriptors = renderer(data, specs, out)
                primary = "numbers" if kind == "numbers" else kind[:-1]
                if kind != "numbers":
                    actual = [d["id"] for d in descriptors if d.get("kind") == primary]
                    if sorted(actual) != sorted(spec["id"] for spec in specs):
                        _fail("Renderer omitted or invented a requested artifact", kind=kind, alias=alias)
                elif not any(d.get("kind") == "numbers" for d in descriptors):
                    _fail("Number renderer returned no number artifacts", alias=alias)
                for descriptor in descriptors:
                    documents.validate_document("rendered_artifact", descriptor)
                    if descriptor["id"] in artifact_ids:
                        _fail("Global rendered artifact ID collision", id=descriptor["id"])
                    artifact_ids.add(descriptor["id"])
                    path = documents.contained_path(out, descriptor["path"])
                    if not path.is_file() or path.is_symlink():
                        _fail("Rendered artifact file is missing or unsafe", path=str(path))
                    if documents.sha256_file(path) != descriptor["sha256"]:
                        _fail("Rendered artifact hash mismatch", "IDENTITY", path=str(path))
                    relative = path.relative_to(sources).as_posix()
                    if relative in artifact_paths:
                        _fail("Global rendered artifact path collision", path=relative)
                    artifact_paths.add(relative)
                    if not descriptor["evidence"]:
                        _fail("Rendered artifact has no evidence", id=descriptor["id"])
                    for evidence in descriptor["evidence"]:
                        if evidence["analysis_id"] != data["analysis_id"]:
                            _fail("Rendered evidence names a different analysis")
                        _pointer(data, evidence["pointer"])
                    artifacts.append({**descriptor, "path": "sources/" + relative,
                                      "analysis_alias": alias})
                    wrapper = "numbers" if descriptor["kind"] == "numbers" else "artifacts"
                    if descriptor["kind"] == "figure":
                        wrappers[wrapper].append("\\begin{figure}[htbp]\n\\centering\n"
                                                 "\\includegraphics[width=\\linewidth]{" + relative + "}\n\\end{figure}")
                    elif descriptor["kind"] == "table":
                        if path.suffix != ".tex":
                            _fail("Text renderer must return TeX", path=relative)
                        # Context columns added by the renderer can exceed the
                        # article text width. Measure first: keep narrow tables
                        # at their natural size and shrink only oversized ones.
                        # Box zero is local to this group; graphicx is already
                        # part of the publication template contract.
                        wrappers[wrapper].append(
                            "\\par\\begingroup\\centering\n"
                            "\\setbox0=\\hbox{\\input{" + relative + "}}\n"
                            "\\ifdim\\wd0>\\linewidth\n"
                            "\\resizebox{\\linewidth}{!}{\\box0}\n"
                            "\\else\\box0\\fi\n\\par\\endgroup")
                    else:
                        if path.suffix != ".tex":
                            _fail("Text renderer must return TeX", path=relative)
                        wrappers[wrapper].append("\\input{" + relative + "}")
            # Reverify archived input bytes after rendering, including its inventory.
            if bundles.bundle_fingerprint(bundles.verify_bundle(directory, "analysis")) != selection["sha256"]:
                _fail("Analysis changed during rendering", "IDENTITY", alias=alias)
        for name, lines in wrappers.items():
            (generated / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        warnings = []
        if fixture:
            warnings.append("SYNTHETIC PIPELINE CHECK")
        if config["mode"] == "draft":
            warnings.append("DRAFT")
        if any(not data["checks"]["passed"] or any(c["status"] == "failed" for c in data["checks"]["items"])
               for _, _, _, data in selected.values()):
            warnings.append("ANALYSIS CHECKS FAILED")
        if any(any(c["status"] not in {"passed", "failed"} for c in data["checks"]["items"])
               for _, _, _, data in selected.values()):
            warnings.append("ANALYSIS CHECKS UNVERIFIED")
        status = ("\\par\\noindent\\fbox{\\parbox{0.9\\linewidth}{\\bfseries " +
                  " --- ".join(warnings) + "}}\\par\n") if warnings else "% Release evidence checks passed.\n"
        (generated / "status.tex").write_text(status, encoding="utf-8")
        # A later renderer must not clobber an earlier artifact or authored text.
        for artifact in artifacts:
            path = documents.contained_path(stage, artifact["path"])
            if not path.is_file() or documents.sha256_file(path) != artifact["sha256"]:
                _fail("Rendered artifact changed during assembly", "IDENTITY", path=artifact["path"])
        for entry in source_files:
            path = documents.contained_path(sources, entry["path"])
            if not path.is_file() or documents.sha256_file(path) != entry["sha256"]:
                _fail("Copied authored text changed during assembly", "IDENTITY", path=entry["path"])
        macros = _check_macros(sources)
        for spec in config["artifacts"]["numbers"]:
            if spec["name"] not in macros:
                _fail("Number renderer omitted a requested macro", name=spec["name"])
        documents.atomic_write_json(stage / "claims.json", {
            "schema_version": 1, "claims": claims,
            "scope": "Declared evidence links and qualifiers; not verification of all authored prose."})
        documents.atomic_write_json(stage / "provenance.json", {
            "schema_version": 1, "analyses": provenance, "artifacts": artifacts,
            "source_identity": metadata_record["source_identity"]})
        documents.atomic_write_json(stage / "publication_config.json", resolved)
        manifest = {"schema_version": 1, "bundle_id": bundle_id, "bundle_kind": "publication",
                    "status": "succeeded", "created_at": datetime.now(timezone.utc).isoformat(),
                    "files": _inventory(stage), "metadata": metadata_record}
        documents.validate_document("publication_manifest", manifest)
        return bundles.commit_bundle(stage, final, manifest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def assemble_publication(project_root: Path, config_path: Path) -> Path:
    """Return a verified publication bundle pinned to explicit analysis fingerprints."""
    try:
        return _assemble(project_root.resolve(), config_path)
    except (OSError, UnicodeError) as exc:
        raise WorkflowError("CONTRACT", "Cannot read or assemble publication files",
                            {"error": str(exc)}) from exc
