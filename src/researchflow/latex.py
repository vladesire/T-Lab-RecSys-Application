"""Bounded TeX compilation of verified, immutable publication snapshots."""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchflow import bundles, documents, processes
from researchflow.contracts import WorkflowError

_TOOLS = ("latexmk", "pdflatex", "bibtex", "kpsewhich")
_PATTERNS = {
    "citations": r"citation[^\n]*(?:undefined|not found)|undefined citations|(?:please .*|warning:.*)(?:rerun|\(re\)run) (?:biber|bibtex)|no file .+\.bbl|empty bibliography|I couldn't open database file",
    "references": r"reference[^\n]*undefined|undefined references|label\(s\) may have changed|rerun to get cross-references right",
    "latex_errors": r"^!|LaTeX Error|Package .+ Error|Undefined control sequence|Emergency stop|Fatal error|no legal \\end found",
    "missing_assets": r"(?:file .+not found|can't find file|could not find file|I couldn't open (?:file|style file)|cannot find|not found:)",
    "overflow": r"(?:Overfull|Underfull) \\[hv]box",
}


def _probe(argv: list[str], cwd: Path) -> tuple[dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="researchflow-tex-probe-") as directory:
        root = Path(directory)
        report = processes.run_process(argv, cwd, 10.0, root / "stdout", root / "stderr")
        text = "\n".join(p.read_text(errors="replace") for p in (root / "stdout", root / "stderr") if p.exists()).strip()
        return report, text


def doctor(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise WorkflowError("CONTRACT", "TeX project root is not a directory", {"path": str(root)})
    tools: dict[str, Any] = {}
    for name in _TOOLS:
        executable = shutil.which(name)
        item: dict[str, Any] = {"path": executable, "version": None, "available": False}
        if executable:
            try:
                report, output = _probe([executable, "--version"], root)
                item.update(version=output, available=report["status"] == "succeeded" and report["exit_code"] == 0,
                            process=report)
            except (WorkflowError, OSError) as exc:
                item["error"] = str(exc)
        tools[name] = item
    # Literal package declarations, including optional arguments and comma lists.
    # TeX-generated names and transitive dependencies are checked by TeX itself.
    required: set[str] = set()
    for source in sorted(root.rglob("*.tex")):
        if source.is_symlink():
            continue
        text = re.sub(r"(?<!\\)%[^\n]*", "", source.read_text(errors="replace"))
        for group in re.findall(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\]\s*)?\{([^}]*)\}", text):
            required.update(p.strip() + ".sty" for p in group.split(",") if p.strip())
    packages: dict[str, Any] = {}
    for name in sorted(required):
        item = {"path": None, "available": False}
        local = root / name
        if not local.is_file():
            local = root / "sources" / name
        if local.is_file() and local.resolve().is_relative_to(root):
            item.update(path=str(local.resolve()), available=True)
        elif tools["kpsewhich"]["available"]:
            try:
                report, output = _probe([tools["kpsewhich"]["path"], name], root)
                candidates = output.splitlines()
                if report["status"] == "succeeded" and report["exit_code"] == 0 and candidates:
                    found = Path(candidates[0])
                    found = found if found.is_absolute() else root / found
                    if found.is_file():
                        item.update(path=str(found.resolve()), available=True)
            except (WorkflowError, OSError) as exc:
                item["error"] = str(exc)
        packages[name] = item
    missing = [name for name, item in {**tools, **packages}.items() if not item["available"]]
    return {"project_root": str(root), "tools": tools, "packages": packages, "missing": missing, "passed": not missing}


def _checks(root: Path, pdf_relative: str, report: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, evidence: list[str]) -> None:
        items.append({"id": name, "status": "passed" if ok else "failed", "detail": detail, "evidence": evidence})

    add("process", report.get("status") == "succeeded" and report.get("exit_code") == 0,
        f"Process status={report.get('status')}, exit_code={report.get('exit_code')}", ["process.json"])
    pdf = documents.contained_path(root, pdf_relative)
    exists = pdf.is_file() and pdf.stat().st_size > 5
    if exists:
        with pdf.open("rb") as stream:
            exists = stream.read(5) == b"%PDF-"
    add("pdf", exists, "Nonempty PDF with PDF header required", [pdf_relative])
    # latexmk stdout/stderr contain *all* passes, including the expected first
    # pass before BibTeX and cross-reference resolution. The final TeX log is
    # overwritten each pass and therefore describes the delivered PDF. Retain
    # the complete transcripts as provenance, but do not treat historical
    # warnings as unresolved diagnostics. BibTeX logs retain bibliography errors.
    final_log = pdf.with_suffix(".log")
    logs = [final_log] if final_log.is_file() else []
    logs += sorted((root / "build").rglob("*.blg")) if (root / "build").exists() else []
    add("log", final_log.is_file(), "Final TeX log required for automated inspection",
        [p.relative_to(root).as_posix() for p in logs])
    for name, pattern in _PATTERNS.items():
        evidence = []
        for log in logs:
            for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    evidence.append(f"{log.relative_to(root).as_posix()}:{number}: {line}")
        add(name, not evidence, "No matching diagnostics" if not evidence else "TeX diagnostics require correction", evidence)
    items.append({"id": "visual_review", "status": "unverified",
                  "detail": "Automated checks do not establish visual acceptance; separate human review is required.", "evidence": []})
    return {"passed": not any(item["status"] == "failed" for item in items), "items": items}


def compile_snapshot(snapshot_dir: Path, output_root: Path, *, timeout_seconds: float = 120.0) -> Path:
    snapshot = Path(snapshot_dir).resolve()
    output = Path(output_root).resolve()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WorkflowError("CONTRACT", "timeout_seconds must be finite and positive")
    if output.is_relative_to(snapshot):
        raise WorkflowError("CONTRACT", "Compilation output must be outside the immutable snapshot")
    manifest = bundles.verify_bundle(snapshot, expected_kind="publication")
    if manifest["status"] != "succeeded":
        raise WorkflowError("CONTRACT", "Cannot compile an unsuccessful publication snapshot")
    main = documents.contained_path(snapshot, "sources/main.tex")
    if not main.is_file():
        raise WorkflowError("CONTRACT", "Snapshot lacks sources/main.tex")
    fingerprint = bundles.bundle_fingerprint(manifest)
    bundle_id = fingerprint[:32]
    final = output / bundle_id
    if final.exists():
        old = bundles.verify_bundle(final, expected_kind="compiled")
        if old["metadata"]["snapshot_sha256"] != fingerprint:
            raise WorkflowError("IDENTITY", "Compiled fingerprint prefix collision")
        checks = inspect_build(final)
        if old["status"] == "succeeded" and checks["passed"]:
            return final
        raise WorkflowError("EXECUTION", "Existing compiled output failed checks", {"path": str(final), "checks": checks})
    toolchain = doctor(snapshot)
    if not toolchain["passed"]:
        raise WorkflowError("DEPENDENCY", "Required TeX dependencies unavailable; install explicitly", toolchain)
    stage = bundles.begin_bundle(output, bundle_id)
    # Work on a complete private copy so even TeX writes outside its outdir
    # cannot alter the input. Keep this copy and all bibliography/recorder files.
    shutil.copytree(snapshot, stage / "input", symlinks=False)
    work = stage / "input/sources"
    (stage / "build").mkdir()
    (stage / "logs").mkdir()
    argv = [toolchain["tools"]["latexmk"]["path"], "-norc", "-pdf", "-interaction=nonstopmode",
            "-halt-on-error", "-file-line-error", "-recorder", "-no-shell-escape",
            f"-outdir={stage / 'build'}", "main.tex"]
    env = dict(os.environ, MIKTEX_ENABLE_INSTALLER="0")
    report = processes.run_process(argv, work, timeout_seconds, stage / "logs/compile.stdout",
                                   stage / "logs/compile.stderr", env)
    documents.validate_document("process_report", report)
    documents.atomic_write_json(stage / "process.json", report)
    # Verification after execution detects external mutation as well as changes
    # to the snapshot manifest; the private working copy may legitimately change.
    after = bundles.verify_bundle(snapshot, expected_kind="publication")
    if bundles.bundle_fingerprint(after) != fingerprint:
        raise WorkflowError("IDENTITY", "Snapshot changed during compilation", {"diagnostics_dir": str(stage)})
    checks = _checks(stage, "build/main.pdf", report)
    documents.atomic_write_json(stage / "checks.json", checks)
    toolchain.update(command=argv, cwd="input/sources", process_report="process.json")
    documents.atomic_write_json(stage / "toolchain.json", toolchain)
    pdf = stage / "build/main.pdf"
    if not pdf.is_file() or not checks["passed"]:
        # The v1 schema requires a real PDF hash even on failed manifests. Keep
        # evidence in staging and raise instead of inventing a missing-PDF hash.
        raise WorkflowError("EXECUTION", "TeX build failed automated checks",
                            {"diagnostics_dir": str(stage), "checks": checks})
    result = {"schema_version": 1, "bundle_id": bundle_id, "bundle_kind": "compiled", "status": "succeeded",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "files": [{"path": p.relative_to(stage).as_posix(), "sha256": documents.sha256_file(p),
                         "size_bytes": p.stat().st_size} for p in sorted(stage.rglob("*")) if p.is_file()],
              "metadata": {"snapshot_sha256": fingerprint, "pdf_path": "build/main.pdf",
                           "pdf_sha256": documents.sha256_file(pdf), "toolchain": toolchain, "checks": checks}}
    documents.validate_document("compiled_manifest", result)
    return bundles.commit_bundle(stage, final, result)


def inspect_build(compiled_dir: Path) -> dict[str, Any]:
    root = Path(compiled_dir).resolve()
    manifest = bundles.verify_bundle(root, expected_kind="compiled")
    documents.validate_document("compiled_manifest", manifest)
    pdf_relative = manifest["metadata"]["pdf_path"]
    report = documents.load_document(root / "process.json")
    documents.validate_document("process_report", report)
    checks = _checks(root, pdf_relative, report)
    pdf = documents.contained_path(root, pdf_relative)
    if not pdf.is_file() or documents.sha256_file(pdf) != manifest["metadata"]["pdf_sha256"]:
        checks["items"].append({"id": "pdf_identity", "status": "failed", "detail": "PDF hash differs from manifest", "evidence": [pdf_relative]})
        checks["passed"] = False
    if manifest["status"] != "succeeded":
        checks["items"].append({"id": "bundle_status", "status": "failed", "detail": "Compiled bundle did not succeed", "evidence": ["manifest.json"]})
        checks["passed"] = False
    return checks
