"""Execute synthetic acceptance in an isolated, retained workspace.

No dependency is faked here. Unit tests patch the documented service functions;
merged integration tests must exercise their real implementations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from uuid import uuid4

import yaml

from researchflow.analysis import analyze
from researchflow.context import build_packet
from researchflow.contracts import WorkflowError
from researchflow.experiments import execute_experiment
from researchflow.latex import compile_snapshot, doctor, inspect_build
from researchflow.publication import assemble_publication
from researchflow.views import build_view, write_view


_FIXTURES = (
    "runners/fixture/runner.py",
    "configs/examples/fixture_experiment.yaml",
    "configs/examples/fixture_analysis.yaml",
    "configs/examples/fixture_publication.yaml",
    "paper/main.tex", "paper/sections/introduction.tex", "paper/sections/methods.tex",
    "paper/sections/results.tex", "paper/references.bib", "paper/texlive-packages.txt",
)


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkflowError("CONTRACT", "Fixture config must be an object", {"path": str(path)})
    return value


def _local(path: Path, workspace: Path) -> Path:
    """Assert dependency outputs stayed in the dedicated smoke workspace."""
    path = Path(path)
    path = (workspace / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(workspace) or not path.is_dir():
        raise WorkflowError("CONTRACT", "Smoke dependency returned an unsafe/missing bundle",
                            {"path": str(path), "workspace": str(workspace)})
    return path


def _fingerprint(manifest: Path) -> str:
    # GLOBAL_RULES: scientific input identity differs from this output fingerprint.
    canonical = json.dumps(_read(manifest), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _records(workspace: Path) -> dict[str, Path]:
    records = {
        "hypothesis": ("hypotheses/H-fixture.yaml", {
            "schema_version": 1, "id": "H-fixture", "branch_ids": ["B-fixture"],
            "question": "Does the synthetic infrastructure fixture preserve known arithmetic?",
            "rationale": "Synthetic acceptance only, not recommendation research.",
            "estimand": {"metric": "ndcg@2"}, "comparison": {}, "decision_rule": {},
            "references": [], "created_at": None}),
        "branch": ("branches/B-fixture.yaml", {
            "schema_version": 1, "id": "B-fixture", "title": "Synthetic infrastructure fixture",
            "purpose": "Acceptance only", "parent_id": None, "hypothesis_ids": ["H-fixture"],
            "planned_experiment_ids": ["E-fixture"], "status": "planned", "status_decision_id": None}),
        "decision": ("decisions/D-fixture.yaml", {
            "schema_version": 1, "id": "D-fixture", "branch_ids": ["B-fixture"],
            "hypothesis_ids": ["H-fixture"],
            "recorded_at": datetime.now(timezone.utc).isoformat(), "status": "prospective",
            "judgment": "Run a synthetic plumbing check; never release this as research evidence.",
            "evidence": [], "alternatives": [], "next_actions": [], "supersedes": [],
            "author": "synthetic-smoke-driver", "model": None}),
    }
    paths = {}
    for kind, (relative, value) in records.items():
        path = workspace / "research" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        paths[kind] = path
    (workspace / "research/journal").mkdir(parents=True)
    return paths


def smoke(project_root: Path, *, compile_pdf: bool = True) -> dict[str, Any]:
    """Run fresh fixture evidence, returning JSON-serializable absolute paths.

    Retains partial work on failure. Does not auto-skip a requested compilation,
    install dependencies, fabricate a review response, or grant visual acceptance.
    """
    root = Path(project_root).resolve()
    for relative in _FIXTURES:
        source = root / relative
        if not source.is_file():
            raise WorkflowError("DEPENDENCY", "Missing smoke fixture template", {"path": str(source)})
    # Resolve before writing so a pre-existing build symlink cannot escape root.
    smoke_root = (root / "build/smoke").resolve()
    if not smoke_root.is_relative_to(root):
        raise WorkflowError("CONTRACT", "Smoke output escapes project root")
    smoke_root.mkdir(parents=True, exist_ok=True)
    attempt = smoke_root / uuid4().hex
    attempt.mkdir(exist_ok=False)
    workspace = attempt / "workspace"
    workspace.mkdir()
    try:
        for relative in _FIXTURES:
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        records = _records(workspace)
        configs = workspace / "configs/smoke"
        experiment = _config(workspace / "configs/examples/fixture_experiment.yaml")
        # Bind the command to this interpreter and the copied runner, independent of cwd/PATH.
        experiment["runtime"]["command"] = [sys.executable, str(workspace / "runners/fixture/runner.py")]
        if experiment["scientific"]["phase"] != "fixture":
            raise WorkflowError("CONTRACT", "Smoke requires fixture phase")
        experiment_path = _write(configs / "experiment.json", experiment)
        run_id = "R-smoke-" + attempt.name
        run = _local(execute_experiment(workspace, experiment_path, run_id=run_id), workspace)
        run_manifest = _read(run / "manifest.json")
        if run_manifest["status"] != "succeeded" or run_manifest["metadata"]["phase"] != "fixture":
            raise WorkflowError("EXECUTION", "Smoke fixture execution did not succeed", {"run": str(run)})

        analysis_config = _config(workspace / "configs/examples/fixture_analysis.yaml")
        analysis_config["inputs"] = [{"path": str(run.relative_to(workspace)),
                                      "reason": "Fresh synthetic infrastructure fixture only"}]
        for series in analysis_config["series"]:
            series["run_id"] = run_id
        analysis_path = _write(configs / "analysis.json", analysis_config)
        analysis = _local(analyze(workspace, analysis_path), workspace)
        analysis_result = _read(analysis / "analysis.json")
        if analysis_result["phases"] != ["fixture"] or not analysis_result["checks"]["passed"]:
            raise WorkflowError("EXECUTION", "Smoke analysis failed fixture checks", {"analysis": str(analysis)})

        view_config = {
            "schema_version": 1, "view_id": "V-fixture", "branch_ids": ["B-fixture"],
            "hypothesis_ids": ["H-fixture"], "phases": ["fixture"],
            "experiment_paths": [str(experiment_path.relative_to(workspace))],
            "run_paths": [str(run.relative_to(workspace))],
            "analysis_paths": [str(analysis.relative_to(workspace))], "include_legacy_journal": False,
        }
        view_path = _write(configs / "view.json", view_config)
        view = _local(write_view(build_view(workspace, view_path), workspace / "views/fixture"), workspace)
        packet_config = {
            "schema_version": 1, "packet_id": "C-fixture", "profile": "results_reviewer",
            "task": "Review synthetic infrastructure arithmetic and limitations only; no scientific conclusion.",
            "max_bytes": 1000000,
            "sources": [{"path": str(path.relative_to(workspace)), "role": role,
                         "mandatory": True, "reason": "Exact synthetic acceptance source"}
                        for path, role in [(analysis / "analysis.json", "analysis"),
                                           (records["hypothesis"], "hypothesis"),
                                           (experiment_path, "config"),
                                           (records["decision"], "decision")]],
        }
        packet_path = _write(configs / "reviewer.json", packet_config)
        packet = _local(build_packet(workspace, packet_path, workspace / "reviews/fixture"), workspace)

        publication = _config(workspace / "configs/examples/fixture_publication.yaml")
        if publication["mode"] != "draft":
            raise WorkflowError("CONTRACT", "Fixture publication must be a marked draft")
        publication["analyses"] = [{"alias": "fixture", "path": str(analysis.relative_to(workspace)),
                                    "sha256": _fingerprint(analysis / "manifest.json")}]
        # Pin the number by semantic row selection; analysis output ordering is not an API.
        matches = [i for i, row in enumerate(analysis_result["metrics"])
                   if row.get("series_id") == "A" and row.get("metric") == "ndcg@2"
                   and row.get("weighting") == "event" and row.get("subgroup") == "all"]
        if len(matches) != 1:
            raise WorkflowError("CONTRACT", "Expected exactly one fixture A/event/all NDCG@2 row")
        for number in publication["artifacts"]["numbers"]:
            if number["name"] == "FixtureNDCG":
                number["pointer"] = f"/metrics/{matches[0]}/value"
        publication_path = _write(configs / "publication.json", publication)
        snapshot = _local(assemble_publication(workspace, publication_path), workspace)
        metadata = _read(snapshot / "manifest.json")["metadata"]
        if metadata["mode"] != "draft" or metadata["is_fixture"] is not True:
            raise WorkflowError("CONTRACT", "Publication did not mark the fixture draft")

        compiled = pdf = inspection_path = doctor_path = None
        if compile_pdf:
            doctor_path = _write(workspace / "tex-doctor.json", doctor(workspace))
            compiled = _local(compile_snapshot(snapshot, workspace / "build/compiled"), workspace)
            inspection = inspect_build(compiled)
            inspection_path = _write(workspace / "tex-inspection.json", inspection)
            manifest = _read(compiled / "manifest.json")
            if manifest["status"] != "succeeded" or not manifest["metadata"]["checks"]["passed"]:
                raise WorkflowError("EXECUTION", "Compiled smoke failed diagnostics", {"compiled": str(compiled)})
            if inspection.get("passed") is False or inspection.get("checks", {}).get("passed") is False:
                raise WorkflowError("EXECUTION", "Smoke build inspection failed", {"inspection": str(inspection_path)})
            pdf = (compiled / manifest["metadata"]["pdf_path"]).resolve()
            if not pdf.is_relative_to(compiled) or not pdf.is_file():
                raise WorkflowError("CONTRACT", "Compiled PDF missing or outside compiled bundle")
        paths = {
            "smoke_root": attempt, "workspace": workspace, "run": run, "analysis": analysis,
            "view": view, "packet": packet, "snapshot": snapshot, "compiled": compiled,
            "pdf": pdf, "doctor": doctor_path, "inspection": inspection_path,
            "experiment_config": experiment_path, "analysis_config": analysis_path,
            "view_config": view_path, "packet_config": packet_path, "publication_config": publication_path,
            **records,
        }
        result = {key: str(value) if value is not None else None for key, value in paths.items()}
        result.update({"is_fixture": True, "run_id": run_id,
                       "visual_inspection": "pending; requires separate human review",
                       "files": sorted(str(p) for p in workspace.rglob("*") if p.is_file())})
        result["summary"] = str(attempt / "smoke.json")
        _write(attempt / "smoke.json", result)
        return result
    except Exception as exc:
        _write(attempt / "failure.json", {"is_fixture": True, "workspace": str(workspace),
                                         "exception": type(exc).__name__, "message": str(exc)})
        if isinstance(exc, WorkflowError):
            exc.details.setdefault("smoke_workspace", str(workspace))
        raise
