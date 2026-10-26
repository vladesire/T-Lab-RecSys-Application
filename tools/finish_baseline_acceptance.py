#!/usr/bin/env python3
"""Analyze an existing complete M19 run and build its development acceptance PDF.

Put in project/tools/. Run from project/ with its installed environment:
  uv run python tools/finish_baseline_acceptance.py --run runs/R-warmup-anchor-001
No experiment, model, network request, LLM call, or final-test scoring is launched.
--tag selects fresh derived outputs; progress allows continuation after a failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


def finish(root: Path, run_dir: Path, tag: str, *, compile_pdf: bool = True) -> dict:
    from researchflow.analysis import analyze
    from researchflow.bundles import bundle_fingerprint, verify_bundle
    from researchflow.context import build_packet
    from researchflow.documents import atomic_write_json, load_document, validate_document
    from researchflow.latex import compile_snapshot, inspect_build
    from researchflow.publication import assemble_publication
    from researchflow.research import append_journal, load_research
    from researchflow.views import build_view, write_view

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", tag):
        raise ValueError("Tag must be 1-40 letters, digits, underscores or hyphens")
    root = root.expanduser().resolve()
    run_dir = (run_dir if run_dir.is_absolute() else root / run_dir).resolve()
    run_rel = run_dir.relative_to(root).as_posix()
    manifest = verify_bundle(run_dir, expected_kind="run")
    meta = manifest["metadata"]
    if manifest["status"] != "succeeded" or meta["phase"] != "development":
        raise ValueError("Expected a successful actual-data development run")
    rid, bid = manifest["bundle_id"], meta["branch_id"]
    result_path = run_dir / meta["result_path"]
    result = load_document(result_path)
    if not result["checks"]["passed"]:
        raise ValueError("Runner checks failed")
    anchor_path = run_dir / "payload/workspace/runs" / rid / "jobs/U20_original/result.json"
    anchor = load_document(anchor_path)
    difference = anchor.get("anchor_prediction_max_absolute_difference")
    if not isinstance(difference, (int, float)) or not 0 <= difference <= 1e-8:
        raise ValueError("The archived anchor prediction criterion has not passed")
    ids = {item["id"] for item in result["predictions"]}
    methods = ["random", "popularity", "recency_rule", "original", "no_pair"]
    expected = {p + "_" + m for p in ("U20", "P20") for m in methods}
    if not expected <= ids:
        raise ValueError("Expected the ten U20/P20 iteration1 controls")

    output = root / "build/baseline-acceptance" / tag
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    fingerprint = bundle_fingerprint(manifest)
    progress = load_document(progress_path) if progress_path.exists() else {
        "schema_version": 1, "run_id": rid, "run_fingerprint": fingerprint}
    if progress["run_fingerprint"] != fingerprint:
        raise ValueError("This tag already belongs to a different run; use a new --tag")

    def rel(path):
        return Path(path).relative_to(root).as_posix()

    def record(name, path):
        progress[name] = rel(path)
        atomic_write_json(progress_path, progress)
        print(name + ": " + str(path), flush=True)
        return path

    def config(name, kind, value):
        validate_document(kind, value)
        path = root / "configs/baseline-acceptance" / tag / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if load_document(path) != value:
                raise ValueError(f"Refusing to change a generated configuration: {path}; use a new --tag")
        else:
            atomic_write_json(path, value)
        return path

    comparisons = []
    for protocol in ("U20", "P20"):
        for right in ("recency_rule", "no_pair"):
            for subgroup in ("all", "novel"):
                for weighting in ("event", "user"):
                    comparisons.append({
                        "id": f"{protocol}-original-vs-{right}-{subgroup}-{weighting}",
                        "left": protocol + "_original", "right": protocol + "_" + right,
                        "metric": "ndcg", "weighting": weighting, "subgroup": subgroup})
    analysis_cfg = config("analysis.json", "analysis_config", {
        "schema_version": 1, "analysis_id": "A-baseline-" + tag, "engine": "ranking",
        "inputs": [{"path": run_rel, "reason": "Fresh anchor reproduction and retrospective development controls"}],
        "series": [{"id": name, "run_id": rid, "prediction_id": name} for name in sorted(expected)],
        "k": 10, "comparisons": comparisons,
        "bootstrap": {"replicates": 2000, "seed": 20261025, "confidence": 0.95}})
    if "analysis" in progress:
        analysis = root / progress["analysis"]
        verify_bundle(analysis, expected_kind="analysis")
    else:
        analysis = record("analysis", analyze(root, analysis_cfg))
    am = verify_bundle(analysis, expected_kind="analysis")
    data = load_document(analysis / am["metadata"]["analysis_path"])
    if not data["checks"]["passed"] or len(data["metrics"]) != 120:
        raise ValueError(f"Baseline analysis did not validate all ten series; inspect {analysis / 'analysis.json'}")
    if any(c["status"] != "computed" for c in data["comparisons"]):
        raise ValueError("A requested within-protocol comparison is unavailable; inspect the analysis")

    now = datetime.now(timezone.utc).isoformat()
    research = load_research(root)
    if not any(item["record"]["id"] == bid for item in research["branches"]):
        branch_path = root / "research/branches" / (bid + ".json")
        branch_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(branch_path, {
            "schema_version": 1, "id": bid, "title": "Warmup baseline reproduction",
            "purpose": "Verify fresh execution, archived anchor and evidence-to-paper delivery; recorded retrospectively.",
            "parent_id": None, "hypothesis_ids": [], "planned_experiment_ids": [],
            "status": "active", "status_decision_id": None})
    did = "D-baseline-" + tag
    decision_path = root / "research/decisions" / (did + ".json")
    if not decision_path.exists():
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(decision_path, {
            "schema_version": 1, "id": did, "branch_ids": [bid], "hypothesis_ids": [],
            "recorded_at": now, "status": "retrospective",
            "judgment": f"Fresh baseline execution passed its anchor criterion (max absolute difference {difference:g}); analysis validates ten development controls. This is reproduction, not additional independent evidence.",
            "evidence": [run_rel + "/manifest.json", rel(analysis / "analysis.json")],
            "alternatives": ["Treat successful execution alone as sufficient; rejected because numerical and delivery checks are separate."],
            "next_actions": ["Inspect the acceptance PDF", "Freeze supported scientific scope before new experiments"],
            "supersedes": [], "author": "AI-assisted acceptance workflow", "model": None})
    jid = "J-baseline-" + tag
    journal_path = root / "research/journal" / (jid + ".md")
    if not journal_path.exists():
        append_journal(root, {
            "schema_version": 1, "id": jid, "recorded_at": now,
            "occurred_at": meta["finished_at"], "status": "retrospective",
            "branch_ids": [bid], "hypothesis_ids": [], "run_ids": [rid],
            "analysis_ids": [data["analysis_id"]], "decision_ids": [did], "supersedes": [],
            "author": "AI-assisted acceptance workflow", "model": None},
            "# Baseline reproduction and delivery acceptance\n\n"
            "Recorded after execution. The analysis comparisons were specified retrospectively; no preregistration is claimed.\n\n"
            f"The native anchor maximum absolute prediction difference is {difference:g}. "
            "The deterministic analyzer recomputes ten controls within their own U20/P20 protocols. "
            "The full development panel is larger than the archived anchor. Final data remain unscored. "
            "A successful PDF build is a separate step; visual review remains manual.\n")

    view_cfg = config("view.json", "view_config", {
        "schema_version": 1, "view_id": "V-baseline-" + tag,
        "branch_ids": [bid], "hypothesis_ids": [], "phases": ["development"],
        "experiment_paths": [], "run_paths": [run_rel], "analysis_paths": [rel(analysis)],
        "include_legacy_journal": False})
    view = record("view", write_view(build_view(root, view_cfg), output / "view"))
    review_cfg = config("review.json", "context_config", {
        "schema_version": 1, "packet_id": "C-baseline-" + tag, "profile": "results_reviewer",
        "task": "Audit development baseline reproduction. Separate facts, interpretation and new proposals. Check subgroup tradeoffs, candidate sampling, dependence and cost accounting. Propose at most three informative next experiments; do not claim final-test or full-catalog evidence.",
        "max_bytes": 500000,
        "sources": [
            {"path": rel(analysis / "analysis.json"), "role": "analysis", "mandatory": True, "reason": "Complete deterministic evidence"},
            {"path": rel(decision_path), "role": "decision", "mandatory": True, "reason": "Recorded retrospective judgment"},
            {"path": rel(journal_path), "role": "history", "mandatory": True, "reason": "Chronology and limitations"}]})
    if "packet" not in progress:
        record("packet", build_packet(root, review_cfg, root / "reviews/baseline-acceptance" / tag))
    else:
        verify_bundle(root / progress["packet"], expected_kind="context")

    paper = root / "paper/baseline-acceptance" / tag
    paper.mkdir(parents=True, exist_ok=True)
    panel = next(cell for cell in data["metrics"] if cell["series_id"] == "U20_original"
                 and cell["metric"] == "ndcg@10" and cell["subgroup"] == "all"
                 and cell["weighting"] == "event")
    main = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\input{generated/numbers.tex}
\title{Baseline Reproduction and Pipeline Acceptance}
\author{T-Lab recommendation study: working development draft}
\date{}
\begin{document}
\maketitle
\input{generated/status.tex}
\section{Scope}
This draft checks the research delivery path on actual development data.
It reproduces earlier evidence and provides no additional independent sample.
Comparisons in this analysis were specified retrospectively. Final targets
remain unscored. This document is not the final research paper.
\section{Protocol}
The run retains the archived cohort and training sample and evaluates PANELQUERIES
development queries from PANELUSERS users. Each target competes with 20 alternatives.
U20 uses uniform alternatives; P20 uses popularity-weighted alternatives.
Metrics use a single positive, cutoff ten, and declared score-independent ties.
The archive's original ANCHORQUERIES-query anchor is checked separately from the full
development panel. Event and equal-user means and repeat/novel subgroups are
kept distinct. Paired intervals resample users within each candidate protocol.
Sampled results do not establish full-catalog ranking quality.
\section{Computed values}
The original model's event-weighted NDCG at ten is \UOriginalNDCG{} under U20
and \POriginalNDCG{} under P20 on the expanded development panel.
The tables and figures below are generated from pinned analysis evidence.
The original, no-pair, recency, popularity and random controls are all retained.
\input{generated/artifacts.tex}
\section{Interpretation boundary}
Successful reproduction establishes a computational check, not a new scientific
claim. The manual feature dictionary, automated selection, artist features,
larger candidate sets and final evaluation require their own explicit studies.
Full feature construction is performed upstream; selected feature count does
not measure construction savings. Compilation checks do not replace visual review.
\end{document}
""".replace("PANELQUERIES", str(panel["n_queries"])).replace("PANELUSERS", str(panel["n_users"]))
    main = main.replace("ANCHORQUERIES", str(anchor["metrics"]["anchor"]["queries"]))
    main_path = paper / "main.tex"
    if main_path.exists() and main_path.read_text() != main:
        raise ValueError("Acceptance paper was edited; choose a new --tag rather than overwrite it")
    if not main_path.exists():
        main_path.write_text(main)
    tables, figures, numbers = [], [], []
    for protocol, macro in (("U20", "UOriginalNDCG"), ("P20", "POriginalNDCG")):
        filters = {"protocol_id": protocol, "metric": "ndcg@10", "weighting": "event", "subgroup": "all"}
        tables.append({"id": protocol + "-controls", "analysis_alias": "baseline", "kind": "metric_table",
                       "filters": {**filters, "subgroup": ["all", "novel"]},
                       "columns": [{"field": "series_id", "label": "Method"}, {"field": "value", "label": "NDCG@10", "digits": 6}],
                       "caption": protocol + " development controls; all and novel targets."})
        figures.append({"id": protocol + "-scores", "analysis_alias": "baseline", "kind": "metric_bars",
                        "filters": filters, "order": [protocol + "_" + m for m in methods],
                        "title": protocol + " development controls", "ylabel": "NDCG at 10"})
        matches = [i for i, cell in enumerate(data["metrics"]) if cell["series_id"] == protocol + "_original"
                   and all(cell[k] == v for k, v in filters.items())]
        if len(matches) != 1:
            raise ValueError("Original-model metric pointer is ambiguous")
        numbers.append({"name": macro, "analysis_alias": "baseline", "pointer": f"/metrics/{matches[0]}/value", "digits": 6})
    publication_cfg = config("publication.json", "publication_config", {
        "schema_version": 1, "publication_id": "P-baseline-" + tag, "mode": "draft", "source_dir": rel(paper),
        "analyses": [{"alias": "baseline", "path": rel(analysis), "sha256": bundle_fingerprint(am)}],
        "artifacts": {"tables": tables, "figures": figures, "numbers": numbers},
        "required_comparison_ids": [c["id"] for c in comparisons], "claims": []})
    snapshot = record("snapshot", assemble_publication(root, publication_cfg))
    if compile_pdf:
        compiled = record("compiled", compile_snapshot(snapshot, root / "build/compiled"))
        checks = inspect_build(compiled)
        if not checks["passed"]:
            raise ValueError("Compiled PDF failed automated inspection")
        cm = verify_bundle(compiled, expected_kind="compiled")
        record("pdf", compiled / cm["metadata"]["pdf_path"])
    progress["visual_review"] = "pending; open the exact generated PDF"
    atomic_write_json(progress_path, progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, default=Path("runs/R-warmup-anchor-001"))
    parser.add_argument("--tag", default="baseline-001")
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    try:
        result = finish(args.root, args.run, args.tag, compile_pdf=not args.no_tex)
    except Exception as error:
        print(f"Baseline acceptance stopped: {error}", file=sys.stderr)
        if getattr(error, "details", None):
            print(json.dumps(error.details, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
