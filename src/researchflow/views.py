"""Disposable, deterministic research-state projection; never executes experiments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import os
import tempfile
from urllib.parse import quote

from researchflow import bundles, documents, research
from researchflow.contracts import WorkflowError


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else documents.contained_path(root, str(value))


def _label(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Sort only projection collections, never canonical record or scientific arrays.
    return sorted(items, key=documents.canonical_bytes)


def build_view(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Join explicit inputs. Empty filter lists mean unrestricted selection.

    Decisions remain current until explicitly superseded, even by an excluded
    record. Independent decisions are not erased merely because one is newer.
    Each run is a separate row so a successful retry cannot conceal a failure.
    """
    root = project_root.resolve()
    config_file = _path(root, config_path)
    config = documents.load_document(config_file)
    documents.validate_document("view_config", config)
    index = deepcopy(research.load_research(root))
    documents.validate_document("research_index", index)
    state: dict[str, Any] = {"schema_version": 1, "view_id": config["view_id"]}
    for name in ("branches", "hypotheses", "decisions", "journal", "experiments",
                 "analyses", "unresolved", "exclusions", "sources"):
        state[name] = []
    state["unresolved"].extend(index["unresolved"])
    known_docs: dict[str, Any] = {}

    def unresolved(kind: str, identifier: str, reason: str) -> None:
        item = {"kind": kind, "id": identifier, "reason": reason}
        if item not in state["unresolved"]:
            state["unresolved"].append(item)

    def source(kind: str, path: Path, value: Any = None,
               digest: str | None = None) -> dict[str, str]:
        label = _label(root, path)
        item = {"kind": kind, "source_path": label,
                "source_sha256": digest if digest is not None else documents.sha256_file(path),
                "source_uri": path.as_uri()}
        if item not in state["sources"]:
            state["sources"].append(item)
        if value is not None:
            known_docs[label] = value
        return item

    def reasons(branch_ids: list[str], hypothesis_ids: list[str],
                phases: list[str] | None = None) -> list[str]:
        result = []
        for key, values in (("branch_ids", branch_ids), ("hypothesis_ids", hypothesis_ids)):
            # Unscoped canonical decisions/journal entries are global context.
            if config[key] and values and not set(config[key]).intersection(values):
                result.append(key + " filter")
        # Mixed-phase evidence is indivisible: do not silently display final data
        # in a development-only view by retaining just its convenient cells.
        if config["phases"] and phases is not None and not set(phases).issubset(config["phases"]):
            result.append("phases filter (all evidence phases must be selected)")
        return result

    def include(kind: str, identifier: str, path: str, why: list[str]) -> bool:
        if why:
            state["exclusions"].append({"kind": kind, "id": identifier,
                                         "source_path": path, "reasons": why})
        return not why

    source("view_config", config_file, config)
    records: dict[str, dict[str, Any]] = {}
    for plural, kind in (("branches", "branch"), ("hypotheses", "hypothesis"),
                         ("decisions", "decision")):
        records[plural] = {}
        for entry in index[plural]:
            rec = entry["record"]
            documents.validate_document(kind, rec)
            if rec["id"] in records[plural]:
                raise WorkflowError("IDENTITY", "Duplicate canonical record ID",
                                    {"kind": kind, "id": rec["id"]})
            records[plural][rec["id"]] = rec
            source(kind, _path(root, entry["source_path"]), rec, entry["source_sha256"])
    for key, plural in (("branch_ids", "branches"), ("hypothesis_ids", "hypotheses")):
        for identifier in config[key]:
            if identifier not in records[plural]:
                unresolved(key, identifier, "Requested canonical record is missing")

    successors: dict[str, list[str]] = {}
    for rec in records["decisions"].values():
        for target in rec["supersedes"]:
            successors.setdefault(target, []).append(rec["id"])
            if target not in records["decisions"]:
                unresolved("decision", target, "Superseded decision is missing")
    # Cycles are unresolved history, never an arbitrary choice of a winner.
    for identifier in records["decisions"]:
        todo = list(successors.get(identifier, []))
        visited = set()
        while todo:
            node = todo.pop()
            if node == identifier:
                unresolved("decision", identifier, "Cyclic supersession; no current judgment inferred")
                break
            if node not in visited:
                visited.add(node)
                todo.extend(successors.get(node, []))

    for plural, kind in (("branches", "branch"), ("hypotheses", "hypothesis"),
                         ("decisions", "decision")):
        for entry in index[plural]:
            rec = entry["record"]
            bids = [rec["id"]] if kind == "branch" else rec["branch_ids"]
            hids = [rec["id"]] if kind == "hypothesis" else rec["hypothesis_ids"]
            if include(kind, rec["id"], entry["source_path"], reasons(bids, hids)):
                row = deepcopy(entry)
                if kind == "decision":
                    row.update(is_current=not successors.get(rec["id"]),
                               superseded_by=sorted(successors.get(rec["id"], [])))
                state[plural].append(row)

    for entry in index["journal"]:
        header = entry["header"]
        if entry["legacy"]:
            if header is not None:
                raise WorkflowError("CONTRACT", "Legacy journal must have null header")
        else:
            if header is None:
                raise WorkflowError("CONTRACT", "Structured journal requires a header")
            documents.validate_document("journal_header", header)
        source("journal", _path(root, entry["source_path"]), digest=entry["source_sha256"])
        why = (["legacy journal disabled"] if entry["legacy"] and not config["include_legacy_journal"]
               else reasons(header["branch_ids"], header["hypothesis_ids"]) if header else [])
        if include("journal", header["id"] if header else entry["source_path"], entry["source_path"], why):
            state["journal"].append(deepcopy(entry))

    run_rows: dict[str, dict[str, Any]] = {}
    all_runs: dict[str, dict[str, Any]] = {}
    for value in sorted(set(config["run_paths"])):
        path = _path(root, value)
        if not path.exists():
            unresolved("run", value, "Explicit run bundle is missing; no completion or score inferred")
            state["experiments"].append({"kind": "run", "id": value, "source_path": value,
                                         "status": "unresolved", "evidence": None})
            continue
        manifest = bundles.verify_bundle(path, "run")
        documents.validate_document("run_manifest", manifest)
        meta = manifest["metadata"]
        src = source("run_manifest", path / "manifest.json", manifest)
        identifier = manifest["bundle_id"]
        if identifier in all_runs:
            raise WorkflowError("IDENTITY", "Duplicate named run ID", {"id": identifier})
        row = {"kind": "run", "id": identifier, "run_id": identifier,
               "experiment_id": meta["experiment_id"], "branch_id": meta["branch_id"],
               "hypothesis_ids": meta["hypothesis_ids"], "phase": meta["phase"],
               "status": "completed" if manifest["status"] == "succeeded" else manifest["status"],
               "recorded_status": manifest["status"], "metadata": deepcopy(meta),
               "source_path": _label(root, path), "source_sha256": src["source_sha256"],
               "bundle_fingerprint": bundles.bundle_fingerprint(manifest), "evidence": None}
        all_runs[identifier] = row
        if not include("run", identifier, row["source_path"],
                       reasons([meta["branch_id"]], meta["hypothesis_ids"], [meta["phase"]])):
            continue
        result_path = meta["result_path"]
        if result_path is not None:
            result_file = documents.contained_path(path, result_path)
            result = documents.load_document(result_file)
            documents.validate_document("runner_result", result)
            source("runner_result", result_file, result)
            row["evidence"] = {k: deepcopy(result[k]) for k in
                               ("kind", "metrics", "coverage", "costs", "checks", "limitations")}
            row["evidence"]["source_path"] = _label(root, result_file)
            # Descriptors retain provenance without opening NPZ/artifact payloads.
            row["evidence"]["predictions"] = deepcopy(result["predictions"])
            if manifest["status"] == "succeeded" and not result["checks"]["passed"]:
                row["status"] = "invalidated"
        elif manifest["status"] == "succeeded":
            row["status"] = "invalidated"
            unresolved("run", identifier, "Succeeded manifest has no result summary")
        run_rows[identifier] = row
        state["experiments"].append(row)

    staged: set[tuple[str, str]] = set()
    configs_seen: set[str] = set()
    for value in sorted(set(config["experiment_paths"])):
        path = _path(root, value)
        if not path.exists():
            unresolved("experiment_config", value, "Explicit staged config is missing")
            continue
        document = documents.load_document(path)
        documents.validate_document("experiment_config", document)
        sci = document["scientific"]
        src = source("experiment_config", path, document)
        identifier = sci["experiment_id"]
        if identifier in configs_seen:
            raise WorkflowError("IDENTITY", "Ambiguous staged experiment ID", {"id": identifier})
        configs_seen.add(identifier)
        if not include("experiment", identifier, src["source_path"],
                       reasons([sci["branch_id"]], sci["hypothesis_ids"], [sci["phase"]])):
            continue
        key = (sci["branch_id"], identifier)
        staged.add(key)
        matches = sorted(r["id"] for r in run_rows.values()
                         if (r["branch_id"], r["experiment_id"], r["phase"]) == (*key, sci["phase"]))
        branch = records["branches"].get(sci["branch_id"], {})
        row = {"kind": "staged", "id": identifier, "experiment_id": identifier,
               "branch_id": sci["branch_id"], "hypothesis_ids": sci["hypothesis_ids"],
               "phase": sci["phase"], "scientific": deepcopy(sci),
               "source_path": src["source_path"], "source_sha256": src["source_sha256"],
               "status": "blocked" if branch.get("status") == "blocked" else "staged",
               "run_ids": matches, "evidence": None,
               "note": "Runs share declared ID/branch/phase; config identity equivalence is not inferred."}
        state["experiments"].append(row)
        if not matches:
            unresolved("experiment", identifier, "Staged configuration has no selected run")

    for entry in state["branches"]:
        rec = entry["record"]
        for identifier in rec["planned_experiment_ids"]:
            if (rec["id"], identifier) in staged or any(
                r["branch_id"] == rec["id"] and r["experiment_id"] == identifier
                for r in run_rows.values()
            ):
                continue
            state["experiments"].append({"kind": "planned", "id": identifier,
                "experiment_id": identifier, "branch_id": rec["id"],
                "hypothesis_ids": rec["hypothesis_ids"], "phase": None,
                "status": "blocked" if rec["status"] == "blocked" else "planned",
                "source_path": entry["source_path"], "run_ids": [], "evidence": None})
            unresolved("experiment", identifier,
                       "Plan has no selected staged config/run; phase and execution remain unknown")

    analysis_ids: set[str] = set()
    for value in sorted(set(config["analysis_paths"])):
        path = _path(root, value)
        if not path.exists():
            unresolved("analysis", value, "Explicit analysis bundle is missing")
            state["analyses"].append({"id": value, "source_path": value,
                                      "status": "unresolved", "evidence": None})
            continue
        manifest = bundles.verify_bundle(path, "analysis")
        documents.validate_document("analysis_manifest", manifest)
        meta = manifest["metadata"]
        src = source("analysis_manifest", path / "manifest.json", manifest)
        identifier = meta["analysis_id"]
        if identifier in analysis_ids:
            raise WorkflowError("IDENTITY", "Duplicate named analysis ID", {"id": identifier})
        analysis_ids.add(identifier)
        result_file = documents.contained_path(path, meta["analysis_path"])
        result = None
        if result_file.is_file():
            result = documents.load_document(result_file)
            documents.validate_document("analysis_result", result)
            source("analysis_result", result_file, result)
            if result["analysis_id"] != identifier or set(result["phases"]) != set(meta["phases"]):
                raise WorkflowError("IDENTITY", "Analysis result disagrees with manifest", {"id": identifier})
        elif manifest["status"] == "succeeded":
            unresolved("analysis", identifier, "Succeeded manifest has no analysis summary")
        inputs = result["inputs"] if result is not None else []
        if not {item["phase"] for item in inputs}.issubset(meta["phases"]):
            raise WorkflowError("IDENTITY", "Analysis phases omit an input phase", {"id": identifier})
        why = reasons([], [], meta["phases"])
        for inp in inputs:
            run = all_runs.get(inp["run_id"])
            if run is None:
                unresolved("run", inp["run_id"], "Analysis input is not among the explicitly named runs")
                if config["branch_ids"] or config["hypothesis_ids"]:
                    why.append("Cannot establish filtered scope of analysis input " + inp["run_id"])
            else:
                why.extend(reasons([run["branch_id"]], run["hypothesis_ids"], [run["phase"]]))
                if inp["phase"] != run["phase"] or inp["origin"] != run["metadata"]["origin"]:
                    raise WorkflowError("IDENTITY", "Analysis input metadata disagrees with run", {"id": identifier})
                # The port does not distinguish canonical-document hashing from
                # file-byte hashing for this field. Both identify this exact
                # verified manifest; retain both rather than invent a convention.
                if inp["manifest_sha256"] not in {run["source_sha256"], run["bundle_fingerprint"]}:
                    unresolved("analysis", identifier, "Input manifest hash does not match named run " + inp["run_id"])
        if not inputs and (config["branch_ids"] or config["hypothesis_ids"]):
            why.append("No analysis inputs available to establish filtered scope")
        if not include("analysis", identifier, _label(root, path), sorted(set(why))):
            continue
        status = "completed" if manifest["status"] == "succeeded" else manifest["status"]
        invalid_hash = any(x["kind"] == "analysis" and x["id"] == identifier and
                           x["reason"].startswith("Input manifest hash") for x in state["unresolved"])
        if status == "completed" and (result is None or not result["checks"]["passed"] or invalid_hash):
            status = "invalidated"
        state["analyses"].append({"id": identifier, "bundle_id": manifest["bundle_id"],
            "source_path": _label(root, path), "source_sha256": src["source_sha256"],
            "status": status, "recorded_status": manifest["status"],
            "metadata": deepcopy(meta), "evidence": deepcopy(result)})

    # Canonical references are resolved against the full index, not just its
    # projection. Exclusion is different from absence.
    def check_refs(rec: Mapping[str, Any]) -> None:
        for key, plural in (("branch_ids", "branches"), ("hypothesis_ids", "hypotheses"),
                            ("decision_ids", "decisions")):
            for identifier in rec.get(key, []):
                if identifier not in records[plural]:
                    unresolved(key, identifier, "Referenced canonical record is missing")
        for key, available in (("run_ids", all_runs), ("analysis_ids", analysis_ids)):
            for identifier in rec.get(key, []):
                if identifier not in available:
                    unresolved(key, identifier, "Referenced evidence bundle was not supplied")

    for plural in ("branches", "hypotheses", "decisions"):
        for entry in state[plural]:
            rec = entry["record"]
            check_refs(rec)
            for key, target in (("parent_id", "branches"), ("status_decision_id", "decisions")):
                if rec.get(key) is not None and rec[key] not in records[target]:
                    unresolved(key, rec[key], "Referenced canonical record is missing")
            for reference in rec.get("evidence", []) + rec.get("references", []):
                file_part, marker, pointer = reference.partition("#")
                evidence_path = _path(root, file_part) if file_part else None
                label = _label(root, evidence_path) if evidence_path else ""
                if label not in known_docs:
                    unresolved("evidence", reference, "Reference is not a supplied summary; not read implicitly")
                elif marker:
                    try:
                        documents.resolve_pointer(known_docs[label], pointer)
                    except WorkflowError:
                        unresolved("evidence", reference, "JSON pointer is unresolved in supplied summary")
    for entry in state["journal"]:
        if entry["header"] is not None:
            check_refs(entry["header"])
    for row in state["experiments"]:
        if row.get("branch_id") is not None and row["branch_id"] not in records["branches"]:
            unresolved("branch", row["branch_id"], "Experiment branch is missing")
        check_refs(row)
    for key in state:
        if isinstance(state[key], list):
            state[key] = _ordered(state[key])
    documents.validate_document("state_view", state)
    return state


def _text(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ").replace("|", "\\|")


def write_view(state: Mapping[str, Any], output_dir: Path) -> Path:
    """Write a disposable projection directory; canonical sources are untouched."""
    documents.validate_document("state_view", state)
    output = output_dir.resolve()
    lines = [f"# Research view: {_text(state['view_id'])}", "",
             "Execution states and recorded judgments do not establish scientific proof.", "",
             "Sources use project-relative labels and absolute operational links.", ""]
    uris = {x["source_path"]: x["source_uri"] for x in state["sources"] if "source_uri" in x}
    def link(path: str) -> str:
        uri = uris.get(path, uris.get(path.rstrip("/") + "/manifest.json"))
        return f"[source](<{uri}>)" if uri else _text(path)
    for key in ("branches", "hypotheses", "decisions", "journal", "experiments", "analyses",
                "unresolved", "exclusions", "sources"):
        lines.extend([f"## {key.capitalize()}", ""])
        for row in state[key]:
            rec = row.get("record", row.get("header") or row)
            identifier = rec.get("id", row.get("source_path", "unknown"))
            status = row.get("status", rec.get("status", "recorded"))
            if key == "decisions":
                status = "current" if row["is_current"] else "superseded by " + ", ".join(row["superseded_by"])
            lines.append(f"- **{_text(identifier)}** — {_text(status)}" +
                         (" · " + link(row["source_path"]) if "source_path" in row else ""))
            if key == "hypotheses":
                lines.append("  " + _text(rec["question"]))
            if key == "decisions":
                lines.append("  " + _text(rec["judgment"]))
                for reference in rec["evidence"]:
                    path, _, pointer = reference.partition("#")
                    uri = uris.get(path)
                    target = uri + "#" + quote(pointer, safe="/~") if uri else None
                    lines.append("  Evidence: " + (f"[{_text(reference)}](<{target}>)" if target else _text(reference)))
            if key == "journal":
                lines.append("  " + _text(row["body"]))
            if key == "experiments" and row.get("branch_id"):
                lines.append(f"  Experiment {_text(row['experiment_id'])}; branch {_text(row['branch_id'])}; phase {_text(row.get('phase'))}.")
            evidence = row.get("evidence")
            if evidence:
                lines.append("  Checks passed: " + _text(evidence["checks"]["passed"]) + ".")
                for metric in evidence["metrics"]:
                    lines.append("  " + _text(f"{metric['method_id']} / {metric['protocol_id']} / {metric['split']} / {metric['subgroup']} / {metric['weighting']}: {metric['metric']} = {metric['value']}"))
                for comparison in evidence.get("comparisons", []):
                    lines.append("  " + _text(f"Comparison {comparison['id']}: {comparison['status']}; estimate {comparison['estimate']}; CI [{comparison['ci_low']}, {comparison['ci_high']}]; reason {comparison['reason']}"))
                for item in evidence["checks"]["items"]:
                    lines.append("  Check: " + _text(f"{item['id']}: {item['status']} — {item['detail']}"))
                for limitation in evidence["limitations"]:
                    lines.append("  Limitation: " + _text(limitation))
            if "reason" in row:
                lines.append("  " + _text(row["reason"]))
            if "reasons" in row:
                lines.append("  " + _text("; ".join(row["reasons"])))
            if "source_sha256" in row:
                lines.append("  SHA256: `" + row["source_sha256"] + "`.")
        if not state[key]:
            lines.append("None.")
        lines.append("")
    output.mkdir(parents=True, exist_ok=True)
    documents.atomic_write_json(output / "view.json", state)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=".view-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(lines))
        os.replace(temporary, output / "view.md")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
