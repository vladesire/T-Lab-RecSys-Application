"""Explicit evidence selection, ranking analysis and external analyzer envelope checks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from researchflow.contracts import WorkflowError
from researchflow.documents import (
    atomic_write_json, canonical_bytes, contained_path, digest_document,
    load_document, sha256_file, validate_document,
)
from researchflow.bundles import begin_bundle, bundle_fingerprint, commit_bundle, verify_bundle
from researchflow.processes import run_process
from researchflow.ranking import aggregate, paired_interval, per_query_metrics


def _require(condition: bool, message: str, code: str = "CONTRACT") -> None:
    if not condition:
        raise WorkflowError(code, message)


def _unique(values: list[str], label: str) -> None:
    _require(len(values) == len(set(values)), f"Duplicate {label}")


def _file(root: Path, relative: str) -> Path:
    path = contained_path(root, relative)
    _require(path.is_file() and not path.is_symlink(), f"Missing/unsafe file: {relative}")
    return path


def _listed(bundle: Path, manifest: dict, relative: str) -> Path:
    _require(relative in {f["path"] for f in manifest["files"]},
             f"Undeclared bundle file: {relative}")
    return _file(bundle, relative)


def _code_identity(root: Path, config: dict) -> dict:
    paths = config.get("external", {}).get("code_paths", []) if config["engine"] == "external" else []
    _require(config["engine"] != "external" or bool(paths), "External analyzer needs code_paths")
    files = []
    for relative in paths:
        path = Path(relative)
        path = path if path.is_absolute() else root / path
        _require(path.exists() and not path.is_symlink(), f"Missing/unsafe analyzer code: {relative}")
        entries = sorted(path.rglob("*")) if path.is_dir() else [path]
        _require(bool(entries), f"Empty analyzer code path: {relative}")
        for entry in entries:
            _require(not entry.is_symlink() and (entry.is_dir() or entry.is_file()),
                     f"Unsafe analyzer code entry: {entry}")
            if entry.is_file():
                files.append({"path": str(entry.resolve()), "sha256": sha256_file(entry)})
    # The ranking implementation itself is scientific code, not merely the wrapper.
    if config["engine"] == "ranking":
        from researchflow import ranking
        files.append({"path": "researchflow/ranking.py", "sha256": sha256_file(Path(ranking.__file__))})
    files.append({"path": "researchflow/analysis.py", "sha256": sha256_file(Path(__file__))})
    return {"engine": config["engine"], "files": files,
            "code_sha256": digest_document(files),
            "command": config.get("external", {}).get("command", []),
            "numpy_version": np.__version__}


def _protocol(bundle: Path, manifest: dict) -> Any:
    """Find schema-shaped resolved/request snapshots in the verified inventory.

    The shared run manifest does not declare a resolved-config path. Accept
    either supplied schema, require their identities/protocols to agree, and
    never resolve a live project config or guess a protocol from a descriptor.
    """
    protocols = []
    for entry in manifest["files"]:
        if not entry["path"].endswith(".json"):
            continue
        # Other inventoried JSON artifacts may be arrays rather than documents.
        # Inspect shape before handing an actual snapshot to schema validation.
        try:
            doc = json.loads(_listed(bundle, manifest, entry["path"]).read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or "scientific" not in doc or "identity" not in doc:
            continue
        if "runtime" in doc:
            validate_document("resolved_experiment", doc)
        elif "run_id" in doc:
            validate_document("runner_request", doc)
            _require(doc["run_id"] == manifest["bundle_id"], "Request run identity mismatch", "IDENTITY")
        else:
            continue
        metadata = manifest["metadata"]
        _require(doc["identity"] == metadata["identity"], "Resolved identity mismatch", "IDENTITY")
        for key in ("phase", "experiment_id", "branch_id", "hypothesis_ids"):
            _require(doc["scientific"][key] == metadata[key], f"Resolved {key} mismatch", "IDENTITY")
        protocols.append(doc["scientific"]["protocol"])
    _require(bool(protocols), "No inventoried resolved_experiment or runner_request snapshot")
    _require(all(p == protocols[0] for p in protocols), "Ambiguous resolved protocols", "IDENTITY")
    _require(isinstance(protocols[0], dict), "Ranking requires a resolved protocol object")
    return protocols[0]


def _rows(bundle: Path, manifest: dict, descriptor: dict, k: int) -> list[dict]:
    protocol = _protocol(bundle, manifest)
    _require(descriptor["format"] == "ranking_npz_v1", "Unsupported prediction format")
    _require(digest_document(protocol) == descriptor["identity"]["protocol_sha256"],
             "Prediction protocol digest mismatch", "IDENTITY")
    # A resolved protocol may describe a family: M19 explicitly declares its
    # U20/P20 candidate protocols under `candidates`, with `id` naming the family.
    # The complete object is still bound by the digest above. Each descriptor
    # must name a declared member; pairing below still requires identical member
    # IDs, splits, queries, and candidates. A single-protocol id remains exact.
    if "candidates" in protocol:
        members = protocol["candidates"]
        _require(isinstance(members, list) and bool(members) and
                 all(isinstance(member, str) and member for member in members),
                 "Protocol candidates must declare nonempty protocol IDs")
        _require(len(members) == len(set(members)), "Duplicate candidate protocol IDs")
        _require(descriptor["protocol_id"] in members, "Undeclared candidate protocol", "IDENTITY")
    elif "id" in protocol:
        _require(protocol["id"] == descriptor["protocol_id"], "Protocol id mismatch", "IDENTITY")
    if "split" in protocol:
        _require(protocol["split"] == descriptor["split"], "Protocol split mismatch", "IDENTITY")
    relative = "payload/" + descriptor["path"]
    # Validate containment relative to payload as well as inventory membership.
    _file(bundle / "payload", descriptor["path"])
    path = _listed(bundle, manifest, relative)
    try:
        with np.load(path, allow_pickle=False) as archive:
            rows = per_query_metrics({key: archive[key] for key in archive.files}, k=k)
    except (ValueError, TypeError, OSError) as exc:
        raise WorkflowError("CONTRACT", "Invalid ranking NPZ", {"path": relative, "error": str(exc)}) from exc
    query_records = [{key: row[key] for key in
                      ("query_id", "user_id", "target_item_id", "timestamp", "is_repeat_target")}
                     for row in rows]
    candidate_records = [{"query_id": row["query_id"], "candidate_set_sha256": row["candidate_set_sha256"]}
                         for row in rows]
    for key, records in (("query_set_sha256", query_records), ("candidate_set_sha256", candidate_records)):
        actual = digest_document(sorted(records, key=lambda row: row["query_id"]))
        _require(actual == descriptor["identity"][key], f"Prediction {key} mismatch", "IDENTITY")
    return rows


def _check(result: dict, id: str, status: str, detail: str, evidence: list[str]) -> None:
    result["checks"]["items"].append(dict(id=id, status=status, detail=detail, evidence=evidence))
    if status == "failed":
        result["checks"]["passed"] = False


def _comparison(spec: dict, available: dict, failures: dict, bootstrap: dict, k: int) -> dict:
    left = available.get(spec["left"])
    right = available.get(spec["right"])
    out = {**spec, "metric": f'{spec["metric"]}@{k}',
           "protocol_id": left["descriptor"]["protocol_id"] if left else "",
           "split": left["descriptor"]["split"] if left else "",
           "estimate": None, "ci_low": None, "ci_high": None, "n_queries": 0, "n_users": 0,
           "status": "unavailable", "reason": None, "evidence": []}
    if left is None or right is None:
        out["reason"] = "; ".join(f'{s}: {failures.get(s, "series unavailable")}'
                                  for s in (spec["left"], spec["right"]) if s not in available)
        return out
    out["evidence"] = [left["evidence"], right["evidence"]]
    ld, rd = left["descriptor"], right["descriptor"]
    if ld["split"] != rd["split"] or ld["protocol_id"] != rd["protocol_id"] or ld["identity"] != rd["identity"]:
        out["reason"] = "Split/protocol/query/candidate identity mismatch"
        return out
    if left["phase"] != right["phase"]:
        out["reason"] = "Input phases differ"
        return out
    try:
        interval = paired_interval(left["rows"], right["rows"], metric=spec["metric"],
                                   weighting=spec["weighting"], subgroup=spec["subgroup"], **bootstrap)
    except WorkflowError as exc:
        if exc.code not in ("CONTRACT", "IDENTITY"):
            raise
        out["reason"] = str(exc)
        return out
    out.update(interval, status="computed", reason=None)
    return out


def _ranking(config: dict, selected: list[dict], stage: Path, result: dict) -> None:
    runs = {item["manifest"]["bundle_id"]: item for item in selected}
    available, failures = {}, {}
    for series in config["series"]:
        sid = series["id"]
        item = runs[series["run_id"]]
        manifest, bundle = item["manifest"], item["path"]
        source = f'inputs/{manifest["bundle_id"]}/manifest.json'
        try:
            _require(manifest["status"] == "succeeded",
                     f'Run status={manifest["status"]}; process={manifest["metadata"]["process_status"]}')
            relative = manifest["metadata"]["result_path"]
            _require(isinstance(relative, str), "Run has no result path")
            runner = load_document(_listed(bundle, manifest, relative))
            validate_document("runner_result", runner)
            _require(runner["kind"] == "ranking", "Ranking engine requires ranking result")
            _unique([d["id"] for d in runner["predictions"]], "prediction IDs")
            matches = [d for d in runner["predictions"] if d["id"] == series["prediction_id"]]
            _require(len(matches) == 1, f'Missing prediction {series["prediction_id"]}')
            descriptor = matches[0]
            rows = _rows(bundle, manifest, descriptor, config["k"])
            evidence = f"per_query/{sid}.jsonl"
            path = stage / evidence
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                for row in rows:
                    stream.write(canonical_bytes(row) + b"\n")
            cells = []
            for metric in ("ndcg", "recall"):
                for weighting in ("event", "user"):
                    for subgroup in ("all", "repeat", "novel"):
                        cell = aggregate(rows, metric=metric, weighting=weighting, subgroup=subgroup)
                        cells.append({**cell, "method_id": descriptor["method_id"],
                            "protocol_id": descriptor["protocol_id"], "split": descriptor["split"],
                            "subgroup": subgroup, "metric": f'{metric}@{config["k"]}',
                            "weighting": weighting, "series_id": sid,
                            "query_set_sha256": descriptor["identity"]["query_set_sha256"],
                            "candidate_set_sha256": descriptor["identity"]["candidate_set_sha256"]})
            # Publish the series only after every aggregate succeeds.
            result["metrics"].extend(cells)
            available[sid] = {"descriptor": descriptor, "rows": rows, "evidence": evidence,
                              "phase": manifest["metadata"]["phase"]}
            _check(result, sid, "passed", "Ranking descriptors and arrays agree", [source, evidence])
        except WorkflowError as exc:
            if exc.code not in ("CONTRACT", "IDENTITY"):
                raise
            failures[sid] = f"{exc.code}: {exc}"
            _check(result, sid, "failed", failures[sid], [source])
        result["coverage"].append({"series_id": sid, "run_id": series["run_id"],
                                   "status": "included" if sid in available else "excluded",
                                   "reason": failures.get(sid)})
    for spec in config["comparisons"]:
        comparison = _comparison(spec, available, failures, config["bootstrap"], config["k"])
        result["comparisons"].append(comparison)
        if comparison["status"] == "unavailable":
            _check(result, spec["id"], "failed", comparison["reason"], comparison["evidence"])


def _evidence_files(output: Path, analysis: dict) -> None:
    for path in output.rglob("*"):
        _require(not path.is_symlink() and (path.is_dir() or path.is_file()),
                 f"External output contains symlink/special file: {path}")
    refs = [ref for check in analysis["checks"]["items"] for ref in check["evidence"]]
    refs += [ref for comparison in analysis["comparisons"] for ref in comparison["evidence"]]
    for ref in refs:
        _file(output, ref)


def _external(root: Path, config: dict, selected: list[dict], stage: Path, expected: dict) -> dict:
    request = {"schema_version": 1, "analysis_id": config["analysis_id"], "config": config,
               "inputs": [{"run_id": item["manifest"]["bundle_id"], "bundle_path": str(item["path"]),
                           "manifest_sha256": item["fingerprint"]} for item in selected]}
    validate_document("analysis_request", request)
    request_path = stage / "analysis_request.json"
    atomic_write_json(request_path, request)
    output = stage / "external"
    output.mkdir()
    ext = config["external"]
    process = run_process([*ext["command"], "--request", str(request_path), "--output", str(output)],
                          cwd=root, timeout_seconds=ext["timeout_seconds"],
                          stdout_path=stage / "stdout.log", stderr_path=stage / "stderr.log")
    validate_document("process_report", process)
    atomic_write_json(stage / "process_report.json", process)
    _require(process["status"] == "succeeded" and process["exit_code"] == 0,
             f'External analyzer status={process["status"]}', "EXECUTION")
    result = load_document(_file(output, "analysis.json"))
    validate_document("analysis_result", result)
    _require(result["analysis_id"] == config["analysis_id"], "External analysis ID mismatch", "IDENTITY")
    _unique([x["run_id"] for x in result["inputs"]], "external run IDs")
    _require(sorted(result["inputs"], key=lambda x: x["run_id"]) ==
             sorted(expected["inputs"], key=lambda x: x["run_id"]), "External input identity mismatch", "IDENTITY")
    _require(sorted(result["phases"]) == expected["phases"], "External phase mismatch", "IDENTITY")
    _require(not (result["checks"]["passed"] and any(c["status"] == "failed" for c in result["checks"]["items"])),
             "External check summary contradicts failed checks")
    known = {s["id"] for s in config["series"]}
    successful = {s["manifest"]["bundle_id"] for s in selected if s["manifest"]["status"] == "succeeded"}
    eligible = {s["id"] for s in config["series"] if s["run_id"] in successful}
    _require(all(m["series_id"] in known for m in result["metrics"]), "External output has undeclared series")
    _require(all(m["series_id"] in eligible for m in result["metrics"]), "External metric uses failed run")
    _unique([canonical_bytes({key: m[key] for key in
                             ("series_id", "metric", "protocol_id", "split", "subgroup", "weighting")}).decode()
             for m in result["metrics"]], "external metric cells")
    specs = {c["id"]: c for c in config["comparisons"]}
    _unique([c["id"] for c in result["comparisons"]], "external comparison IDs")
    _require(set(specs) == {c["id"] for c in result["comparisons"]}, "External comparisons differ from request")
    for c in result["comparisons"]:
        spec = specs[c["id"]]
        _require(all(c[key] == spec[key] for key in ("left", "right", "weighting", "subgroup")) and
                 c["metric"] in (spec["metric"], f'{spec["metric"]}@{config["k"]}'),
                 "External comparison does not match request")
        if c["status"] == "unavailable":
            _require(all(c[key] is None for key in ("estimate", "ci_low", "ci_high")) and bool(c["reason"]),
                     "Unavailable external comparison needs null values and a reason")
        else:
            _require(c["left"] in eligible and c["right"] in eligible, "External comparison uses failed run")
            _require(all(c[key] is not None for key in ("estimate", "ci_low", "ci_high")) and
                     c["ci_low"] <= c["ci_high"], "Invalid external comparison interval")
    _evidence_files(output, result)
    # Retain original output unchanged; root evidence paths refer to its copy.
    result = json.loads(canonical_bytes(result))
    for record in [*result["checks"]["items"], *result["comparisons"]]:
        record["evidence"] = ["external/" + p for p in record["evidence"]]
    result["limitations"].append("External method is not independently verified; only schema/provenance and contained evidence are checked.")
    return result


def analyze(project_root: Path, config_path: Path) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path)
    config = load_document(config_path if config_path.is_absolute() else root / config_path)
    validate_document("analysis_config", config)
    _require(bool(config["inputs"]), "Explicit nonempty input selection required")
    _unique([s["id"] for s in config["series"]], "series IDs")
    _unique([c["id"] for c in config["comparisons"]], "comparison IDs")
    series_ids = {s["id"] for s in config["series"]}
    _require(all(c[side] in series_ids for c in config["comparisons"] for side in ("left", "right")),
             "Comparison refers to unknown series")
    selected = []
    for source in config["inputs"]:
        _require(bool(source["path"].strip()) and bool(source["reason"].strip()), "Input path/reason must be explicit")
        path = Path(source["path"])
        path = (path if path.is_absolute() else root / path).resolve()
        manifest = verify_bundle(path, expected_kind="run")
        validate_document("run_manifest", manifest)
        selected.append({"path": path, "manifest": manifest, "fingerprint": bundle_fingerprint(manifest),
                         "reason": source["reason"]})
    _unique([s["manifest"]["bundle_id"] for s in selected], "selected run IDs")
    run_ids = {s["manifest"]["bundle_id"] for s in selected}
    _require(all(s["run_id"] in run_ids for s in config["series"]), "Series refers to unselected run")
    identity = _code_identity(root, config)
    input_digest = digest_document({"config": config, "inputs": [s["fingerprint"] for s in selected],
                                    "analyzer_identity": identity})
    bundle_id = config["analysis_id"][:103] + "-" + input_digest[:24]
    final = root / "analyses" / bundle_id
    # Always use M03 collision behavior; never overwrite an existing analysis.
    stage = begin_bundle(root / "analyses", bundle_id)
    try:
        result = {"schema_version": 1, "analysis_id": config["analysis_id"],
                  "inputs": [{"run_id": s["manifest"]["bundle_id"], "manifest_sha256": s["fingerprint"],
                              "phase": s["manifest"]["metadata"]["phase"],
                              "origin": s["manifest"]["metadata"]["origin"]} for s in selected],
                  "phases": sorted({s["manifest"]["metadata"]["phase"] for s in selected}),
                  "metrics": [], "comparisons": [], "coverage": [], "costs": [],
                  "checks": {"passed": True, "items": []},
                  "limitations": ["Runs, reproductions, estimator seeds and populations are not pooled.",
                                  "Bundle integrity does not establish temporal validity or historical tie-key independence."]}
        atomic_write_json(stage / "analysis_config.json", config)
        atomic_write_json(stage / "selection.json", {"input_digest": input_digest, "inputs": [
            {"path": str(s["path"]), "reason": s["reason"], "fingerprint": s["fingerprint"]} for s in selected]})
        for s in selected:
            dest = stage / "inputs" / s["manifest"]["bundle_id"]
            dest.mkdir(parents=True)
            shutil.copyfile(_file(s["path"], "manifest.json"), dest / "manifest.json")
        if config["engine"] == "external":
            result = _external(root, config, selected, stage, result)
        else:
            _ranking(config, selected, stage, result)
        # Preserve input facts even if there is no selected/usable ranking series.
        for s in selected:
            manifest = s["manifest"]
            rid, meta = manifest["bundle_id"], manifest["metadata"]
            provenance = {"run_id": rid, "phase": meta["phase"], "origin": meta["origin"]}
            result["coverage"].append({**provenance, "status": manifest["status"],
                                       "process_status": meta["process_status"], "source_lineage": meta["source_lineage"]})
            if manifest["status"] != "succeeded":
                _check(result, f"run:{rid}", "failed", f'Excluded run status={manifest["status"]}',
                       [f"inputs/{rid}/manifest.json"])
                continue
            if meta["result_path"] is None:
                _check(result, f"run:{rid}", "failed", "Run has no result path", [f"inputs/{rid}/manifest.json"])
                continue
            try:
                runner = load_document(_listed(s["path"], manifest, meta["result_path"]))
                validate_document("runner_result", runner)
            except WorkflowError as exc:
                if exc.code not in ("CONTRACT", "IDENTITY"):
                    raise
                _check(result, f"run:{rid}", "failed", f"Excluded invalid result: {exc}",
                       [f"inputs/{rid}/manifest.json"])
                continue
            atomic_write_json(stage / "inputs" / rid / "result.json", runner)
            for key in ("coverage", "costs"):
                result[key].extend({**record, **provenance} for record in runner[key])
            for check in runner["checks"]["items"]:
                _check(result, f'{rid}:{check["id"]}', check["status"], check["detail"], [f"inputs/{rid}/result.json"])
            if not runner["checks"]["passed"]:
                _check(result, f"{rid}:summary", "failed", "Runner reported failed checks",
                       [f"inputs/{rid}/result.json"])
            result["limitations"].extend(f"{rid}: {limitation}" for limitation in runner["limitations"])
        validate_document("analysis_result", result)
        atomic_write_json(stage / "analysis.json", result)
        atomic_write_json(stage / "tables" / "metrics.json", result["metrics"])
        atomic_write_json(stage / "tables" / "comparisons.json", result["comparisons"])
        report = [f'# Analysis {config["analysis_id"]}', "", f'Engine: {config["engine"]}. Phases: {", ".join(result["phases"])}.',
                  f'Metric cells: {len(result["metrics"])}. Comparisons: {len(result["comparisons"])}.', "",
                  "## Selected inputs", ""]
        report += [f'- {s["manifest"]["bundle_id"]} ({s["manifest"]["metadata"]["phase"]}, '
                   f'{s["manifest"]["metadata"]["origin"]}): {s["reason"]}; fingerprint {s["fingerprint"]}' for s in selected]
        report += ["", "## Checks", ""] + [f'- {c["id"]}: {c["status"]}: {c["detail"]}' for c in result["checks"]["items"]]
        report += ["", "## Limitations", ""] + [f'- {x}' for x in result["limitations"]]
        (stage / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        # Catch changed inputs/code during a long-running external analysis.
        for s in selected:
            _require(bundle_fingerprint(verify_bundle(s["path"], expected_kind="run")) == s["fingerprint"],
                     "Input bundle changed during analysis", "IDENTITY")
        _require(_code_identity(root, config) == identity, "Analyzer code changed during analysis", "IDENTITY")
        files = []
        for path in sorted(stage.rglob("*")):
            _require(not path.is_symlink() and (path.is_file() or path.is_dir()), "Unsafe analysis output")
            if path.is_file():
                files.append({"path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path),
                              "size_bytes": path.stat().st_size})
        manifest = {"schema_version": 1, "bundle_id": bundle_id, "bundle_kind": "analysis", "status": "succeeded",
                    "created_at": datetime.now(timezone.utc).isoformat(), "files": files,
                    "metadata": {"analysis_id": config["analysis_id"], "analysis_path": "analysis.json",
                                 "input_fingerprints": [s["fingerprint"] for s in selected], "phases": result["phases"],
                                 "analyzer_identity": {**identity, "input_digest": input_digest}}}
        validate_document("analysis_manifest", manifest)
        return commit_bundle(stage, final, manifest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
