"""Fixed, evidence-linked LaTeX renderers; no scientific aggregation occurs here."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from researchflow import documents
from researchflow.contracts import WorkflowError

_CONTEXT = ("protocol_id", "split", "subgroup", "weighting")
_METRIC_FIELDS = frozenset((*_CONTEXT, "method_id", "metric", "value", "n_queries",
                            "n_users", "query_set_sha256", "candidate_set_sha256", "series_id"))
_COMPARISON_FIELDS = frozenset((*_CONTEXT, "id", "left", "right", "metric", "estimate",
                                "ci_low", "ci_high", "n_queries", "n_users", "status", "reason"))
_LABELS = {"protocol_id": "Protocol", "split": "Split", "subgroup": "Subgroup",
           "weighting": "Weighting", "metric": "Metric", "status": "Status", "reason": "Reason"}
_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_MISSING = r"\textemdash{}"


def _fail(message: str, **details: Any) -> None:
    raise WorkflowError("CONTRACT", message, details)


def _escape(value: str) -> str:
    if any(ord(c) < 32 and c not in "\t\n\r" for c in value):
        _fail("LaTeX text contains a control character")
    return "".join(_ESCAPES.get(c, " " if c in "\t\n\r" else c) for c in value)


def _format(value: Any, digits: int | None = None) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        _fail("Boolean evidence is not a numerical or text cell")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            _fail("Nonfinite numerical evidence")
        if digits is not None:
            return format(value, f".{digits}f")
        return str(value) if isinstance(value, int) else format(value, ".4f")
    if isinstance(value, str):
        return _escape(value)
    _fail("Only scalar text, number or null evidence can be rendered")


def _evidence(analysis: Mapping[str, Any], pointer: str) -> dict[str, Any]:
    # Resolve even automatically constructed pointers through the shared API.
    documents.resolve_pointer(analysis, pointer)
    return {"analysis_id": analysis["analysis_id"], "pointer": pointer}


def _write(analysis: Mapping[str, Any], plans: list[tuple[str, str, str, list[dict[str, Any]]]],
           output_dir: Path) -> list[dict[str, Any]]:
    """Preflight every path before writing; never overwrite existing authored files."""
    paths = []
    for artifact_id, kind, content, evidence in plans:
        relative = artifact_id + ".tex"
        path = documents.contained_path(output_dir, relative)
        if path.exists() or path.is_symlink():
            raise WorkflowError("IDENTITY", "Artifact destination already exists", {"path": relative})
        paths.append((path, relative, artifact_id, kind, content, evidence))
    result = []
    try:
        if paths:
            output_dir.mkdir(parents=True, exist_ok=True)
        for path, relative, artifact_id, kind, content, evidence in paths:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            descriptor = {"id": artifact_id, "kind": kind, "path": relative,
                          "sha256": documents.sha256_file(path), "evidence": evidence}
            documents.validate_document("rendered_artifact", descriptor)
            result.append(descriptor)
    except OSError as exc:
        raise WorkflowError("EXECUTION", "Could not write rendered artifact", {"error": str(exc)}) from exc
    return result


def _selected_rows(analysis: Mapping[str, Any], collection: str, filters: Mapping[str, Any],
                   fields: frozenset[str]) -> list[tuple[int, Mapping[str, Any]]]:
    for field, wanted in filters.items():
        if field not in fields:
            _fail("Unknown filter field", field=field)
        choices = wanted if isinstance(wanted, list) else [wanted]
        if not choices or any(isinstance(v, (dict, list, bool)) or
                              (v is not None and not isinstance(v, (str, int, float)))
                              for v in choices):
            _fail("Filters require a scalar or nonempty list of scalars", field=field)
        for value in choices:
            if isinstance(value, float) and not math.isfinite(value):
                _fail("Filter contains a nonfinite value", field=field)
    rows = [(i, row) for i, row in enumerate(analysis[collection])
            if all(row[field] in (wanted if isinstance(wanted, list) else [wanted])
                   for field, wanted in filters.items())]
    if filters and not rows:
        _fail("Explicit table filters matched no evidence", filters=dict(filters))
    return rows


def render_tables(analysis: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Render specs against one analysis_result, retaining input row order.

    Filters use exact scalar equality or list membership. Missing contextual
    columns are appended so that even a one-row table states its estimand.
    analysis_alias is routing metadata: the caller supplies its resolved analysis.
    """
    documents.validate_document("analysis_result", analysis)
    plans = []
    ids: set[str] = set()
    for spec in specs:
        documents.validate_document("table_spec", spec)
        artifact_id = spec["id"]
        if artifact_id in ids:
            _fail("Duplicate artifact ID", id=artifact_id)
        ids.add(artifact_id)
        collection, fields = (("metrics", _METRIC_FIELDS) if spec["kind"] == "metric_table"
                              else ("comparisons", _COMPARISON_FIELDS))
        columns = [dict(column) for column in spec["columns"]]
        column_fields = [column["field"] for column in columns]
        if len(set(column_fields)) != len(column_fields):
            _fail("Duplicate table column field", id=artifact_id)
        for field in column_fields:
            if field not in fields:
                _fail("Unknown table column field", field=field)
        required = (*_CONTEXT, "metric")
        if collection == "comparisons":
            required = (*required, "status", "reason")
        for field in required:
            if field not in column_fields:
                columns.append({"field": field, "label": _LABELS[field]})
        rows = _selected_rows(analysis, collection, spec["filters"], fields)
        lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
                 " & ".join(_escape(column["label"]) for column in columns) + r" \\", r"\hline"]
        evidence = []
        for index, row in rows:
            cells = []
            for column in columns:
                pointer = f"/{collection}/{index}/{column['field']}"
                value = documents.resolve_pointer(analysis, pointer)
                cells.append(_format(value, column.get("digits")))
                evidence.append(_evidence(analysis, pointer))
            for field in spec["filters"]:
                item = _evidence(analysis, f"/{collection}/{index}/{field}")
                if item not in evidence:
                    evidence.append(item)
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        # A comment is safe to include inside a caller-owned float/tabular wrapper.
        # Escape first and collapse newlines so captions cannot inject TeX commands.
        lines.insert(0, "% Caption: " + _escape(spec["caption"]))
        plans.append((artifact_id, "table", "\n".join(lines) + "\n", evidence))
    return _write(analysis, plans, output_dir)


def render_numbers(analysis: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Write a single numbers.tex, including a valid empty file for no specs."""
    documents.validate_document("analysis_result", analysis)
    names: set[str] = set()
    lines = []
    evidence = []
    for spec in specs:
        documents.validate_document("number_spec", spec)
        name = spec["name"]
        if not re.fullmatch(r"[A-Za-z]+", name):
            _fail("Macro names must contain ASCII letters only", name=name)
        if name in names:
            _fail("Duplicate macro name", name=name)
        names.add(name)
        pointer = spec["pointer"]
        value = documents.resolve_pointer(analysis, pointer)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            _fail("Number macro evidence must be numeric or null", pointer=pointer)
        rendered = _format(value, spec["digits"])
        lines.append("\\newcommand{\\" + name + "}{" + rendered + "}")
        item = _evidence(analysis, pointer)
        if item not in evidence:
            evidence.append(item)
    content = "\n".join(lines) + ("\n" if lines else "")
    return _write(analysis, [("numbers", "numbers", content, evidence)], output_dir)
