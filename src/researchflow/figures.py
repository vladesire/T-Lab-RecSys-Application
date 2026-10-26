"""Vector PDF figures with explicit selection and analysis evidence pointers.

Filters exactly match string row fields. Order lists series_id for bars and
comparison id for intervals; an empty order selects all matches sorted by ID.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from researchflow.contracts import WorkflowError
from researchflow.documents import contained_path, sha256_file, validate_document

_DIMENSIONS = ("protocol_id", "split", "subgroup", "metric", "weighting")
_STYLE = {
    "text.usetex": False, "text.parse_math": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": False, "axes.formatter.useoffset": False,
    "pdf.fonttype": 42, "savefig.transparent": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
}


def _fail(message: str, **details: Any) -> None:
    raise WorkflowError("CONTRACT", message, details)


def _finite(value: Any, field: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _fail("Figure values must be finite numbers or null", field=field)


def _select(analysis: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, list[tuple[int, Mapping[str, Any]]]]:
    bars = spec["kind"] == "metric_bars"
    collection = "metrics" if bars else "comparisons"
    key = "series_id" if bars else "id"
    allowed = set(_DIMENSIONS) | ({"method_id", "series_id"} if bars else {"id", "left", "right", "status"})
    filters = spec["filters"]
    for field, value in filters.items():
        if field not in allowed or not isinstance(value, str):
            _fail("Filters require supported row fields and exact string values", field=field)
    rows = [(i, row) for i, row in enumerate(analysis[collection])
            if all(row[field] == value for field, value in filters.items())]
    if not rows:
        _fail("Figure filters matched no rows", artifact_id=spec["id"])
    order = list(spec["order"])
    if len(order) != len(set(order)):
        _fail("Figure order contains duplicate IDs", artifact_id=spec["id"])
    available = {row[key] for _, row in rows}
    unknown = set(order) - available
    if unknown:
        _fail("Unknown requested figure IDs", ids=sorted(unknown))
    # Explicit order also selects; never append unrequested rows.
    order = order or sorted(available)
    selected = []
    for identity in order:
        matches = [(i, row) for i, row in rows if row[key] == identity]
        if len(matches) != 1:
            _fail("Ambiguous figure row; narrow filters", identity=identity)
        selected.extend(matches)
    for field in _DIMENSIONS:
        if len({row[field] for _, row in selected}) != 1:
            _fail("Figure mixes scientific strata; narrow filters", field=field)
    for _, row in selected:
        for field in (("value",) if bars else ("estimate", "ci_low", "ci_high")):
            _finite(row[field], field)
        if not bars:
            low, high = row["ci_low"], row["ci_high"]
            if (low is None) != (high is None):
                _fail("Comparison interval has only one saved limit", identity=row["id"])
            if low is not None and (low > high or row["estimate"] is None):
                _fail("Invalid comparison interval", identity=row["id"])
            if row["status"] == "unavailable" and any(
                row[field] is not None for field in ("estimate", "ci_low", "ci_high")
            ):
                _fail("Unavailable comparison contains numerical results", identity=row["id"])
    return collection, selected


def _draw(spec: Mapping[str, Any], rows: list[tuple[int, Mapping[str, Any]]]) -> Figure:
    """Private construction hook; caller owns and clears the figure."""
    fig = Figure(figsize=(max(6.0, 0.8 * len(rows)), 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    bars = spec["kind"] == "metric_bars"
    labels = []
    bounds = [0.0]
    for x, (_, row) in enumerate(rows):
        labels.append(row["series_id"] if bars else f'{row["id"]}\n{row["left"]} − {row["right"]}')
        value = row["value"] if bars else row["estimate"]
        if value is None:
            ax.text(x, 0.04, "missing", ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(), color="#666666")
            continue
        bounds.append(value)
        if bars:
            ax.bar(x, value, width=0.65, color="#356d9c")
        else:
            ax.plot([x], [value], marker="o", linestyle="none", color="#356d9c")
            low, high = row["ci_low"], row["ci_high"]
            if low is None:
                ax.text(x, 0.04, "CI missing", ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(), color="#666666")
            else:
                # Saved percentile CIs need not contain the point estimate.
                ax.vlines(x, low, high, color="#356d9c", linewidth=1.5)
                ax.hlines([low, high], x - 0.12, x + 0.12, color="#356d9c")
                bounds.extend((low, high))
    low, high = min(bounds), max(bounds)
    span = high - low
    padding = 0.08 * span if span else max(abs(low) * 0.08, 0.05)
    bottom, top = low - padding, high + padding
    if not math.isfinite(bottom) or not math.isfinite(top):
        fig.clear()
        _fail("Figure range exceeds floating point plotting limits")
    ax.set_ylim(bottom, top)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.axhline(0, color="#888888", linewidth=0.7, zorder=0)
    ax.set_xticks(range(len(rows)), labels, rotation=30 if len(rows) > 4 else 0,
                  ha="right" if len(rows) > 4 else "center")
    first = rows[0][1]
    context = (f'protocol: {first["protocol_id"]} | split: {first["split"]} | '
               f'subgroup: {first["subgroup"]}')
    ax.set_title(spec["title"] + "\n" + context, wrap=True)
    metric_label = f'{first["metric"]} | weighting: {first["weighting"]}'
    if not bars:
        metric_label = "left − right: " + metric_label + " | saved CI"
    ax.set_ylabel((spec["ylabel"] + "\n" if spec["ylabel"] else "") + metric_label)
    ax.set_xlabel("Series" if bars else "Comparison")
    return fig


def render_figures(analysis: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Render requested rows; return PDF hashes and analysis JSON pointers.

    analysis_alias is resolved by the caller: this API receives one analysis
    document, not an alias map. Missing values/CIs are annotated, never imputed.
    All specs are checked before files are written. Existing paths fail.
    """
    validate_document("analysis_result", analysis)
    plans = []
    seen: set[str] = set()
    for spec in specs:
        validate_document("figure_spec", spec)
        if spec["id"] in seen:
            _fail("Duplicate artifact ID", artifact_id=spec["id"])
        seen.add(spec["id"])
        collection, rows = _select(analysis, spec)
        path = contained_path(output_dir, spec["id"] + ".pdf")
        if path.exists() or path.is_symlink():
            _fail("Figure output already exists", path=str(path))
        plans.append((spec, collection, rows, path))
    artifacts = []
    with mpl.rc_context(_STYLE):
        for spec, collection, rows, path in plans:
            fig = _draw(spec, rows)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    fig.savefig(stream, format="pdf", bbox_inches="tight",
                                metadata={"CreationDate": None, "ModDate": None})
                artifact = {
                    "id": spec["id"], "kind": "figure",
                    "path": path.relative_to(output_dir.resolve()).as_posix(),
                    "sha256": sha256_file(path),
                    "evidence": [{"analysis_id": analysis["analysis_id"],
                                  "pointer": f"/{collection}/{i}"} for i, _ in rows],
                }
                validate_document("rendered_artifact", artifact)
                artifacts.append(artifact)
            except OSError as exc:
                raise WorkflowError("EXECUTION", "Could not write figure PDF",
                                    {"path": str(path), "reason": str(exc)}) from exc
            finally:
                fig.clear()
    return artifacts
