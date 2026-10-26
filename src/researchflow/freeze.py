"""Create-only final-study design records and an identity-only launch gate.

M02 owns scientific/code/data identity calculation, including declared model
inputs in config/data manifests. A freeze records those identities; it does not
attest that a holdout has never been observed or import a historical freeze.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from uuid import uuid4

from researchflow.configuration import resolve_experiment
from researchflow.contracts import WorkflowError
from researchflow.documents import atomic_write_json, load_document, validate_document


def _require_final(resolved: Mapping[str, Any]) -> None:
    validate_document("resolved_experiment", resolved)
    if resolved["scientific"]["phase"] != "final":
        raise WorkflowError(
            "CONTRACT", "A study freeze and its gate require final phase.",
            {"experiment_id": resolved["scientific"]["experiment_id"],
             "phase": resolved["scientific"]["phase"]},
        )


def _unique_configs(configs: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in configs:
        experiment_id = entry["experiment_id"]
        if experiment_id in seen:
            raise WorkflowError(
                "CONTRACT", "A freeze must contain each experiment ID exactly once.",
                {"experiment_id": experiment_id},
            )
        seen.add(experiment_id)


def _validate_freeze(freeze: Mapping[str, Any]) -> None:
    validate_document("study_freeze", freeze)
    _unique_configs(freeze["configs"])
    # JSON Schema format support can depend on optional RFC3339 packages.
    # Enforce the temporal audit requirement even when those are unavailable.
    try:
        created_at = datetime.fromisoformat(freeze["created_at"])
        if created_at.utcoffset() is None:
            raise ValueError("timestamp has no timezone")
    except ValueError as exc:
        raise WorkflowError("CONTRACT", "Freeze creation time must be a timezone-aware timestamp.") from exc


def freeze_study(project_root: Path, config_paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    """Resolve explicit final configs, then publish one new study_freeze JSON.

    Relative input and output paths use project_root. The returned document is
    the persisted freeze. Existing paths (including dangling symlinks) fail.
    """
    root = Path(project_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    # Do not resolve the output leaf: even a dangling symlink is a collision.
    if os.path.lexists(output):
        raise WorkflowError("IDENTITY", "Study freeze already exists.", {"path": str(output)})
    if not config_paths:
        raise WorkflowError("CONTRACT", "At least one explicit final configuration is required.")

    configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        path = Path(config_path)
        if not path.is_absolute():
            path = root / path
        resolved = resolve_experiment(root, path)
        _require_final(resolved)
        configs.append({
            "experiment_id": resolved["scientific"]["experiment_id"],
            "identity": deepcopy(resolved["identity"]),
        })
    _unique_configs(configs)
    freeze = {
        "schema_version": 1,
        "freeze_id": "F-" + uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configs": configs,
    }
    _validate_freeze(freeze)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        # M01's atomic writer may replace a destination. Write in private
        # staging instead, then link a complete file into the final name.
        # Same-filesystem link is atomic and refuses an existing destination,
        # including a concurrently created freeze. No overwrite fallback.
        with TemporaryDirectory(prefix=".study-freeze-", dir=output.parent) as staging:
            staged_path = Path(staging) / "freeze.json"
            atomic_write_json(staged_path, freeze)
            os.link(staged_path, output)
    except FileExistsError as exc:
        raise WorkflowError("IDENTITY", "Study freeze already exists.", {"path": str(output)}) from exc
    except OSError as exc:
        raise WorkflowError(
            "EXECUTION", "Could not create study freeze.",
            {"path": str(output), "reason": str(exc)},
        ) from exc
    return freeze


def check_final_gate(freeze_path: Path, resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated freeze iff a freshly resolved final identity matches.

    The caller must resolve current inputs with M02 before calling this gate.
    Development/fixture runs should bypass this function at the executor.
    No resolver, model, runner, phase mutation, or provenance copying occurs here.
    """
    _require_final(resolved)
    freeze = load_document(Path(freeze_path))
    _validate_freeze(freeze)
    experiment_id = resolved["scientific"]["experiment_id"]
    for entry in freeze["configs"]:
        if entry["experiment_id"] == experiment_id:
            changed = sorted(
                key for key, value in entry["identity"].items()
                if value != resolved["identity"][key]
            )
            if changed:
                raise WorkflowError(
                    "IDENTITY", "Final experiment identity differs from the study freeze.",
                    {"experiment_id": experiment_id, "changed": changed},
                )
            return freeze
    raise WorkflowError(
        "IDENTITY", "Final experiment is absent from the study freeze.",
        {"experiment_id": experiment_id},
    )
