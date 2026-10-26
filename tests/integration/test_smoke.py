"""Run only after real M01--M16 dependencies merge; stubs must fail, not skip.

Run: python -m pytest tests/integration/test_smoke.py -v
Only the TeX test may skip for absent executables. No service mocks are used.
Human PDF inspection is a separate acceptance step and never fabricated here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from researchflow.analysis import analyze
from researchflow.contracts import WorkflowError
from researchflow.publication import assemble_publication
from researchflow.smoke import smoke

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-smoke")
    # Copy owned templates only; installed researchflow provides actual dependencies.
    for rel in ("runners/fixture/runner.py", "configs/examples/fixture_experiment.yaml",
                "configs/examples/fixture_analysis.yaml", "configs/examples/fixture_publication.yaml",
                "paper/main.tex", "paper/sections/introduction.tex", "paper/sections/methods.tex",
                "paper/sections/results.tex", "paper/references.bib", "paper/texlive-packages.txt"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return root, smoke(root, compile_pdf=False)


def test_real_merged_pipeline_preserves_arithmetic_and_is_marked(scenario):
    root, result = scenario
    workspace = Path(result["workspace"])
    # M03 keeps an empty staging area beside its terminal bundles.
    for name in ("runs", "analyses"):
        entries = list((workspace / name).iterdir())
        assert len([p for p in entries if p.name != ".staging"]) == 1
        staging = workspace / name / ".staging"
        if staging.exists():
            assert not list(staging.iterdir())
    assert not (root / "research").exists()
    assert not (root / "runs").exists()
    assert (Path(result["view"]) / "view.md").is_file()
    view = read(Path(result["view"]) / "view.json")
    assert view["branches"] and view["hypotheses"] and view["analyses"] and view["experiments"]
    assert "H-fixture" in json.dumps(view) and "B-fixture" in json.dumps(view)
    assert (Path(result["packet"]) / "manifest.json").is_file()
    analysis = read(Path(result["analysis"]) / "analysis.json")
    expected = read(ROOT / "tests/fixtures/ranking_expected.json")
    assert analysis["checks"]["passed"] and analysis["phases"] == ["fixture"]
    for method in ("A", "B"):
        for weighting in ("event", "user"):
            for metric in ("ndcg", "recall"):
                rows = [r for r in analysis["metrics"] if r["series_id"] == method
                        and r["weighting"] == weighting and r["metric"] == metric + "@2"
                        and r["subgroup"] == "all"]
                assert len(rows) == 1
                assert rows[0]["value"] == pytest.approx(expected[f"{method}_{weighting}_{metric}"])
                assert rows[0]["n_queries"] == 3 and rows[0]["n_users"] == 2
    snapshot = Path(result["snapshot"])
    metadata = read(snapshot / "manifest.json")["metadata"]
    assert metadata["is_fixture"] is True and metadata["mode"] == "draft"
    assert list(snapshot.rglob("status.tex"))
    assert "synthetic pipeline check" in next(snapshot.rglob("status.tex")).read_text().lower()
    assert list(snapshot.rglob("numbers.tex")) and list(snapshot.rglob("artifacts.tex"))
    assert any("fixture-table" in str(p) and p.suffix == ".tex" for p in snapshot.rglob("*"))
    assert any("fixture-figure" in str(p) and p.suffix in {".pdf", ".png", ".svg"}
               for p in snapshot.rglob("*"))


def test_real_analysis_collision_preserves_existing_bundle(scenario):
    _, result = scenario
    bundle = Path(result["analysis"])
    before = inventory(bundle)
    # M08 intentionally uses M03's exclusive collision policy.
    with pytest.raises(WorkflowError) as error:
        analyze(Path(result["workspace"]), Path(result["analysis_config"]))
    assert error.value.code == "IDENTITY"
    assert inventory(bundle) == before


def test_changed_scientific_parameter_retains_prior_bundles(scenario):
    root, first = scenario
    previous = inventory(Path(first["smoke_root"]))
    path = root / "configs/examples/fixture_experiment.yaml"
    original = path.read_bytes()
    config = yaml.safe_load(original)
    config["scientific"]["parameters"]["score_scale"] = 2
    path.write_text(yaml.safe_dump(config))
    try:
        second = smoke(root, compile_pdf=False)
    finally:
        path.write_bytes(original)
    identities = [read(Path(r["run"]) / "manifest.json")["metadata"]["identity"]
                  for r in (first, second)]
    assert identities[0]["scientific_sha256"] != identities[1]["scientific_sha256"]
    assert first["snapshot"] != second["snapshot"]
    assert inventory(Path(first["smoke_root"])) == previous


def test_fixture_publication_release_is_rejected(scenario):
    _, result = scenario
    workspace = Path(result["workspace"])
    config = read(Path(result["publication_config"]))
    config["mode"] = "release"
    path = workspace / "configs/smoke/release-rejection.json"
    path.write_text(json.dumps(config))
    with pytest.raises(WorkflowError) as error:
        assemble_publication(workspace, path)
    assert error.value.code in {"CONTRACT", "IDENTITY"}
    assert any(word in str(error.value).lower() for word in ("fixture", "final", "phase"))


def _tex_executable(name):
    return shutil.which(name) or (str(Path("/Library/TeX/texbin") / name)
                                 if (Path("/Library/TeX/texbin") / name).is_file() else None)


def test_actual_basictex_compilation(scenario):
    # No opt-in environment variable: on an equipped Mac this executes automatically.
    missing = [name for name in ("latexmk", "pdflatex", "bibtex", "kpsewhich") if not _tex_executable(name)]
    if missing:
        pytest.skip("Actual TeX portion only: missing " + ", ".join(missing))
    root, _ = scenario
    result = smoke(root, compile_pdf=True)
    assert Path(result["pdf"]).read_bytes().startswith(b"%PDF-")
    assert Path(result["doctor"]).is_file() and Path(result["inspection"]).is_file()
    assert "pending" in result["visual_inspection"]
    # Retained output is evidence for separate visual inspection, not a visual signoff.
    print("Actual TeX PDF requiring human visual inspection:", result["pdf"])
