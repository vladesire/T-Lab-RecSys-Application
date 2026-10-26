"""M13 isolation tests. M01/M03/M11/M12 are fakes, never edited stubs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import jsonschema
import pytest

from researchflow import publication as pub
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src/researchflow/schemas"
EXAMPLES = ROOT / "tests/fixtures/schema_examples"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate(kind, value):
    schema = json.loads((SCHEMAS / (kind + ".json")).read_text())
    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)
    except jsonschema.ValidationError as exc:
        raise WorkflowError("CONTRACT", str(exc)) from exc


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise WorkflowError("CONTRACT", "Unsafe relative path")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise WorkflowError("CONTRACT", "Escaping path")
    return result


def pointer(value, path):
    if path == "":
        return value
    if not path.startswith("/"):
        raise WorkflowError("CONTRACT", "Bad JSON pointer")
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def inventory(directory):
    return [{"path": path.relative_to(directory).as_posix(), "sha256": sha(path),
             "size_bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "manifest.json"]


def verify(directory, expected_kind=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    if expected_kind and manifest["bundle_kind"] != expected_kind:
        raise WorkflowError("CONTRACT", "Wrong bundle kind")
    if manifest["files"] != inventory(directory):
        raise WorkflowError("IDENTITY", "Archived bundle inventory mismatch")
    return manifest


def begin(output_root, bundle_id):
    output_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".stage-", dir=output_root))


def commit(stage, final, manifest):
    validate("publication_manifest", manifest)
    assert manifest["files"] == inventory(stage)
    assert not final.exists()
    write_json(stage / "manifest.json", manifest)
    stage.rename(final)
    return final


def rendered(path, output, artifact_id, kind, analysis, evidence):
    return {"id": artifact_id, "kind": kind, "path": path.relative_to(output).as_posix(),
            "sha256": sha(path), "evidence": [{"analysis_id": analysis["analysis_id"],
                                                   "pointer": evidence}]}


def render_tables(analysis, specs, output_dir):
    descriptors = []
    for spec in specs:
        path = output_dir / "nested" / (spec["id"] + ".tex")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\\begin{tabular}{r}" + str(analysis["metrics"][0]["value"]) + "\\end{tabular}\n")
        descriptors.append(rendered(path, output_dir, spec["id"], "table", analysis, "/metrics/0"))
    return descriptors


def render_numbers(analysis, specs, output_dir):
    descriptors = []
    for spec in specs:
        path = output_dir / (spec["name"] + ".tex")
        path.write_text("\\newcommand{\\" + spec["name"] + "}{" +
                        str(pointer(analysis, spec["pointer"])) + "}\n")
        descriptors.append(rendered(path, output_dir, spec["name"], "numbers", analysis, spec["pointer"]))
    return descriptors


def render_figures(analysis, specs, output_dir):
    descriptors = []
    for spec in specs:
        path = output_dir / "pdf" / (spec["id"] + ".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not a real PDF: no figure rendering/TeX integration is claimed.
        path.write_bytes(b"synthetic figure bytes: " + str(analysis["metrics"][0]["value"]).encode())
        descriptors.append(rendered(path, output_dir, spec["id"], "figure", analysis, "/metrics/0"))
    return descriptors


@pytest.fixture(autouse=True)
def dependencies(monkeypatch):
    for name, fake in {
        "load_document": lambda path: json.loads(path.read_text()), "canonical_bytes": canonical,
        "digest_document": digest, "sha256_file": sha, "atomic_write_json": write_json,
        "validate_document": validate, "contained_path": contained, "resolve_pointer": pointer,
    }.items():
        monkeypatch.setattr(pub.documents, name, fake)
    for name, fake in {"verify_bundle": verify, "bundle_fingerprint": digest,
                       "begin_bundle": begin, "commit_bundle": commit}.items():
        monkeypatch.setattr(pub.bundles, name, fake)
    monkeypatch.setattr(pub.tables, "render_tables", render_tables)
    monkeypatch.setattr(pub.tables, "render_numbers", render_numbers)
    monkeypatch.setattr(pub.figures, "render_figures", render_figures)


def make_analysis(root, *, name="selected", fixture=False, value=0.7):
    directory = root / "analyses" / name
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads((EXAMPLES / "analysis_result.json").read_text())
    phase = "fixture" if fixture else "final"
    data["phases"] = [phase]
    data["inputs"][0]["phase"] = phase
    data["metrics"][0]["value"] = value
    data["analysis_id"] = "A-" + name
    data["comparisons"] = [{
        "id": "required", "left": "A", "right": "B", "protocol_id": "P",
        "split": "final", "subgroup": "all", "metric": "ndcg@2", "weighting": "event",
        "estimate": 0.1, "ci_low": 0.01, "ci_high": 0.2, "n_queries": 3, "n_users": 2,
        "status": "computed", "reason": None, "evidence": ["/metrics/0"],
    }]
    return pin_analysis(directory, data)


def pin_analysis(directory, data):
    write_json(directory / "analysis.json", data)
    manifest = json.loads((EXAMPLES / "analysis_manifest.json").read_text())
    manifest["metadata"]["analysis_id"] = data["analysis_id"]
    manifest["metadata"]["phases"] = data["phases"]
    manifest["files"] = inventory(directory)
    write_json(directory / "manifest.json", manifest)
    return {"alias": "main", "path": str(directory), "sha256": digest(manifest)}


@pytest.fixture
def project(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n"
        "\\input{generated/status.tex}\n\\input{generated/numbers.tex}\n"
        "\\input{section.tex}\n\\input{generated/artifacts.tex}\n\\end{document}\n")
    (paper / "section.tex").write_text("An authored interpretation.\n")
    (paper / "refs.bib").write_text("@misc{source,title={Authored bibliography}}\n")
    (paper / "packages.txt").write_text("graphicx\n")
    config = json.loads((EXAMPLES / "publication_config.json").read_text())
    config.update(publication_id="paper", analyses=[make_analysis(tmp_path)],
                  required_comparison_ids=["required"], claims=[{
                      "id": "claim", "statement": "A scoped claim.", "analysis_alias": "main",
                      "pointer": "/comparisons/0/estimate", "qualifier": "This population only."}])
    config["artifacts"] = {
        "tables": [{"id": "results", "analysis_alias": "main", "kind": "metric_table",
                    "filters": {}, "columns": [{"field": "value", "label": "Score"}], "caption": "Results"}],
        "numbers": [{"name": "Score", "analysis_alias": "main", "pointer": "/metrics/0/value", "digits": 3}],
        "figures": [{"id": "scores", "analysis_alias": "main", "kind": "metric_bars",
                     "filters": {}, "order": ["A"], "title": "Scores", "ylabel": "Score"}],
    }
    write_json(tmp_path / "config.json", config)
    return tmp_path, config


def assemble(project):
    root, config = project
    write_json(root / "config.json", config)
    return pub.assemble_publication(root, Path("config.json"))


def test_real_tex_wide_table_fits_without_overflow(project, monkeypatch):
    if os.environ.get("RESEARCHFLOW_REAL_TEX") != "1":
        pytest.skip("Set RESEARCHFLOW_REAL_TEX=1 for actual table layout compilation")
    if not shutil.which("pdflatex"):
        pytest.skip("Actual pdflatex unavailable")
    root, config = project
    config["artifacts"]["figures"] = []

    def wide_table(analysis, specs, output_dir):
        path = output_dir / "wide.tex"
        path.write_text(r"\begin{tabular}{l}\rule{500pt}{1pt}\\Wide table\end{tabular}" + "\n")
        return [rendered(path, output_dir, specs[0]["id"], "table", analysis, "/metrics/0")]

    monkeypatch.setattr(pub.tables, "render_tables", wide_table)
    output = assemble(project)
    sources = output / "sources"
    build = root / "tex-output"
    build.mkdir()
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                             "-no-shell-escape", "-output-directory=" + str(build), "main.tex"],
                            cwd=sources, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    log = (build / "main.log").read_text()
    assert "Overfull" not in log and "Underfull" not in log, log
    assert (build / "main.pdf").is_file()


def change_analysis(project, change):
    root, config = project
    directory = Path(config["analyses"][0]["path"])
    data = json.loads((directory / "analysis.json").read_text())
    change(data)
    config["analyses"][0] = pin_analysis(directory, data)


def test_snapshot_inventory_paths_and_authored_claims(project, monkeypatch, tmp_path):
    root, config = project
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = assemble(project)
    manifest = verify(output, "publication")
    validate("publication_manifest", manifest)
    assert output.parent == root / "build/paper"
    for name in ("main.tex", "section.tex", "refs.bib", "packages.txt"):
        assert (output / "sources" / name).read_bytes() == (root / "paper" / name).read_bytes()
    assert "generated/main/tables/nested/results.tex" in (output / "sources/generated/artifacts.tex").read_text()
    assert "generated/main/figures/pdf/scores.pdf" in (output / "sources/generated/artifacts.tex").read_text()
    assert "generated/main/Score.tex" in (output / "sources/generated/numbers.tex").read_text()
    claims = json.loads((output / "claims.json").read_text())
    assert claims["claims"][0]["qualifier"] == config["claims"][0]["qualifier"]
    assert claims["claims"][0]["analysis_fingerprint"] == config["analyses"][0]["sha256"]
    assert (output / "evidence/main/analysis.json").is_file()
    provenance = json.loads((output / "provenance.json").read_text())
    for artifact in provenance["artifacts"]:
        assert sha(output / artifact["path"]) == artifact["sha256"]


@pytest.mark.parametrize("target", ["section.tex", "refs.bib", "packages.txt"])
def test_authored_edits_change_snapshot(project, target):
    root, _ = project
    before = assemble(project)
    original = sha(before / "manifest.json")
    path = root / "paper" / target
    path.write_text(path.read_text() + "\n% changed\n")
    after = assemble(project)
    assert before != after
    assert sha(before / "manifest.json") == original


def test_authored_figure_and_sidecar_are_pinned_and_archived(project):
    root, _ = project
    asset = root / "paper" / "plot.pdf"
    asset.write_bytes(b"%PDF-1.4\nexplicit source fixture\n")
    sidecar = root / "paper" / "plot.json"
    sidecar.write_text('{"selected_evidence":"first"}\n')
    before = assemble(project)
    assert (before / "sources/plot.pdf").read_bytes() == asset.read_bytes()
    assert (before / "sources/plot.json").read_bytes() == sidecar.read_bytes()
    sidecar.write_text('{"selected_evidence":"second"}\n')
    after = assemble(project)
    assert after != before
    assert (before / "sources/plot.json").read_text() == '{"selected_evidence":"first"}\n'


def test_unverified_draft_check_is_not_labelled_as_a_failure(project):
    root, config = project
    config["mode"] = "draft"
    directory = Path(config["analyses"][0]["path"])
    data = json.loads((directory / "analysis.json").read_text())
    data["checks"]["items"][0]["status"] = "unverified"
    config["analyses"][0] = pin_analysis(directory, data)
    output = assemble(project)
    status = (output / "sources/generated/status.tex").read_text()
    assert "ANALYSIS CHECKS UNVERIFIED" in status
    assert "FAILED" not in status
    config["mode"] = "release"
    with pytest.raises(WorkflowError):
        assemble(project)


def test_selected_analysis_changes_snapshot(project):
    before = assemble(project)
    change_analysis(project, lambda data: data["metrics"][0].update(value=0.8))
    after = assemble(project)
    assert before != after
    assert "0.8" in (after / "sources/generated/main/Score.tex").read_text()


def test_unselected_finishing_run_does_not_change_snapshot(project):
    root, _ = project
    before = assemble(project)
    fingerprint = sha(before / "manifest.json")
    make_analysis(root, name="unselected", value=0.99)
    assert assemble(project) == before
    assert sha(before / "manifest.json") == fingerprint


def test_fixture_draft_visibly_marked_release_rejected(project):
    root, config = project
    # Fixture taint in an input alone must survive inconsistent phase summaries.
    change_analysis(project, lambda data: data["inputs"][0].update(phase="fixture"))
    draft = assemble(project)
    assert "SYNTHETIC PIPELINE CHECK" in (draft / "sources/generated/status.tex").read_text()
    assert verify(draft)["metadata"]["is_fixture"] is True
    config["mode"] = "release"
    with pytest.raises(WorkflowError, match="fixture"):
        assemble(project)


def test_clean_release(project):
    project[1]["mode"] = "release"
    output = assemble(project)
    assert not verify(output)["metadata"]["is_fixture"]
    assert "DRAFT" not in (output / "sources/generated/status.tex").read_text()


@pytest.mark.parametrize("change", [
    lambda data: data["checks"].update(passed=False),
    lambda data: data["checks"]["items"][0].update(status="failed"),
    lambda data: data["checks"]["items"][0].update(status="unverified"),
])
def test_release_rejects_failed_or_unverified_checks(project, change):
    project[1]["mode"] = "release"
    change_analysis(project, change)
    with pytest.raises(WorkflowError, match="checks"):
        assemble(project)


@pytest.mark.parametrize("mode", ["draft", "release"])
@pytest.mark.parametrize("failure", ["missing", "unavailable", "null"])
def test_required_comparison_failures(project, mode, failure):
    project[1]["mode"] = mode
    project[1]["claims"] = []
    def change(data):
        if failure == "missing":
            data["comparisons"] = []
        elif failure == "unavailable":
            data["comparisons"][0].update(status="unavailable", estimate=None, reason="missing observations")
        else:
            data["comparisons"][0]["estimate"] = None
    change_analysis(project, change)
    with pytest.raises(WorkflowError, match="comparison"):
        assemble(project)


@pytest.mark.parametrize("where", ["claims", "numbers"])
def test_missing_evidence_pointer(project, where):
    config = project[1]
    (config["claims"] if where == "claims" else config["artifacts"]["numbers"])[0]["pointer"] = "/missing/0"
    with pytest.raises(WorkflowError, match="pointer"):
        assemble(project)


def test_null_claim_evidence(project):
    project[1]["claims"][0]["pointer"] = "/comparisons/0/reason"
    with pytest.raises(WorkflowError, match="unavailable"):
        assemble(project)


@pytest.mark.parametrize("broken", ["file", "descriptor", "pointer", "hash", "escape", "analysis"])
def test_broken_figure_rejected_and_stage_cleaned(project, monkeypatch, broken):
    def fake(analysis, specs, output_dir):
        descriptors = render_figures(analysis, specs, output_dir)
        if broken == "file":
            (output_dir / descriptors[0]["path"]).unlink()
        elif broken == "descriptor":
            return []
        elif broken == "pointer":
            descriptors[0]["evidence"][0]["pointer"] = "/absent"
        elif broken == "hash":
            descriptors[0]["sha256"] = "0" * 64
        elif broken == "escape":
            descriptors[0]["path"] = "../outside.pdf"
        else:
            descriptors[0]["evidence"][0]["analysis_id"] = "wrong"
        return descriptors
    monkeypatch.setattr(pub.figures, "render_figures", fake)
    with pytest.raises(WorkflowError):
        assemble(project)
    assert list((project[0] / "build/paper").iterdir()) == []


def test_archived_input_hash_mismatch(project):
    root, config = project
    path = Path(config["analyses"][0]["path"]) / "analysis.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(WorkflowError) as error:
        assemble(project)
    assert error.value.code == "IDENTITY"


def test_selected_fingerprint_mismatch(project):
    project[1]["analyses"][0]["sha256"] = "0" * 64
    with pytest.raises(WorkflowError, match="fingerprint"):
        assemble(project)


@pytest.mark.parametrize("name", ["status", "numbers", "artifacts"])
def test_main_includes_cannot_be_comments(project, name):
    main = project[0] / "paper/main.tex"
    main.write_text(main.read_text().replace("\\input{generated/" + name, "%\\input{generated/" + name))
    with pytest.raises(WorkflowError, match="main.tex"):
        assemble(project)


@pytest.mark.parametrize("collision", ["spec", "macro", "descriptor"])
def test_global_collisions(project, monkeypatch, collision):
    if collision == "spec":
        project[1]["artifacts"]["figures"][0]["id"] = "results"
    elif collision == "macro":
        (project[0] / "paper/section.tex").write_text("\\newcommand{\\Score}{42}")
    else:
        project[1]["artifacts"]["figures"][0]["id"] = "Score"
    with pytest.raises(WorkflowError, match="[Cc]ollision|Duplicate"):
        assemble(project)


def test_duplicate_aliases_and_unknown_alias(project):
    config = project[1]
    selection = deepcopy(config["analyses"][0])
    config["analyses"].append(selection)
    with pytest.raises(WorkflowError, match="Duplicate"):
        assemble(project)
    config["analyses"].pop()
    config["claims"][0]["analysis_alias"] = "absent"
    with pytest.raises(WorkflowError, match="alias"):
        assemble(project)


def test_two_aliases_have_distinct_relative_outputs(project):
    root, config = project
    other = make_analysis(root, name="secondary", value=0.9)
    other["alias"] = "secondary"
    config["analyses"].append(other)
    config["artifacts"]["figures"].append({**config["artifacts"]["figures"][0],
                                             "id": "secondaryplot", "analysis_alias": "secondary"})
    output = assemble(project)
    assert (output / "sources/generated/secondary/figures/pdf/secondaryplot.pdf").is_file()
    assert (output / "sources/generated/main/figures/pdf/scores.pdf").is_file()


def test_renderer_source_identity_changes_snapshot(project, monkeypatch):
    before = assemble(project)
    actual = pub._code_identity()
    altered = deepcopy(actual)
    altered["sources"]["figures"] = "f" * 64
    monkeypatch.setattr(pub, "_code_identity", lambda: altered)
    assert assemble(project) != before


def test_corrupt_existing_output_is_not_reused(project):
    before = assemble(project)
    (before / "sources/section.tex").write_text("tampered")
    with pytest.raises(WorkflowError) as error:
        assemble(project)
    assert error.value.code == "IDENTITY"


def test_renderer_mutates_selected_bundle(project, monkeypatch):
    root, config = project
    def fake(analysis, specs, output_dir):
        result = render_figures(analysis, specs, output_dir)
        path = Path(config["analyses"][0]["path"]) / "analysis.json"
        path.write_text(path.read_text() + " ")
        return result
    monkeypatch.setattr(pub.figures, "render_figures", fake)
    with pytest.raises(WorkflowError) as error:
        assemble(project)
    assert error.value.code == "IDENTITY"


def test_no_latest_selection(project):
    project[1]["analyses"][0]["path"] = "analyses/latest"
    with pytest.raises(WorkflowError, match="explicitly"):
        assemble(project)


def test_symlink_and_reserved_generated_source_rejected(project):
    root, _ = project
    link = root / "paper/link.tex"
    link.symlink_to(root / "paper/section.tex")
    with pytest.raises(WorkflowError, match="symlink"):
        assemble(project)
    link.unlink()
    generated = root / "paper/generated"
    generated.mkdir()
    (generated / "status.tex").write_text("authored")
    with pytest.raises(WorkflowError, match="reserved"):
        assemble(project)


@pytest.mark.parametrize("target", ["authored", "artifact"])
def test_later_renderer_cannot_clobber_pinned_files(project, monkeypatch, target):
    def fake(analysis, specs, output_dir):
        result = render_figures(analysis, specs, output_dir)
        if target == "artifact":
            path = output_dir.parent / "tables/nested/results.tex"
        else:
            path = output_dir.parents[2] / "section.tex"
        path.write_text("clobbered")
        return result
    monkeypatch.setattr(pub.figures, "render_figures", fake)
    with pytest.raises(WorkflowError, match="changed during assembly"):
        assemble(project)


def test_duplicate_global_number_macros_across_aliases(project):
    root, config = project
    second = make_analysis(root, name="other")
    second["alias"] = "other"
    config["analyses"].append(second)
    config["artifacts"]["numbers"].append({**config["artifacts"]["numbers"][0], "analysis_alias": "other"})
    with pytest.raises(WorkflowError, match="Duplicate number"):
        assemble(project)


def test_alias_number_wrapper_supports_multiple_renderers(project):
    root, config = project
    second = make_analysis(root, name="other")
    second["alias"] = "other"
    config["analyses"].append(second)
    config["artifacts"]["numbers"].append({**config["artifacts"]["numbers"][0],
                                            "analysis_alias": "other", "name": "OtherScore"})
    output = assemble(project)
    wrapper = (output / "sources/generated/numbers.tex").read_text()
    assert "generated/main/Score.tex" in wrapper
    assert "generated/other/OtherScore.tex" in wrapper


def test_missing_number_macro_rejected(project, monkeypatch):
    def fake(analysis, specs, output_dir):
        result = render_numbers(analysis, specs, output_dir)
        for descriptor in result:
            path = output_dir / descriptor["path"]
            path.write_text("% omitted macro\n")
            descriptor["sha256"] = sha(path)
        return result
    monkeypatch.setattr(pub.tables, "render_numbers", fake)
    with pytest.raises(WorkflowError, match="omitted a requested macro"):
        assemble(project)
