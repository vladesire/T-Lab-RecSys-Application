"""Isolated M14 tests: M01/M03/M04 are contract-shaped fakes."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import pytest

from researchflow import latex
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
MINIMAL_TEX = r"""\documentclass{article}
\begin{document}
Synthetic infrastructure test. No research result is asserted.
\end{document}
"""


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": _hash(p), "size_bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p != root / "manifest.json"]


@pytest.fixture
def dependencies(monkeypatch):
    def validate(kind, value):
        schema = json.loads((ROOT / f"src/researchflow/schemas/{kind}.json").read_text())
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)

    def load(path):
        return json.loads(path.read_text())

    def write(path, value):
        path.write_text(json.dumps(value, indent=2))

    def contained(root, relative):
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
            raise WorkflowError("CONTRACT", "Escaping path")
        return path

    def verify(bundle_dir: Path, expected_kind: str | None = None) -> dict[str, Any]:
        manifest = load(bundle_dir / "manifest.json")
        validate(manifest["bundle_kind"] + "_manifest", manifest)
        assert manifest["bundle_kind"] == expected_kind
        if _inventory(bundle_dir) != manifest["files"]:
            raise WorkflowError("IDENTITY", "Fake verifier detected tampering")
        return manifest

    def begin(output_root: Path, bundle_id: str) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        stage = output_root / (".stage-" + uuid.uuid4().hex)
        stage.mkdir()
        return stage

    def commit(stage_dir: Path, final_dir: Path, manifest: Mapping[str, Any]) -> Path:
        assert _inventory(stage_dir) == manifest["files"]
        write(stage_dir / "manifest.json", manifest)
        stage_dir.rename(final_dir)
        return final_dir

    def fingerprint(manifest):
        return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    calls = []

    def run(argv: Sequence[str], cwd: Path, timeout_seconds: float, stdout_path: Path,
            stderr_path: Path, env: Mapping[str, str] | None = None, *, stop_file: Path | None = None,
            memory_limit_bytes: int | None = None) -> dict[str, Any]:
        calls.append(list(argv))
        report = json.loads((ROOT / "tests/fixtures/schema_examples/process_report.json").read_text())
        report.update(argv=list(argv), warnings=[])
        try:
            completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout_seconds)
            stdout_path.write_bytes(completed.stdout)
            stderr_path.write_bytes(completed.stderr)
            report.update(exit_code=completed.returncode, status="succeeded" if completed.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            stdout_path.write_text("")
            stderr_path.write_text("Test fake process timed out")
            report.update(exit_code=None, status="timed_out")
        return report

    for name, function in {"load_document": load, "atomic_write_json": write, "validate_document": validate,
                           "contained_path": contained, "sha256_file": _hash}.items():
        monkeypatch.setattr(latex.documents, name, function)
    for name, function in {"verify_bundle": verify, "begin_bundle": begin, "commit_bundle": commit,
                           "bundle_fingerprint": fingerprint}.items():
        monkeypatch.setattr(latex.bundles, name, function)
    monkeypatch.setattr(latex.processes, "run_process", run)
    return calls


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "publication"
    (root / "sources").mkdir(parents=True)
    (root / "sources/main.tex").write_text(MINIMAL_TEX)
    manifest = json.loads((ROOT / "tests/fixtures/schema_examples/publication_manifest.json").read_text())
    manifest["files"] = _inventory(root)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    bindir = tmp_path / "fake bin 'quoted'"
    bindir.mkdir()
    payloads = bindir / "payloads"
    payloads.mkdir()
    # Python interpreter paths cannot be quoted portably in a shebang. Use a
    # short /bin/sh shebang and quote the command arguments instead. Exercise
    # whitespace, apostrophes and an interpreter path longer than 127 bytes in
    # every fake-tools test, including on macOS with long temporary paths.
    interpreter = tmp_path / ("python env 'quoted' " + "x" * 80) / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    assert len(os.fsencode(interpreter)) > 127
    sty = tmp_path / "fake.sty"
    sty.write_text("% synthetic package")
    script = r'''
import os, pathlib, sys, time
name = pathlib.Path(sys.argv[0]).name
if '--version' in sys.argv:
    print(name + ' synthetic 1.0')
    sys.exit(0)
if name == 'kpsewhich':
    if os.getenv('M14_MISSING_STY'):
        sys.exit(1)
    print(os.environ['M14_STY_PATH'])
    sys.exit(0)
assert name == 'latexmk'
assert '-norc' in sys.argv and '-no-shell-escape' in sys.argv
out = pathlib.Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('-outdir=')))
out.mkdir(exist_ok=True)
(out/'main.log').write_text(os.getenv('M14_LOG', 'Output written on main.pdf (1 page).'))
(out/'main.fls').write_text('INPUT main.tex\nOUTPUT main.pdf\n')
(out/'main.bbl').write_text('% synthetic bibliography')
(out/'main.blg').write_text('synthetic bibliography log')
pathlib.Path('working-copy-only.aux').write_text('TeX may write in cwd')
if os.getenv('M14_SLEEP'):
    time.sleep(2)
if not os.getenv('M14_NO_PDF'):
    (out/'main.pdf').write_bytes(b'%PDF-1.4\nSynthetic fake PDF, not visually valid.\n%%EOF')
print(os.getenv('M14_STDOUT', 'synthetic stdout'))
print(os.getenv('M14_STDERR', 'synthetic stderr'), file=sys.stderr)
sys.exit(int(os.getenv('M14_EXIT', '0')))
'''
    for name in latex._TOOLS:
        path = bindir / name
        payload = payloads / name
        payload.write_text(script)
        path.write_text(
            "#!/bin/sh\nexec " + shlex.quote(str(interpreter)) + " "
            + shlex.quote(str(payload)) + ' "$@"\n'
        )
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("M14_STY_PATH", str(sty))
    return bindir


def test_doctor_reports_versions_paths_and_packages(snapshot, dependencies, fake_tools):
    (snapshot / "sources/main.tex").write_text(r"\usepackage[option]{alpha, beta}" + "\n% \\usepackage{ignored}")
    result = latex.doctor(snapshot)
    assert result["passed"], result
    assert set(result["packages"]) == {"alpha.sty", "beta.sty"}
    assert all(t["version"].endswith("synthetic 1.0") and Path(t["path"]).is_file() for t in result["tools"].values())


def test_missing_tool_no_installer(snapshot, tmp_path, dependencies, fake_tools):
    (fake_tools / "latexmk").unlink()
    assert "latexmk" in latex.doctor(snapshot)["missing"]
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "compiled")
    assert error.value.code == "DEPENDENCY"
    assert all(Path(call[0]).name in latex._TOOLS for call in dependencies)
    assert not (tmp_path / "compiled").exists()


def test_missing_package(snapshot, tmp_path, dependencies, fake_tools, monkeypatch):
    (snapshot / "sources/main.tex").write_text(r"\usepackage{absent}")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["files"] = _inventory(snapshot)
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("M14_MISSING_STY", "1")
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "out")
    assert error.value.code == "DEPENDENCY"
    assert "absent.sty" in error.value.details["missing"]


def test_success_exact_snapshot_immutable_evidence_and_reuse(snapshot, tmp_path, dependencies, fake_tools):
    before = {p.relative_to(snapshot).as_posix(): _hash(p) for p in snapshot.rglob("*") if p.is_file()}
    expected = latex.bundles.bundle_fingerprint(json.loads((snapshot / "manifest.json").read_text()))
    compiled = latex.compile_snapshot(snapshot, tmp_path / "out")
    manifest = json.loads((compiled / "manifest.json").read_text())
    assert manifest["metadata"]["snapshot_sha256"] == expected
    assert compiled.name == expected[:32]
    assert manifest["metadata"]["pdf_sha256"] == _hash(compiled / "build/main.pdf")
    assert manifest["metadata"]["toolchain"]["command"][1:5] == ['-norc', '-pdf', '-interaction=nonstopmode', '-halt-on-error']
    for relative in ("build/main.fls", "build/main.bbl", "build/main.blg", "logs/compile.stdout", "logs/compile.stderr", "input/sources/working-copy-only.aux"):
        assert (compiled / relative).is_file()
    assert before == {p.relative_to(snapshot).as_posix(): _hash(p) for p in snapshot.rglob("*") if p.is_file()}
    checks = latex.inspect_build(compiled)
    assert checks["passed"]
    assert next(i for i in checks["items"] if i["id"] == "visual_review")["status"] == "unverified"
    count = len(dependencies)
    assert latex.compile_snapshot(snapshot, tmp_path / "out") == compiled
    assert len(dependencies) == count


@pytest.mark.parametrize("log,check", [
    ("! Undefined control sequence.\nl.4 broken", "latex_errors"),
    ("LaTeX Warning: Citation `x' on page 1 undefined on input line 4.", "citations"),
    ("Package natbib Warning: There were undefined citations.", "citations"),
    ("LaTeX Warning: Reference `fig:x' on page 1 undefined on input line 4.", "references"),
    ("LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.", "references"),
    ("! LaTeX Error: File `missing.png' not found.", "missing_assets"),
    (r"Overfull \hbox (10.0pt too wide) at lines 1--2", "overflow"),
    ("Package biblatex Warning: Please (re)run Biber on the file:", "citations"),
])
def test_log_diagnostics_fail_compile_and_inspection(log, check, snapshot, tmp_path, dependencies, fake_tools, monkeypatch):
    # First create a verified fake build, then construct an independently failed
    # bundle with its inventory updated: inspect must recompute, not trust checks.
    compiled = latex.compile_snapshot(snapshot, tmp_path / "clean")
    (compiled / "build/main.log").write_text(log)
    manifest = json.loads((compiled / "manifest.json").read_text())
    manifest["files"] = _inventory(compiled)
    (compiled / "manifest.json").write_text(json.dumps(manifest))
    checks = latex.inspect_build(compiled)
    assert not checks["passed"]
    item = next(i for i in checks["items"] if i["id"] == check)
    assert item["status"] == "failed" and "main.log:1:" in item["evidence"][0]
    monkeypatch.setenv("M14_LOG", log)
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "bad")
    assert error.value.code == "EXECUTION"
    stage = Path(error.value.details["diagnostics_dir"])
    assert (stage / "process.json").is_file() and not (stage / "manifest.json").exists()


@pytest.mark.parametrize("variable,value", [("M14_EXIT", "2"), ("M14_NO_PDF", "1"), ("M14_SLEEP", "1")])
def test_failed_missing_pdf_timeout_never_publish(variable, value, snapshot, tmp_path, dependencies, fake_tools, monkeypatch):
    monkeypatch.setenv(variable, value)
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "out", timeout_seconds=0.2)
    assert error.value.code == "EXECUTION"
    assert not list((tmp_path / "out").glob("*/manifest.json"))
    assert (Path(error.value.details["diagnostics_dir"]) / "checks.json").is_file()


def test_tampered_snapshot_rejected(snapshot, tmp_path, dependencies, fake_tools):
    (snapshot / "sources/main.tex").write_text("changed")
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "out")
    assert error.value.code == "IDENTITY" and not dependencies


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(snapshot, tmp_path, dependencies, timeout):
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "out", timeout_seconds=timeout)
    assert error.value.code == "CONTRACT"


def test_output_inside_snapshot_rejected(snapshot, dependencies):
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, snapshot / "out")
    assert error.value.code == "CONTRACT"


def test_real_tex_opt_in(snapshot, tmp_path, dependencies):
    if os.environ.get("RESEARCHFLOW_REAL_TEX") != "1":
        pytest.skip("Opt-in actual TeX test: set RESEARCHFLOW_REAL_TEX=1 on the Mac; visual review remains manual")
    missing = [tool for tool in latex._TOOLS if not shutil.which(tool)]
    if missing:
        pytest.skip("Actual TeX unavailable: " + ", ".join(missing))
    compiled = latex.compile_snapshot(snapshot, tmp_path / "real-compiled")
    assert latex.inspect_build(compiled)["passed"]
    # A human must open this exact PDF; successful TeX execution is not visual QA.
    print("Manual visual inspection required:", compiled / "build/main.pdf")


def test_resolved_first_pass_warnings_do_not_fail(snapshot, tmp_path, dependencies, fake_tools, monkeypatch):
    early = "LaTeX Warning: Citation `source' undefined.\nNo file main.bbl.\nLaTeX Warning: There were undefined references.\nOverfull \\hbox (10pt too wide)"
    monkeypatch.setenv("M14_STDOUT", early)
    monkeypatch.setenv("M14_STDERR", "No file main.bbl.")
    compiled = latex.compile_snapshot(snapshot, tmp_path / "resolved")
    assert early in (compiled / "logs/compile.stdout").read_text()
    assert latex.inspect_build(compiled)["passed"]


def test_missing_final_log_cannot_be_replaced_by_transcript(tmp_path, dependencies):
    (tmp_path / "build").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "build/main.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (tmp_path / "logs/compile.stdout").write_text("Output written on main.pdf")
    checks = latex._checks(tmp_path, "build/main.pdf", {"status": "succeeded", "exit_code": 0})
    assert not checks["passed"]
    assert next(i for i in checks["items"] if i["id"] == "log")["status"] == "failed"


def test_final_bibliography_error_remains_a_failure(tmp_path, dependencies):
    (tmp_path / "build").mkdir()
    (tmp_path / "build/main.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (tmp_path / "build/main.log").write_text("Output written on main.pdf")
    (tmp_path / "build/main.blg").write_text("I couldn't open database file references.bib")
    checks = latex._checks(tmp_path, "build/main.pdf", {"status": "succeeded", "exit_code": 0})
    assert not checks["passed"]
    citations = next(i for i in checks["items"] if i["id"] == "citations")
    assert citations["status"] == "failed"
    assert "main.blg:1:" in citations["evidence"][0]


def test_real_tex_resolves_bibliography_across_passes(snapshot, tmp_path, dependencies):
    if os.environ.get("RESEARCHFLOW_REAL_TEX") != "1":
        pytest.skip("Set RESEARCHFLOW_REAL_TEX=1 for actual bibliography compilation")
    missing = [tool for tool in latex._TOOLS if not shutil.which(tool)]
    if missing:
        pytest.skip("Actual TeX unavailable: " + ", ".join(missing))
    (snapshot / "sources/main.tex").write_text(r"""\documentclass{article}
\begin{document}
Synthetic bibliography test \cite{fixture}.
\bibliographystyle{plain}
\bibliography{references}
\end{document}
""")
    (snapshot / "sources/references.bib").write_text("@misc{fixture,author={Fixture Author},title={Synthetic Test},year={2026}}\n")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["files"] = _inventory(snapshot)
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    compiled = latex.compile_snapshot(snapshot, tmp_path / "bibliography")
    assert latex.inspect_build(compiled)["passed"]
    transcript = (compiled / "logs/compile.stdout").read_text()
    assert "undefined" in transcript.lower() or "No file main.bbl" in transcript
    assert "Fixture Author" in (compiled / "build/main.bbl").read_text()


def test_pdf_identity_and_failed_status_rechecked(snapshot, tmp_path, dependencies, fake_tools):
    compiled = latex.compile_snapshot(snapshot, tmp_path / "out")
    manifest = json.loads((compiled / "manifest.json").read_text())
    manifest["metadata"]["pdf_sha256"] = "a" * 64
    manifest["status"] = "failed"
    (compiled / "manifest.json").write_text(json.dumps(manifest))
    checks = latex.inspect_build(compiled)
    assert not checks["passed"]
    failed = {i["id"] for i in checks["items"] if i["status"] == "failed"}
    assert {"pdf_identity", "bundle_status"} <= failed


def test_snapshot_mutation_during_process_detected(snapshot, tmp_path, dependencies, fake_tools, monkeypatch):
    original = latex.processes.run_process

    def mutate(argv: Sequence[str], cwd: Path, timeout_seconds: float, stdout_path: Path,
               stderr_path: Path, env: Mapping[str, str] | None = None, *, stop_file: Path | None = None,
               memory_limit_bytes: int | None = None) -> dict[str, Any]:
        report = original(argv, cwd, timeout_seconds, stdout_path, stderr_path, env,
                          stop_file=stop_file, memory_limit_bytes=memory_limit_bytes)
        if "-pdf" in argv:
            (snapshot / "sources/main.tex").write_text("external mutation")
        return report

    monkeypatch.setattr(latex.processes, "run_process", mutate)
    with pytest.raises(WorkflowError) as error:
        latex.compile_snapshot(snapshot, tmp_path / "out")
    assert error.value.code == "IDENTITY"
    assert not list((tmp_path / "out").glob("*/manifest.json"))
