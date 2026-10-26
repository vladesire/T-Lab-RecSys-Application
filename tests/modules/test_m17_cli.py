"""Dispatch isolation only: all service implementations are dependency fakes."""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import pytest

from researchflow.cli import main
from researchflow.contracts import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ("experiments", "analysis", "views", "context", "publication", "latex", "freeze", "release", "smoke")
COMMANDS = {
    "doctor": [], "smoke": [],
    "experiment": ["--config", "config file.json"],
    "analyze": ["--config", "config file.json"],
    "view": ["--config", "config file.json", "--output", "output dir"],
    "review": ["--config", "config file.json", "--output", "output dir"],
    "artifacts": ["--publication", "publication file.json"],
    "freeze-study": ["--config", "config file.json", "--output", "freeze file.json"],
    "paper": ["--publication", "publication file.json"],
    "paper-check": ["--compiled", "compiled dir"],
    "release": ["--snapshot", "snapshot dir", "--compiled", "compiled dir", "--output", "output dir", "--visual-review", "visual review.json"],
}


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    calls = []
    bundle = tmp_path / "synthetic bundle"
    state = json.loads((ROOT / "tests/fixtures/schema_examples/state_view.json").read_text())

    def execute_experiment(project_root: Path, config_path: Path, *, run_id: str | None = None, freeze_path: Path | None = None) -> Path:
        calls.append(("experiment", project_root, config_path, run_id, freeze_path))
        return bundle

    def analyze(project_root: Path, config_path: Path) -> Path:
        calls.append(("analyze", project_root, config_path))
        return bundle

    def build_view(project_root: Path, config_path: Path) -> dict:
        calls.append(("build_view", project_root, config_path))
        return state

    def write_view(value, output_dir: Path) -> Path:
        assert value is state
        calls.append(("write_view", value, output_dir))
        return output_dir

    def build_packet(project_root: Path, config_path: Path, output_dir: Path) -> Path:
        calls.append(("review", project_root, config_path, output_dir))
        return output_dir

    def assemble_publication(project_root: Path, config_path: Path) -> Path:
        calls.append(("assemble", project_root, config_path))
        return bundle

    def doctor(project_root: Path) -> dict:
        calls.append(("doctor", project_root))
        return state

    def compile_snapshot(snapshot_dir: Path, output_root: Path, *, timeout_seconds: float = 120.0) -> Path:
        calls.append(("compile", snapshot_dir, output_root, timeout_seconds))
        return bundle

    def inspect_build(compiled_dir: Path) -> dict:
        calls.append(("inspect", compiled_dir))
        return state

    def freeze_study(project_root: Path, config_paths, output_path: Path) -> dict:
        calls.append(("freeze", project_root, config_paths, output_path))
        return json.loads((ROOT / "tests/fixtures/schema_examples/study_freeze.json").read_text())

    def package_release(project_root: Path, snapshot_dir: Path, compiled_dir: Path, output_dir: Path, *, visual_review: Path | None = None) -> Path:
        calls.append(("release", project_root, snapshot_dir, compiled_dir, output_dir, visual_review))
        return output_dir

    def smoke(project_root: Path, *, compile_pdf: bool = True) -> dict:
        calls.append(("smoke", project_root, compile_pdf))
        return state

    exports = {
        "experiments": {"execute_experiment": execute_experiment},
        "analysis": {"analyze": analyze},
        "views": {"build_view": build_view, "write_view": write_view},
        "context": {"build_packet": build_packet},
        "publication": {"assemble_publication": assemble_publication},
        "latex": {"doctor": doctor, "compile_snapshot": compile_snapshot, "inspect_build": inspect_build},
        "freeze": {"freeze_study": freeze_study},
        "release": {"package_release": package_release}, "smoke": {"smoke": smoke},
    }
    for name, symbols in exports.items():
        module = types.ModuleType("researchflow." + name)
        module.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls, bundle, state


@pytest.mark.parametrize("command", COMMANDS)
def test_exact_dispatch(command, fakes, tmp_path, capsys):
    calls, bundle, state = fakes
    root = tmp_path / "project root"
    expected = {
        "doctor": [("doctor", root)], "smoke": [("smoke", root, True)],
        "experiment": [("experiment", root, root / "config file.json", None, None)],
        "analyze": [("analyze", root, root / "config file.json")],
        "view": [("build_view", root, root / "config file.json"), ("write_view", state, root / "output dir")],
        "review": [("review", root, root / "config file.json", root / "output dir")],
        "artifacts": [("assemble", root, root / "publication file.json")],
        "freeze-study": [("freeze", root, [root / "config file.json"], root / "freeze file.json")],
        "paper": [("assemble", root, root / "publication file.json"), ("compile", bundle, root / "build/compiled", 120.0)],
        "paper-check": [("inspect", root / "compiled dir")],
        "release": [("release", root, root / "snapshot dir", root / "compiled dir", root / "output dir", root / "visual review.json")],
    }
    assert main([command, "--root", str(root), *COMMANDS[command]]) == 0
    assert calls == expected[command]
    captured = capsys.readouterr()
    assert not captured.err
    assert captured.out.strip()
    if command in ("doctor", "smoke", "paper-check", "freeze-study"):
        expected_summary = json.loads((ROOT / "tests/fixtures/schema_examples/study_freeze.json").read_text()) if command == "freeze-study" else state
        assert json.loads(captured.out) == expected_summary
    else:
        output = root / "output dir" if command in ("view", "review", "release") else bundle
        assert captured.out.strip() == str(output)


@pytest.mark.parametrize("argv", [["--help"]] + [[c, "--help"] for c in COMMANDS])
def test_help_does_not_import_services(monkeypatch, argv):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"researchflow." + s for s in SERVICES} or name in {"numpy", "matplotlib", "yaml"}:
            raise AssertionError("Optional computation import during parsing: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert main(argv) == 0


@pytest.mark.parametrize("command", COMMANDS)
def test_all_commands_parse_before_missing_dependency(monkeypatch, command, capsys):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name in {"researchflow." + s for s in SERVICES}:
            raise ModuleNotFoundError("synthetic missing service: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert main([command, *COMMANDS[command]]) == 4
    captured = capsys.readouterr()
    assert "DEPENDENCY" in captured.err and "synthetic missing service" in captured.err
    assert not captured.out


def test_options_root_precedence_absolute_paths_and_freezes(fakes, tmp_path, monkeypatch):
    calls, _, _ = fakes
    monkeypatch.chdir(tmp_path)
    assert main(["--root", "relative root", "experiment", "--config", "c.json", "--run-id", "R-fixed", "--freeze", "f.json"]) == 0
    root = tmp_path / "relative root"
    assert calls.pop() == ("experiment", root, root / "c.json", "R-fixed", root / "f.json")
    absolute = tmp_path / "outside.json"
    assert main(["--root", "ignored root", "freeze-study", "--root", str(root), "--config", "a.json", "--config", str(absolute), "--output", "freeze.json"]) == 0
    assert calls.pop() == ("freeze", root, [root / "a.json", absolute], root / "freeze.json")
    assert main(["smoke", "--no-tex"]) == 0
    assert calls.pop() == ("smoke", tmp_path, False)


@pytest.mark.parametrize("code,expected", [("CONTRACT", 2), ("IDENTITY", 2), ("EXECUTION", 3), ("DEPENDENCY", 4), ("FUTURE_CODE", 1)])
def test_workflow_failure_is_nonzero(code, expected, fakes, monkeypatch, tmp_path, capsys):
    def failed(*args, **kwargs):
        raise WorkflowError(code, "job failed", {"evidence": str(tmp_path / "failure logs")})

    monkeypatch.setattr(sys.modules["researchflow.experiments"], "execute_experiment", failed)
    assert main(["experiment", "--config", "c.json"]) == expected
    captured = capsys.readouterr()
    assert not captured.out
    assert code in captured.err and "job failed" in captured.err
    assert str(tmp_path / "failure logs") in captured.err


@pytest.mark.parametrize("stage", ["assemble_publication", "compile_snapshot"])
def test_paper_stops_on_failure_and_never_launches_experiment(stage, fakes, monkeypatch, capsys):
    calls, _, _ = fakes

    def failed(*args, **kwargs):
        raise WorkflowError("EXECUTION", "synthetic paper failure")

    module = "publication" if stage == "assemble_publication" else "latex"
    monkeypatch.setattr(sys.modules["researchflow." + module], stage, failed)
    assert main(["paper", "--publication", "p.json"]) == 3
    assert [call[0] for call in calls] == ([] if module == "publication" else ["assemble"])
    assert not capsys.readouterr().out


def test_unknown_failure(fakes, monkeypatch, capsys):
    def broken(*args):
        raise RuntimeError("synthetic unexpected error")

    monkeypatch.setattr(sys.modules["researchflow.analysis"], "analyze", broken)
    assert main(["analyze", "--config", "c.json"]) == 1
    assert "RuntimeError: synthetic unexpected error" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["unknown"], ["experiment"], ["release", "--snapshot", "x"], ["doctor", "--config", "x"]])
def test_invalid_cli_returns_two(argv, fakes):
    assert main(argv) == 2
    assert not fakes[0]


MAKE_ARGS = {
    "doctor": [], "smoke": [],
    "experiment": ["--config", "config path", "--run-id", "R fixed", "--freeze", "freeze path"],
    "analyze": ["--config", "config path"],
    "view": ["--config", "config path", "--output", "output path"],
    "review": ["--config", "config path", "--output", "output path"],
    "artifacts": ["--publication", "publication path"],
    "freeze-study": ["--config", "config path", "--output", "output path"],
    "paper": ["--publication", "publication path"],
    "paper-check": ["--compiled", "compiled path"],
    "release": ["--snapshot", "snapshot path", "--compiled", "compiled path", "--output", "output path", "--visual-review", "review path"],
}
MAKE_VARS = ["PROJECT_ROOT=project root", "UV=uv path", "CONFIG=config path", "OUTPUT=output path", "RUN_ID=R fixed", "FREEZE=freeze path", "PUBLICATION=publication path", "COMPILED=compiled path", "SNAPSHOT=snapshot path", "VISUAL_REVIEW=review path"]


@pytest.mark.parametrize("target", MAKE_ARGS)
def test_make_dry_runs_preserve_arguments(target):
    result = subprocess.run(["make", "--no-print-directory", "-n", target, *MAKE_VARS], cwd=ROOT, text=True, capture_output=True, check=True)
    lines = [shlex.split(line) for line in result.stdout.splitlines() if " run researchflow " in line]
    assert lines == [["uv path", "run", "researchflow", target, "--root", "project root", *MAKE_ARGS[target]]]
    if target == "paper":
        assert "experiment" not in result.stdout and "review" not in result.stdout


@pytest.mark.parametrize("target,variables", [("experiment", ["CONFIG"]), ("analyze", ["CONFIG"]), ("view", ["CONFIG", "OUTPUT"]), ("review", ["CONFIG", "OUTPUT"]), ("artifacts", ["PUBLICATION"]), ("freeze-study", ["CONFIG", "OUTPUT"]), ("paper", ["PUBLICATION"]), ("paper-check", ["COMPILED"]), ("release", ["SNAPSHOT", "COMPILED", "OUTPUT", "VISUAL_REVIEW"])])
def test_make_missing_each_required_variable_fails(target, variables):
    for missing in variables:
        # Clear inherited make flags; explicit assignments override any exported variables.
        env = {**os.environ, "MAKEFLAGS": ""}
        assignments = [name + ("=" if name == missing else "=valid path") for name in variables]
        result = subprocess.run(["make", "--no-print-directory", target, "UV=/must/not/launch", *assignments], cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode != 0
        assert "Missing " + missing in result.stderr
        assert "/must/not/launch" not in result.stdout


@pytest.mark.parametrize("inherited_uv", ["uv", "/opt/homebrew/bin/uv"])
def test_make_default_help_and_shell_quoting(tmp_path, inherited_uv):
    env = {**os.environ, "UV": inherited_uv, "MAKEFLAGS": "", "MFLAGS": "", "MAKEOVERRIDES": ""}
    help_result = subprocess.run(["make", "--no-print-directory"], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "make TARGET" in help_result.stdout
    assert "uv run" not in help_result.stdout
    tricky = "path with 'quotes' ; echo injected"
    result = subprocess.run(["make", "--no-print-directory", "-n", "paper", "PUBLICATION=" + tricky], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    line = next(line for line in result.stdout.splitlines() if " run researchflow " in line)
    # Exercise a real POSIX shell with a fake executable; no uv or service is run.
    recorder = tmp_path / "uv recorder"
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    recorder.chmod(0o755)
    prefix = "'" + inherited_uv + "'"
    assert line.startswith(prefix + " ")
    line = shlex.quote(str(recorder)) + line[len(prefix):]
    actual = subprocess.run(["/bin/sh", "-c", line], text=True, capture_output=True, check=True)
    assert actual.stdout.splitlines() == ["run", "researchflow", "paper", "--root", str(ROOT), "--publication", tricky]


def test_make_propagates_failed_cli_status(tmp_path):
    executable = tmp_path / "failing uv"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "synthetic dependency failed" >&2\nexit 4\n')
    executable.chmod(0o755)
    result = subprocess.run(["make", "--no-print-directory", "doctor", "UV=" + str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "synthetic dependency failed" in result.stderr
