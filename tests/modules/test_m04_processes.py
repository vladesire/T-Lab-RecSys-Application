"""M04 isolated tests: real POSIX children plus explicitly mocked RSS samples."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import jsonschema
import psutil
import pytest

from researchflow import processes
from researchflow.contracts import WorkflowError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
SCHEMA = json.loads((Path(__file__).parents[2] / "src/researchflow/schemas/process_report.json").read_text())


def run(tmp_path, script, **kwargs):
    report = processes.run_process(
        [sys.executable, "-u", "-c", script], tmp_path,
        kwargs.pop("timeout_seconds", 3), Path("stdout.log"), Path("stderr.log"), **kwargs,
    )
    jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.FormatChecker()).validate(report)
    return report


def assert_pid_gone(pid):
    # kill(0) checks the actual OS rather than trusting this sandbox's /proc view.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_distinct_streams_nonzero_and_real_timing(tmp_path):
    before = datetime.now().astimezone()
    report = run(tmp_path, "import sys; print('output'); print('error', file=sys.stderr); sys.exit(17)")
    after = datetime.now().astimezone()
    assert report["status"] == "failed"
    assert report["exit_code"] == 17
    assert (tmp_path / "stdout.log").read_text() == "output\n"
    assert (tmp_path / "stderr.log").read_text() == "error\n"
    assert before <= datetime.fromisoformat(report["started_at"]) <= datetime.fromisoformat(report["finished_at"]) <= after
    assert 0 < report["elapsed_seconds"] < 3


def test_env_overlay_and_cwd_without_shell(tmp_path, monkeypatch):
    monkeypatch.delenv("M04_TEST_OVERLAY", raising=False)
    script = "import os; print(os.environ['M04_TEST_OVERLAY']); print(os.getcwd()); print(os.environ['PATH'])"
    report = run(tmp_path, script, env={"M04_TEST_OVERLAY": "$(echo unsafe); literal"})
    assert report["status"] == "succeeded"
    assert report["exit_code"] == 0
    assert (tmp_path / "stdout.log").read_text().splitlines() == ["$(echo unsafe); literal", str(tmp_path), os.environ["PATH"]]
    assert "M04_TEST_OVERLAY" not in os.environ


# This parent reaps its child on TERM, so this test can assert actual disappearance,
# even on container hosts whose PID 1 does not reap unrelated orphan zombies.
TREE = """
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def stop(*args):
    child.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
open('descendant.pid', 'w').write(str(child.pid))
print(os.getpid(), flush=True)
time.sleep(60)
"""
HEARTBEAT = """
import signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready', flush=True)
while True:
    Path('heartbeat').write_text(str(time.monotonic_ns()))
    time.sleep(0.01)
"""


def test_timeout_terminates_and_reaps_sleeping_descendant(tmp_path):
    report = run(tmp_path, TREE, timeout_seconds=0.5)
    assert report["status"] == "timed_out"
    assert report["exit_code"] == 0  # Actual handler exit; status still records timeout.
    pid = int((tmp_path / "descendant.pid").read_text())
    assert_pid_gone(pid)
    assert 0.5 <= report["elapsed_seconds"] < 3


def test_kill_escalation_with_resistant_descendant(tmp_path):
    # Parent waits for its TERM-resistant child; escalation must kill both.
    script = f"""
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-u', '-c', {HEARTBEAT!r}], stdout=subprocess.PIPE)
child.stdout.readline()
open('descendant.pid', 'w').write(str(child.pid))
def stop(*args):
    child.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
time.sleep(60)
"""
    report = run(tmp_path, script, timeout_seconds=0.5)
    assert report["status"] == "timed_out"
    # The parent may be killed during escalation, leaving a zombie for PID 1.
    # Frozen heartbeat independently verifies no descendant code still executes.
    heartbeat = (tmp_path / "heartbeat").read_text()
    time.sleep(0.06)
    assert (tmp_path / "heartbeat").read_text() == heartbeat
    assert report["elapsed_seconds"] < 3


def test_preexisting_stop_prevents_launch_without_invented_code(tmp_path):
    (tmp_path / "STOP").touch()
    report = run(tmp_path, "raise AssertionError('must never run')", stop_file=Path("STOP"))
    assert report["status"] == "cancelled"
    assert report["exit_code"] is None
    assert report["peak_rss_bytes"] is None
    assert (tmp_path / "stderr.log").read_bytes() == b""


def test_successful_leader_cannot_leave_running_background_child(tmp_path):
    script = f"""
import subprocess, sys
child = subprocess.Popen([sys.executable, '-u', '-c', {HEARTBEAT!r}], stdout=subprocess.PIPE)
child.stdout.readline()
"""
    report = run(tmp_path, script)
    assert report["status"] == "succeeded" and report["exit_code"] == 0
    assert "Leader exited with descendants" in " ".join(report["warnings"])
    heartbeat = (tmp_path / "heartbeat").read_text()
    time.sleep(0.06)
    assert (tmp_path / "heartbeat").read_text() == heartbeat


def test_live_stop_file_cancels_tree(tmp_path):
    errors = []

    def stop_when_started():
        deadline = time.monotonic() + 2
        while not (tmp_path / "descendant.pid").exists():
            if time.monotonic() > deadline:
                errors.append("child did not start")
                return
            time.sleep(0.005)
        (tmp_path / "STOP").touch()

    thread = threading.Thread(target=stop_when_started)
    thread.start()
    try:
        report = run(tmp_path, TREE, stop_file=Path("STOP"))
    finally:
        thread.join(timeout=3)
    assert not errors and not thread.is_alive()
    assert report["status"] == "cancelled"
    assert report["exit_code"] == 0  # Observed cooperative shutdown, never invented.
    assert_pid_gone(int((tmp_path / "descendant.pid").read_text()))


def test_mocked_rss_exceeds_limit_and_records_limitations(tmp_path, monkeypatch):
    # Only sampling is faked; launch, signal escalation and waiting are real.
    monkeypatch.setattr(processes, "_sample_tree", lambda root, known, warnings: 101)
    report = run(tmp_path, "import time; time.sleep(60)", memory_limit_bytes=100)
    assert report["status"] == "memory_exceeded", json.dumps(report)
    assert report["peak_rss_bytes"] == 101
    assert report["exit_code"] == -signal.SIGTERM
    warnings = " ".join(report["warnings"])
    assert "best-effort" in warnings
    assert "not OS-enforced" in warnings
    assert "unified-memory" in warnings


def test_tree_rss_sums_children_and_reports_access_denied(monkeypatch):
    class Sample:
        def __init__(self, pid, rss):
            self.pid, self.rss = pid, rss
        def is_running(self):
            return True
        def memory_info(self):
            if self.rss is None:
                raise psutil.AccessDenied(self.pid)
            from types import SimpleNamespace
            return SimpleNamespace(rss=self.rss)
        def children(self, recursive):
            assert recursive
            return [Sample(2, 30), Sample(3, None)]

    known, warnings = {}, []
    assert processes._sample_tree(Sample(1, 10), known, warnings) == 40
    assert set(known) == {2, 3}
    assert "AccessDenied" in " ".join(warnings)


def test_missing_executable_is_execution_error(tmp_path):
    with pytest.raises(WorkflowError) as caught:
        processes.run_process([str(tmp_path / "absent")], tmp_path, 1,
                              tmp_path / "out", tmp_path / "err")
    assert caught.value.code == "EXECUTION"
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("argv", [[], "echo x", [""], [123], ["bad\0executable"]])
def test_invalid_argv(tmp_path, argv):
    with pytest.raises(WorkflowError) as caught:
        processes.run_process(argv, tmp_path, 1, Path("out"), Path("err"))
    assert caught.value.code == "CONTRACT"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "1"])
def test_invalid_timeout(tmp_path, timeout):
    with pytest.raises(WorkflowError) as caught:
        run(tmp_path, "pass", timeout_seconds=timeout)
    assert caught.value.code == "CONTRACT"


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_memory_limit(tmp_path, limit):
    with pytest.raises(WorkflowError) as caught:
        run(tmp_path, "pass", memory_limit_bytes=limit)
    assert caught.value.code == "CONTRACT"


def test_logs_cannot_alias_each_other_or_stop_file(tmp_path):
    (tmp_path / "out").touch()
    os.link(tmp_path / "out", tmp_path / "alias")
    for stdout, stderr, stop in [("out", "out", None), ("out", "alias", None), ("out", "err", "out")]:
        with pytest.raises(WorkflowError) as caught:
            processes.run_process([sys.executable, "-c", "pass"], tmp_path, 1,
                                  Path(stdout), Path(stderr), stop_file=Path(stop) if stop else None)
        assert caught.value.code == "CONTRACT"
