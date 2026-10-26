"""Sequential POSIX subprocess execution with bounded, sampled supervision."""
from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from researchflow.contracts import WorkflowError

_POLL_SECONDS = 0.02
_TERM_GRACE_SECONDS = 0.3
_KILL_GRACE_SECONDS = 0.3
_RSS_LIMITATIONS = (
    "Process-tree RSS is best-effort psutil sampling: short peaks and descendants "
    "that exit, reparent, or escape between samples may be missed; inaccessible "
    "processes are omitted. Summed RSS can double-count shared pages and excludes "
    "swapped and some accelerator allocations. This is not OS-enforced memory "
    "isolation and does not measure total unified-memory pressure."
)


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _sample_tree(root: psutil.Process | None, known: dict[int, psutil.Process],
                 warnings: list[str]) -> int | None:
    """Track identities as Process objects, so psutil guards signals against reuse."""
    if root is None:
        return None
    try:
        for child in root.children(recursive=True):
            known[child.pid] = child
    except psutil.Error as exc:
        _warn(warnings, f"Process-tree discovery incomplete: {type(exc).__name__}.")
    rss = 0
    measured = False
    for process in [root, *known.values()]:
        try:
            if process.is_running():
                rss += process.memory_info().rss
                measured = True
        except psutil.Error as exc:
            _warn(warnings, f"RSS sample incomplete: {type(exc).__name__}.")
    return rss if measured else None


def _signal_group(pid: int, sig: int, warnings: list[str]) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        _warn(warnings, f"Process-group signal {sig} failed: {exc}.")


def _signal_known(known: dict[int, psutil.Process], sig: int,
                  warnings: list[str]) -> None:
    for process in known.values():
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            _warn(warnings, f"Descendant signal {sig} failed: {type(exc).__name__}.")


def _terminate(process: subprocess.Popen, root: psutil.Process | None,
               known: dict[int, psutil.Process], warnings: list[str]) -> None:
    # Do not stop cleanup when the leader exits: its group can still have children.
    if root is not None:
        _sample_tree(root, known, warnings)
    _signal_group(process.pid, signal.SIGTERM, warnings)
    _signal_known(known, signal.SIGTERM, warnings)
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if root is not None:
            _sample_tree(root, known, warnings)
        time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
    _signal_group(process.pid, signal.SIGKILL, warnings)
    _signal_known(known, signal.SIGKILL, warnings)
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _warn(warnings, "Leader did not become waitable within the kill grace period.")
    # Only the direct child can be reaped here. Other zombies need their OS parent.
    try:
        _, alive = psutil.wait_procs(list(known.values()), timeout=_KILL_GRACE_SECONDS)
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
            _warn(warnings, "Some observed descendants survived the bounded cleanup.")
        if any(p.is_running() and p.status() == psutil.STATUS_ZOMBIE for p in alive):
            _warn(warnings, "Dead descendant zombies await reaping by their OS parent.")
    except psutil.Error as exc:
        _warn(warnings, f"Descendant cleanup verification incomplete: {type(exc).__name__}.")


def run_process(argv: Sequence[str], cwd: Path, timeout_seconds: float, stdout_path: Path, stderr_path: Path, env: Mapping[str, str] | None = None, *, stop_file: Path | None = None, memory_limit_bytes: int | None = None) -> dict[str, Any]:
    """Return the v1 process report; cancellation exit codes are observed or null.

    Relative log/stop paths resolve against cwd. Existing logs are overwritten;
    their parent directories must exist. A pre-existing stop file prevents launch.
    """
    if (isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence)
            or not argv or any(not isinstance(a, str) or '\0' in a for a in argv)
            or not argv[0]):
        raise WorkflowError("CONTRACT", "argv must be a nonempty sequence of strings with a nonempty executable and no NULs")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise WorkflowError("CONTRACT", "timeout_seconds must be finite and positive")
    if memory_limit_bytes is not None and (
        isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
    ):
        raise WorkflowError("CONTRACT", "memory_limit_bytes must be a positive integer")
    if env is not None and (not isinstance(env, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or '=' in k
        or '\0' in k or '\0' in v for k, v in env.items()
    )):
        raise WorkflowError("CONTRACT", "env must map valid environment names to strings")
    if os.name != "posix":
        raise WorkflowError("DEPENDENCY", "Process-group supervision requires macOS/Linux POSIX semantics")

    cwd = Path(cwd).resolve()

    def resolve(path: Path) -> Path:
        return (cwd / path).resolve()

    stdout_path, stderr_path = resolve(stdout_path), resolve(stderr_path)
    stop_file = resolve(stop_file) if stop_file is not None else None
    if stdout_path == stderr_path or (
        stdout_path.exists() and stderr_path.exists()
        and os.path.samefile(stdout_path, stderr_path)
    ):
        raise WorkflowError("CONTRACT", "stdout_path and stderr_path must be distinct files")
    if stop_file is not None and any(
        stop_file == path or (stop_file.exists() and path.exists()
                             and os.path.samefile(stop_file, path))
        for path in (stdout_path, stderr_path)
    ):
        raise WorkflowError("CONTRACT", "stop_file must be distinct from the log files")

    child_env = os.environ.copy()
    if env is not None:
        child_env.update(env)
    warnings = [_RSS_LIMITATIONS]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    process = None
    root = None
    known: dict[int, psutil.Process] = {}
    peak = None
    status = "cancelled"
    cleaned = False
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            if stop_file is None or not stop_file.exists():
                process = subprocess.Popen(
                    list(argv), cwd=cwd, env=child_env, stdout=stdout, stderr=stderr,
                    stdin=subprocess.DEVNULL, shell=False, start_new_session=True,
                )
                try:
                    root = psutil.Process(process.pid)
                except psutil.Error as exc:
                    _warn(warnings, f"Process observation unavailable: {type(exc).__name__}.")
                while True:
                    if root is None and process.poll() is None:
                        # Process metadata can briefly be unavailable immediately
                        # after spawn. Retry observation, never the actual launch.
                        try:
                            root = psutil.Process(process.pid)
                        except psutil.Error as exc:
                            _warn(warnings, f"Process observation unavailable: {type(exc).__name__}.")
                    rss = _sample_tree(root, known, warnings)
                    if rss is not None:
                        peak = rss if peak is None else max(peak, rss)
                    code = process.poll()
                    if code is not None:
                        status = "succeeded" if code == 0 else "failed"
                        break
                    if stop_file is not None and stop_file.exists():
                        status = "cancelled"
                    elif time.monotonic() - start >= timeout_seconds:
                        status = "timed_out"
                    elif memory_limit_bytes is not None and peak is not None and peak > memory_limit_bytes:
                        status = "memory_exceeded"
                    else:
                        time.sleep(min(_POLL_SECONDS, max(0, timeout_seconds - (time.monotonic() - start))))
                        continue
                    _terminate(process, root, known, warnings)
                    cleaned = True
                    break
                # A leader that exits normally must not leave background workers.
                if not cleaned:
                    group_exists = False
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        group_exists = True
                    if group_exists or any(p.is_running() for p in known.values()):
                        _warn(warnings, "Leader exited with descendants still present; cleaned up observed descendants and the process group.")
                        _terminate(process, root, known, warnings)
                        cleaned = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise WorkflowError("EXECUTION", f"Process launch or supervision failed: {exc}",
                            {"argv": list(argv), "cwd": str(cwd)}) from exc
    finally:
        if process is not None and not cleaned and process.poll() is None:
            _terminate(process, root, known, warnings)
    if process is not None and peak is None:
        _warn(warnings, "No RSS sample was available; any configured memory limit could not be evaluated.")
    return {
        "schema_version": 1, "argv": list(argv), "status": status,
        "exit_code": process.returncode if process is not None else None,
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start, "peak_rss_bytes": peak,
        "warnings": warnings,
    }
