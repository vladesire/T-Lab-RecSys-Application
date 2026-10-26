#!/usr/bin/env python3
"""Cold-start the repository with its two preserved lockfiles.

Python standard library only until both environments are installed. No engine
source, scientific settings, historical bundles or dependency locks are edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid


def python_at(project: Path) -> Path:
    return project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(argv: list[str], cwd: Path) -> None:
    print("+ " + shlex.join([str(a) for a in argv]), flush=True)
    subprocess.run([str(a) for a in argv], cwd=cwd, check=True)


def setup(root: Path, uv: str) -> tuple[Path, Path]:
    engine = root / "engines/tlab_warmup"
    for project in (root, engine):
        if not (project / "uv.lock").is_file():
            raise ValueError(f"Required preserved lockfile missing: {project / 'uv.lock'}")
    executable = shutil.which(uv)
    if executable is None:
        raise ValueError("uv was not found; install uv and rerun this command")
    run([executable, "sync", "--frozen", "--python", "3.12", "--extra", "dev"], root)
    run([executable, "sync", "--frozen", "--python", "3.12"], engine)
    return python_at(root), python_at(engine)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "data", "baseline"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--existing-data", type=Path, help="Use an existing pinned raw Parquet; do not copy it")
    parser.add_argument("--no-tex", action="store_true", help="Omit PDF compilation; all other stages still run")
    parser.add_argument("--skip-setup", action="store_true", help="Use already-installed repository environments")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        if args.skip_setup:
            infrastructure, scientific = python_at(root), python_at(root / "engines/tlab_warmup")
            if not infrastructure.is_file() or not scientific.is_file():
                raise ValueError("Run setup before --skip-setup")
        else:
            infrastructure, scientific = setup(root, args.uv)
        if args.action == "setup":
            print("Both locked environments are ready.")
            return 0
        if args.action == "baseline" and not args.no_tex:
            # Fail before downloading/fitting if PDF prerequisites are absent.
            run([infrastructure, "-c", "from pathlib import Path; from researchflow.latex import doctor; import json,sys; r=doctor(Path.cwd()); print(json.dumps(r,indent=2)); sys.exit(0 if r['passed'] else 4)"], root)
        data_command = [scientific, "-B", root / "tools/fetch_pinned_data.py", "--root", root]
        if args.existing_data:
            data_command += ["--existing-data", args.existing_data.expanduser().resolve()]
        run(data_command, root)
        data_record = json.loads((root / "data/source.json").read_text())
        raw = Path(data_record["path"])
        if args.action == "data":
            print("Verified raw data: " + str(raw))
            return 0
        run([infrastructure, "-m", "researchflow.cli", "smoke", "--root", root, "--no-tex"], root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_id, tag = "R-baseline-" + stamp, "coldstart-" + uuid.uuid4().hex[:12]
        config_dir = root / "inputs" / run_id
        run([infrastructure, root / "tools/prepare_warmup_baseline.py", "--root", root,
             "--warmup", root / "engines/tlab_warmup", "--python", scientific,
             "--data", raw, "--output", config_dir], root)
        run([infrastructure, "-m", "researchflow.cli", "experiment", "--root", root,
             "--config", config_dir / "warmup-experiment.json", "--run-id", run_id], root)
        finish = [infrastructure, root / "tools/finish_baseline_acceptance.py", "--root", root,
                  "--run", root / "runs" / run_id, "--tag", tag]
        if args.no_tex:
            finish.append("--no-tex")
        run(finish, root)
        progress = root / "build/baseline-acceptance" / tag / "progress.json"
        print("Cold-start completed. Exact output map: " + str(progress))
        print("Human visual inspection is still required for a generated PDF.")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("Cold-start stopped: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
