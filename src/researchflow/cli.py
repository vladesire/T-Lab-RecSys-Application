"""Thin command composition; service imports occur only after argument parsing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from researchflow.contracts import WorkflowError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchflow", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: cwd)")
    commands = parser.add_subparsers(dest="command", required=True)
    options = {
        "doctor": (), "smoke": (), "experiment": ("config",),
        "analyze": ("config",), "view": ("config", "output"),
        "review": ("config", "output"), "artifacts": ("publication",),
        "freeze-study": ("config", "output"), "paper": ("publication",),
        "paper-check": ("compiled",),
        "release": ("snapshot", "compiled", "output", "visual-review"),
    }
    for name, required in options.items():
        command = commands.add_parser(name)
        # SUPPRESS preserves a root supplied before the subcommand.
        command.add_argument("--root", type=Path, default=argparse.SUPPRESS,
                             help="Project root (default: cwd)")
        for option in required:
            kwargs = {"action": "append"} if name == "freeze-study" and option == "config" else {}
            command.add_argument("--" + option, type=Path, required=True, **kwargs)
        if name == "experiment":
            command.add_argument("--run-id")
            command.add_argument("--freeze", type=Path)
        elif name == "smoke":
            command.add_argument("--no-tex", action="store_true")
    return parser


def _dispatch(args: argparse.Namespace, root: Path) -> Any:
    def path(value: Path) -> Path:
        return (root / value).resolve()

    if args.command == "doctor":
        from researchflow.latex import doctor
        return doctor(root)
    if args.command == "smoke":
        from researchflow.smoke import smoke
        return smoke(root, compile_pdf=not args.no_tex)
    if args.command == "experiment":
        from researchflow.experiments import execute_experiment
        return execute_experiment(root, path(args.config), run_id=args.run_id,
                                  freeze_path=path(args.freeze) if args.freeze is not None else None)
    if args.command == "analyze":
        from researchflow.analysis import analyze
        return analyze(root, path(args.config))
    if args.command == "view":
        from researchflow.views import build_view, write_view
        return write_view(build_view(root, path(args.config)), path(args.output))
    if args.command == "review":
        from researchflow.context import build_packet
        return build_packet(root, path(args.config), path(args.output))
    if args.command == "freeze-study":
        from researchflow.freeze import freeze_study
        return freeze_study(root, [path(config) for config in args.config], path(args.output))
    if args.command in ("artifacts", "paper"):
        from researchflow.publication import assemble_publication
        snapshot = assemble_publication(root, path(args.publication))
        if args.command == "artifacts":
            return snapshot
        from researchflow.latex import compile_snapshot
        return compile_snapshot(snapshot, root / "build" / "compiled")
    if args.command == "paper-check":
        from researchflow.latex import inspect_build
        return inspect_build(path(args.compiled))
    if args.command == "release":
        from researchflow.release import package_release
        return package_release(root, path(args.snapshot), path(args.compiled), path(args.output),
                               visual_review=path(args.visual_review))
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # Keep the callable usable by tests/embedding, including help and syntax errors.
        return int(exc.code)
    root = args.root
    try:
        root = root.resolve()
        result = _dispatch(args, root)
        print(str(result) if isinstance(result, Path) else
              json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except WorkflowError as exc:
        print(f"{args.command} (root: {root}): {exc.code}: {exc}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return {"CONTRACT": 2, "IDENTITY": 2, "EXECUTION": 3, "DEPENDENCY": 4}.get(exc.code, 1)
    except ImportError as exc:
        print(f"{args.command} (root: {root}): DEPENDENCY: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"{args.command} (root: {root}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
