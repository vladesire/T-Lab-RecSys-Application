#!/usr/bin/env python3
"""Independent runner; no researchflow import or infrastructure installation."""
import argparse
import json
from pathlib import Path
import sys

from bridge import capture_source, execute, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-source", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    try:
        if args.capture_source:
            if not args.project or not args.python or args.request:
                parser.error("--capture-source requires --project and --python, no --request")
            if args.output.exists():
                parser.error("Manifest output already exists")
            write(args.output, capture_source(args.project, args.python))
        else:
            if not args.request or args.project or args.python:
                parser.error("Execution requires --request and --output")
            execute(json.loads(args.request.read_text()), args.output)
    except Exception as exc:
        print(f"warmup bridge: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
