"""Small atomic records and content identities for the sequential batch."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from .data import sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def save_queries(path, queries):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    with gzip.open(temp, "wt") as f:
        json.dump(queries, f, separators=(",", ":"))
    temp.replace(path)


def load_queries(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def code_state(root):
    root = Path(root)
    files = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    hashes = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    state = {"tree_sha256": digest(hashes), "files": hashes}
    try:
        state["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        state["git_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError):
        state.update(git_head=None, git_status="not a git checkout; content hash is authoritative")
    return state


def outputs_manifest(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "COMPLETE.json" and not p.name.endswith(".partial")}


def valid_output(directory):
    directory = Path(directory)
    marker = directory / "COMPLETE.json"
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text())
        return saved["files"] == outputs_manifest(directory)
    except (OSError, ValueError, KeyError):
        return False
