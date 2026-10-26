"""Pinned acquisition and two-pass, full-release user extraction."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ["uid", "item_id", "timestamp", "event_type", "is_organic",
           "played_ratio_pct", "track_length_seconds"]
EVENT_TYPES = {"listen", "like", "dislike", "unlike", "undislike"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(config, cache, existing, log):
    source = config["source"]
    path = Path(existing).expanduser().resolve() if existing else cache / "multi_event.parquet"
    if path.exists():
        log("Checking existing full-release file fingerprint")
    elif existing:
        raise FileNotFoundError(f"Configured data path does not exist: {path}")
    else:
        if source["bytes"] > config["maximum_download_bytes"]:
            raise ValueError("Pinned file exceeds configured download byte limit")
        partial = path.with_suffix(".parquet.part")
        size = partial.stat().st_size if partial.exists() else 0
        if size > source["bytes"]:
            raise ValueError(f"Oversized partial file; remove {partial}")
        start, last_log = time.monotonic(), 0.0
        if size < source["bytes"]:
            url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
                   f"{source['revision']}/{source['file']}")
            headers = {"Range": f"bytes={size}-", "Accept-Encoding": "identity"} if size else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                if size and (response.status != 206 or not response.headers.get(
                        "Content-Range", "").startswith(f"bytes {size}-")):
                    raise RuntimeError(f"Server refused resume. Remove {partial} and retry")
                with open(partial, "ab" if size else "wb") as f:
                    while True:
                        if time.monotonic() - start > config["acquisition_seconds"]:
                            raise TimeoutError("Acquisition budget reached; partial retained. Rerun same command")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        size += len(chunk)
                        if size > source["bytes"]:
                            raise ValueError("Download exceeds pinned size")
                        if time.monotonic() - last_log >= 5:
                            log(f"Downloaded {size / 1e6:.1f}/{source['bytes'] / 1e6:.1f} MB")
                            last_log = time.monotonic()
        if size != source["bytes"] or sha256(partial) != source["sha256"]:
            raise ValueError(f"Partial download fingerprint mismatch: {partial}; remove and retry")
        partial.replace(path)
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError("Source bytes/SHA256 do not match pinned release; refusing unverified data")
    return path


def extract(path, config, cache, log):
    """Pass one enumerates every user; pass two scans every row for selected IDs.

    Cache reuse requires raw-source hash (verified by acquire), protocol config,
    code fingerprint, selected-file hash, and prior full-scan reconciliation.
    """
    source = pq.ParquetFile(path)
    if not set(COLUMNS).issubset(source.schema_arrow.names):
        raise ValueError(f"Missing required columns. Actual schema: {source.schema_arrow}")
    for col in ["uid", "item_id", "timestamp"]:
        if not pa.types.is_integer(source.schema_arrow.field(col).type):
            raise ValueError(f"Expected integer {col}; actual {source.schema_arrow}")
    identity = {"source": config["source"], "seed": config["seed"], "users": config["users"],
                "code_sha256": sha256(__file__), "maximum_selected_rows": config["maximum_selected_rows"]}
    if "retain_user_ids" in config:
        identity["retain_user_ids"] = sorted(config["retain_user_ids"])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cached, meta = cache / f"users-{key}.parquet", cache / f"users-{key}.json"
    if cached.exists() and meta.exists():
        try:
            info = json.loads(meta.read_text())
        except (OSError, ValueError):
            info = {}
        if info.get("identity") == identity and sha256(cached) == info.get("selected_sha256"):
            log("Reusing verified complete-history extraction cache")
            return pq.read_table(cached), info
        log("Invalid extraction cache; rebuilding")
    counts = {}
    lo, hi, scanned = None, None, 0
    for batch in source.iter_batches(batch_size=262144, columns=["uid", "timestamp"]):
        if any(c.null_count for c in batch.columns):
            raise ValueError("Null user or timestamp in source")
        uid, ts = (c.to_numpy(zero_copy_only=False).astype(np.int64) for c in batch.columns)
        if np.any(ts < 0):
            raise ValueError("Negative timestamp")
        ids, ns = np.unique(uid, return_counts=True)
        for u, n in zip(ids, ns):
            counts[int(u)] = counts.get(int(u), 0) + int(n)
        lo = min(lo, int(ts.min())) if lo is not None else int(ts.min())
        hi = max(hi, int(ts.max())) if hi is not None else int(ts.max())
        scanned += len(batch)
        if scanned % (262144 * 20) == 0:
            log(f"User enumeration: {scanned:,}/{source.metadata.num_rows:,} rows")
    if scanned != source.metadata.num_rows or not counts:
        raise ValueError("Incomplete/empty first scan")
    population = np.array(sorted(counts))
    retained = np.array(sorted(set(config.get("retain_user_ids", []))), dtype=np.int64)
    if len(retained) > config["users"] or not set(retained).issubset(counts):
        raise ValueError("Retained cohort is unavailable or larger than the requested panel")
    if len(retained):
        remaining = population[~np.isin(population, retained)]
        extra = np.random.default_rng(config["seed"]).choice(remaining,
            min(config["users"] - len(retained), len(remaining)), replace=False)
        users = np.sort(np.concatenate([retained, extra]))
    else:
        users = np.sort(np.random.default_rng(config["seed"]).choice(
            population, min(config["users"], len(population)), replace=False))
    expected = sum(counts[int(u)] for u in users)
    if expected > config["maximum_selected_rows"]:
        raise ValueError(f"Selected {expected:,} rows exceed resource cap; lower users explicitly; no truncation performed")
    chunks, scanned2 = [], 0
    user_set = pa.array(users, type=source.schema_arrow.field("uid").type)
    for batch in source.iter_batches(batch_size=262144, columns=COLUMNS):
        mask = pc.is_in(batch.column("uid"), value_set=user_set)
        selected = pa.Table.from_batches([batch]).filter(mask)
        if len(selected):
            chunks.append(selected)
        scanned2 += len(batch)
        if scanned2 % (262144 * 20) == 0:
            log(f"History extraction: {scanned2:,}/{source.metadata.num_rows:,} rows")
    if not chunks:
        raise ValueError("No selected rows")
    table = pa.concat_tables(chunks).combine_chunks()
    actual_ids, ns = np.unique(table["uid"].to_numpy(), return_counts=True)
    actual = dict(zip(map(int, actual_ids), map(int, ns)))
    expected_counts = {int(u): counts[int(u)] for u in users}
    if actual != expected_counts or len(table) != expected or scanned2 != scanned:
        raise ValueError("Full-history extraction reconciliation failed")
    partial_cache = cached.with_name(cached.name + ".partial")
    pq.write_table(table, partial_cache, compression="zstd")
    partial_cache.replace(cached)
    info = {"identity": identity, "selected_users": users.tolist(),
            "selected_counts": expected_counts, "source_schema": str(source.schema_arrow),
            "source_rows": scanned, "source_users": len(counts), "source_min_timestamp": lo,
            "source_max_timestamp": hi, "pass2_scanned_rows": scanned2,
            "complete_histories_reconciled": True, "selected_rows": len(table),
            "selected_sha256": sha256(cached), "selected_cache_bytes": cached.stat().st_size}
    partial_meta = meta.with_name(meta.name + ".partial")
    partial_meta.write_text(json.dumps(info, indent=2) + "\n")
    partial_meta.replace(meta)
    return table, info
