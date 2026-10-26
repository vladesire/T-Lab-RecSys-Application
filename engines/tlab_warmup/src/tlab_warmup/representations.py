"""Compact indexed histories for a frozen 47-column temporal dictionary."""
import numpy as np
from .core import FEATURE_NAMES

DAYS = [1, 7, 30, 90]
BASE = ["user_count", "item_count", "user_item_count", "user_recency_seconds",
        "item_recency_seconds", "user_item_recency_seconds", "user_item_missing"]


def manifest():
    names = list(BASE)
    for relation in ["user", "item", "user_item"]:
        for kind in ["hard", "decay"]:
            for day in DAYS:
                names.append(f"{relation}_{kind}_{day}d")
    for kind in ["hard", "decay"]:
        for day in DAYS:
            names += [f"item_{kind}_share_{day}d", f"pair_{kind}_affinity_{day}d"]
    original = ["user_count", "user_hard_7d", "item_count", "item_hard_7d",
                "item_hard_share_7d", "user_item_count", "user_recency_seconds",
                "user_item_recency_seconds", "user_item_missing"]
    hard = list(BASE) + [f"{r}_hard_{d}d" for r in ["user", "item", "user_item"] for d in [7, 30]]
    decay = list(BASE) + [f"{r}_decay_{d}d" for r in ["user", "item", "user_item"] for d in [7, 30]]
    for d in [7, 30]:
        hard += [f"item_hard_share_{d}d", f"pair_hard_affinity_{d}d"]
        decay += [f"item_decay_share_{d}d", f"pair_decay_affinity_{d}d"]
    groups = {}
    for relation in ["user", "item", "user_item"]:
        for kind in ["hard_extra", "decay_all"]:
            mode = "hard" if kind == "hard_extra" else "decay"
            ds = [1, 90] if mode == "hard" else DAYS
            group = [f"{relation}_{mode}_{d}d" for d in ds]
            if relation == "item":
                group += [f"item_{mode}_share_{d}d" for d in ds]
            if relation == "user_item":
                group += [f"pair_{mode}_affinity_{d}d" for d in ds]
            groups[f"{relation}_{kind}"] = group
    assert len(names) == len(set(names)) <= 64
    assert set(hard).union(*(set(v) for v in groups.values())) == set(names)
    return {"names": names, "base": BASE, "original": original, "original_wire_names": FEATURE_NAMES,
            "no_pair": [n for n, wire in zip(original, FEATURE_NAMES) if wire not in {
                "user_item_count", "user_item_recency_seconds", "user_item_missing"}],
            "manual": hard, "hard": hard, "decay": decay, "all": names, "groups": groups,
            "aliases": {"F_hard": "manual", "manual_without_artist": "manual"},
            "semantics": {"counts": "all prior logged event identities", "cutoff": "strictly less than query timestamp",
                "hard_window": "[t-horizon,t)", "decay": "sum(exp(-ln(2)*(t-s)/half_life)) for s<t",
                "item_share": "(weighted_item_count+1)/(cumulative_item_count+2)",
                "pair_affinity": "(weighted_pair_count+1)/(weighted_user_count+2)",
                "missing_recency": -1, "artist": "blocked: mapping semantics/access not validated; absent from all methods"}}


def relation_keys(events, relation):
    if relation == "user":
        return events[:, 0].astype(np.uint64)
    if relation == "item":
        return events[:, 1].astype(np.uint64)
    if np.any(events[:, :2] > np.iinfo(np.uint32).max):
        raise ValueError("Pair encoding requires uint32 identifiers")
    return (events[:, 0].astype(np.uint64) << np.uint64(32)) | events[:, 1].astype(np.uint64)


class HistoryIndex:
    """Entity/time sorted records, segmented decay prefix sums; no cross-joins."""
    def __init__(self, keys, timestamps):
        order = np.lexsort((timestamps, keys))
        self.records = np.empty(len(keys), dtype=[("key", "<u8"), ("time", "<i8")])
        self.records["key"], self.records["time"] = keys[order], timestamps[order]
        self.ids, self.starts, self.sizes = np.unique(self.records["key"], return_index=True, return_counts=True)

    def positions(self, keys, times):
        query = np.empty(len(keys), dtype=self.records.dtype)
        query["key"], query["time"] = keys, times
        stop = np.searchsorted(self.records, query, side="left")
        idpos = np.searchsorted(self.ids, keys)
        valid = idpos < len(self.ids)
        valid[valid] &= self.ids[idpos[valid]] == keys[valid]
        starts = np.zeros(len(keys), dtype=np.int64)
        starts[valid] = self.starts[idpos[valid]]
        counts = np.where(valid, stop - starts, 0)
        return stop, starts, counts

    def values(self, keys, times, day_seconds):
        stop, starts, counts = self.positions(keys, times)
        recency = np.where(counts > 0, times - self.records["time"][np.maximum(stop - 1, 0)], -1)
        out = {"count": counts.astype(float), "recency": recency.astype(float)}
        for day in DAYS:
            low, _, _ = self.positions(keys, times - int(day * day_seconds))
            out[f"hard_{day}d"] = np.where(counts > 0, stop - low, 0).astype(float)
            rate = np.log(2) / (day * day_seconds)
            prefix = np.empty(len(self.records), dtype=np.float64)
            singles = self.starts[self.sizes == 1]
            prefix[singles] = 1.0
            for start, size in zip(self.starts[self.sizes > 1], self.sizes[self.sizes > 1]):
                dt = self.records["time"][start:start + size] - self.records["time"][start]
                if float(dt.max()) * rate > 650:
                    raise ValueError("Decay prefix exponent range exceeded; no unstable fallback")
                np.cumsum(np.exp(rate * dt), out=prefix[start:start + size])
            origin = self.records["time"][np.minimum(starts, len(self.records) - 1)]
            out[f"decay_{day}d"] = np.where(counts > 0,
                prefix[np.maximum(stop - 1, 0)] * np.exp(-rate * (times - origin)), 0.0)
        return out


def build_dictionary(events, queries, path, day_seconds=86400):
    spec = manifest()
    n = len(queries)
    k = len(queries[0]["candidates"])
    if any(len(q["candidates"]) != k for q in queries):
        raise ValueError("Mixed candidate cardinalities")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n * k, len(spec["names"])))
    lookup = {name: j for j, name in enumerate(spec["names"])}
    times = np.repeat([q["timestamp"] for q in queries], k).astype(np.int64)
    uid = np.repeat([q["uid"] for q in queries], k).astype(np.uint64)
    items = np.asarray([q["candidates"] for q in queries], dtype=np.uint64).reshape(-1)
    for relation in ["user", "item", "user_item"]:
        keys = uid if relation == "user" else items if relation == "item" else (uid << np.uint64(32)) | items
        index = HistoryIndex(relation_keys(events, relation), events[:, 2])
        values = index.values(keys, times, day_seconds)
        matrix[:, lookup[f"{relation}_count"]] = values["count"]
        recency_name = f"{relation}_recency_seconds"
        matrix[:, lookup[recency_name]] = values["recency"]
        if relation == "user_item":
            matrix[:, lookup["user_item_missing"]] = (values["count"] == 0).astype(float)
        for mode in ["hard", "decay"]:
            for day in DAYS:
                matrix[:, lookup[f"{relation}_{mode}_{day}d"]] = values[f"{mode}_{day}d"]
        del index, values
    for mode in ["hard", "decay"]:
        for day in DAYS:
            matrix[:, lookup[f"item_{mode}_share_{day}d"]] = (matrix[:, lookup[f"item_{mode}_{day}d"]] + 1) / (matrix[:, lookup["item_count"]] + 2)
            matrix[:, lookup[f"pair_{mode}_affinity_{day}d"]] = (matrix[:, lookup[f"user_item_{mode}_{day}d"]] + 1) / (matrix[:, lookup[f"user_{mode}_{day}d"]] + 2)
    matrix.flush()
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite feature dictionary")
    return spec
