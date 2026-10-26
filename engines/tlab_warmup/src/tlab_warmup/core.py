"""Temporal features and ranking protocol; no model or I/O dependencies."""
from collections import defaultdict
import hashlib
import math

import numpy as np

FEATURE_NAMES = [
    "user_count", "user_window_count", "item_count", "item_window_count",
    "item_window_share_smoothed", "user_item_count", "user_recency_seconds",
    "user_item_recency_seconds", "user_item_missing",
]
PAIR_COLUMNS = [5, 7, 8]


def tie_key(seed, uid, timestamp, item):
    text = f"{seed}:{uid}:{timestamp}:{item}".encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "big")


def ranked_metrics(items, scores, positive, seed, uid, timestamp, k=10):
    """One relevant item; ties use a label-independent seeded item hash."""
    if len(set(map(int, items))) != len(items) or positive not in items:
        raise ValueError("Candidates must be unique and include the positive")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ranking score")
    order = sorted(range(len(items)), key=lambda j: (
        -float(scores[j]), tie_key(seed, uid, timestamp, int(items[j])), int(items[j])))
    rank = next(r + 1 for r, j in enumerate(order) if int(items[j]) == positive)
    return {"rank": rank, "ndcg10": 1 / math.log2(rank + 1) if rank <= k else 0.0,
            "recall10": float(rank <= k)}


def sample_candidates(universe, first_seen, timestamp, positive, count, seed, uid):
    """Uniform without replacement from items observed strictly before query time.

    universe and first_seen are aligned and sorted by (first_seen, item_id).
    Target exclusion does not exclude repeat-consumption items.
    """
    n = int(np.searchsorted(first_seen, timestamp, side="left"))
    available = universe[:n]
    if positive not in available:
        raise ValueError("Target unavailable in historical item universe")
    if n < count + 1:
        raise ValueError("Insufficient previously observed candidate items")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid), int(timestamp)]))
    drawn = available[rng.choice(n, count + 1, replace=False)]
    alternatives = [int(x) for x in drawn if int(x) != positive][:count]
    return np.array(sorted(alternatives + [int(positive)]), dtype=np.int64)


def build_features(events, queries, window, smoothing=1.0):
    """Single sweep over events, with strict t_event < t_query.

    Events: time-sorted integer array [uid, item_id, timestamp]. Queries contain
    uid, timestamp, candidates. Window is [t-window, t). All event types count.
    No labels or response metadata enter this function.
    """
    if window <= 0 or smoothing <= 0:
        raise ValueError("Window and smoothing must be positive")
    if len(events) and np.any(np.diff(events[:, 2]) < 0):
        raise ValueError("Events must be time sorted")
    uc, ic, pc, uw, iw = (defaultdict(int) for _ in range(5))
    ul, pl = {}, {}
    cursor = left = 0
    outputs = [None] * len(queries)
    for qi in sorted(range(len(queries)), key=lambda i: queries[i]["timestamp"]):
        q = queries[qi]
        t, u = int(q["timestamp"]), int(q["uid"])
        while cursor < len(events) and int(events[cursor, 2]) < t:
            eu, it, et = map(int, events[cursor])
            uc[eu] += 1
            ic[it] += 1
            pc[eu, it] += 1
            uw[eu] += 1
            iw[it] += 1
            ul[eu], pl[eu, it] = et, et
            cursor += 1
        while left < cursor and int(events[left, 2]) < t - window:
            eu, it = map(int, events[left, :2])
            uw[eu] -= 1
            iw[it] -= 1
            left += 1
        rows = []
        for it in q["candidates"]:
            it = int(it)
            prev = pl.get((u, it))
            rows.append([uc[u], uw[u], ic[it], iw[it],
                         (iw[it] + smoothing) / (ic[it] + 2 * smoothing),
                         pc[u, it], t - ul[u] if u in ul else -1,
                         t - prev if prev is not None else -1, int(prev is None)])
        outputs[qi] = np.asarray(rows, dtype=np.float64)
    return outputs


def brute_features(events, query, window, smoothing=1.0):
    """Independent slow reference: direct masks, no shared sweep state."""
    u, t = query["uid"], query["timestamp"]
    past = events[events[:, 2] < t]
    recent = past[past[:, 2] >= t - window]
    user = past[past[:, 0] == u]
    result = []
    for it in query["candidates"]:
        item = past[past[:, 1] == it]
        pair = user[user[:, 1] == it]
        result.append([len(user), int(np.sum(recent[:, 0] == u)), len(item),
                       int(np.sum(recent[:, 1] == it)),
                       (np.sum(recent[:, 1] == it) + smoothing) / (len(item) + 2 * smoothing),
                       len(pair), t - int(user[:, 2].max()) if len(user) else -1,
                       t - int(pair[:, 2].max()) if len(pair) else -1, int(not len(pair))])
    return np.array(result, dtype=float)
