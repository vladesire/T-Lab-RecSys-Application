"""Development-only metrics and prespecified user-cluster paired intervals."""
import numpy as np
from .core import tie_key


def per_query(queries, scores, seed):
    items = np.asarray([q["candidates"] for q in queries], dtype=np.int64)
    positive = np.asarray([q["positive"] for q in queries])
    if scores.shape != items.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid scores")
    if np.any(np.sort(items, axis=1)[:, 1:] == np.sort(items, axis=1)[:, :-1]):
        raise ValueError("Duplicate candidates")
    if not np.all((items == positive[:, None]).sum(axis=1) == 1):
        raise ValueError("Exactly one positive required")
    ties = np.array([[tie_key(seed, q["uid"], q["timestamp"], it) for it in row]
                     for q, row in zip(queries, items)], dtype=np.uint64)
    order = np.lexsort((items, ties, -scores), axis=1)
    rank = np.argmax(np.take_along_axis(items, order, axis=1) == positive[:, None], axis=1) + 1
    return np.column_stack([np.where(rank <= 10, 1 / np.log2(rank + 1), 0), (rank <= 10).astype(float), rank])


def aggregate(queries, values, repeat, midpoint):
    uid = np.array([q["uid"] for q in queries])
    t = np.array([q["timestamp"] for q in queries])
    anchor = np.array([q.get("anchor", False) for q in queries])
    masks = {"all": np.ones(len(queries), dtype=bool), "repeat": repeat, "novel": ~repeat,
             "V_search": t < midpoint, "V_check": t >= midpoint,
             "anchor": anchor, "new_queries": ~anchor}
    for slice_name, mask in list(masks.items()):
        if slice_name in {"V_search", "V_check"}:
            masks[slice_name + "_repeat"] = mask & repeat
            masks[slice_name + "_novel"] = mask & ~repeat
    result = {}
    for name, mask in masks.items():
        rows, users = values[mask], np.unique(uid[mask])
        result[name] = {"queries": int(mask.sum()), "users": len(users),
                       "ndcg10": float(rows[:, 0].mean()) if len(rows) else None,
                       "recall10": float(rows[:, 1].mean()) if len(rows) else None,
                       "ndcg10_user_macro": float(np.mean([values[mask & (uid == u), 0].mean() for u in users])) if len(users) else None,
                       "recall10_user_macro": float(np.mean([values[mask & (uid == u), 1].mean() for u in users])) if len(users) else None}
    return result


def paired_interval(queries, left, right, mask, seed, replicates=2000):
    uid = np.array([q["uid"] for q in queries])[mask]
    delta = (left[:, 0] - right[:, 0])[mask]
    users = np.unique(uid)
    if len(users) < 2:
        return {"status": "insufficient_users", "users": len(users), "queries": len(delta)}
    counts = np.array([sum(uid == u) for u in users])
    sums = np.array([delta[uid == u].sum() for u in users])
    rng = np.random.default_rng(seed)
    intervals = []
    for start in range(0, replicates, 250):
        draws = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=min(250, replicates - start))
        intervals.extend((draws @ sums) / (draws @ counts))
    return {"delta_ndcg10": float(delta.mean()), "percentile_95": np.quantile(intervals, [.025, .975]).tolist(),
            "users": len(users), "queries": len(delta), "replicates": replicates,
            "limitations": "conditional development evidence; user clusters, no temporal-shift or post-selection guarantee"}
