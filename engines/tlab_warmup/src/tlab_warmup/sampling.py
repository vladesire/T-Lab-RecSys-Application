"""Exact chronological popularity-weighted sampling, without replacement."""
import numpy as np
from .core import sample_candidates


class Fenwick:
    def __init__(self, n):
        self.tree = np.zeros(n + 1, dtype=np.float64)
        self.values = np.zeros(n, dtype=np.float64)

    def add(self, index, delta):
        self.values[index] += delta
        j = index + 1
        while j < len(self.tree):
            self.tree[j] += delta
            j += j & -j

    def total(self):
        j, result = len(self.values), 0.0
        while j:
            result += self.tree[j]
            j -= j & -j
        return result

    def pick(self, value):
        if not 0 <= value < self.total():
            raise ValueError("Draw outside positive cumulative mass")
        index = 0
        bit = 1 << (len(self.values).bit_length() - 1)
        while bit:
            trial = index + bit
            if trial < len(self.tree) and self.tree[trial] <= value:
                value -= self.tree[trial]
                index = trial
            bit >>= 1
        return index

    def draw(self, rng, k, exclude):
        if self.values[exclude] <= 0:
            raise ValueError("Target has no strictly-past availability")
        removed = [(exclude, self.values[exclude])]
        self.add(exclude, -removed[0][1])
        chosen = []
        try:
            for _ in range(k):
                total = self.total()
                if total <= 0:
                    raise ValueError("Insufficient weighted candidate support")
                j = self.pick(min(rng.random() * total, np.nextafter(total, 0)))
                weight = self.values[j]
                if weight <= 0:
                    raise AssertionError("Nonpositive selected mass")
                chosen.append(j)
                removed.append((j, weight))
                self.add(j, -weight)
        finally:
            for j, weight in removed:
                self.add(j, weight)
        return chosen


def catalog(events, train_end):
    past = events[events[:, 2] < train_end]
    ids, start = np.unique(past[:, 1], return_index=True)
    first = past[start, 2]
    order = np.lexsort((ids, first))
    return ids[order], first[order]


def candidate_queries(events, targets, train_end, seed, protocol, alternatives=20):
    """P20 weights c_i(t)+1, exponent exactly 1; no lagged snapshots.

    The training-defined catalog is further restricted to first_seen < t.
    Inactive items have zero mass; smoothing is added only on activation.
    U20 delegates to the original sampler to preserve its exact RNG behavior.
    """
    universe, first = catalog(events, train_end)
    if protocol == "U20":
        return [{**q, "candidates": sample_candidates(universe, first, q["timestamp"],
                 q["positive"], alternatives, seed, q["uid"]).tolist()} for q in targets]
    if protocol != "P20":
        raise ValueError(f"Unknown candidate protocol {protocol}")
    ids = np.sort(universe)
    index = {int(it): j for j, it in enumerate(ids)}
    tree = Fenwick(len(ids))
    cursor = 0
    result = [None] * len(targets)
    for qi in sorted(range(len(targets)), key=lambda i: targets[i]["timestamp"]):
        q = targets[qi]
        stop = int(np.searchsorted(events[:, 2], q["timestamp"], side="left"))
        changed, counts = np.unique(events[cursor:stop, 1], return_counts=True)
        for it, count in zip(changed, counts):
            j = index.get(int(it))
            if j is not None:
                tree.add(j, int(count) + int(tree.values[j] == 0))
        cursor = stop
        rng = np.random.default_rng(np.random.SeedSequence([seed, q["uid"], q["timestamp"], 0x503230]))
        selected = tree.draw(rng, alternatives, index[q["positive"]])
        result[qi] = {**q, "candidates": sorted([q["positive"], *map(int, ids[selected])])}
    return result
