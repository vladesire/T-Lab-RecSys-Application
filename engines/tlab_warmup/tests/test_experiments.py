import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tlab_warmup.core import build_features, ranked_metrics
from tlab_warmup.sampling import Fenwick, candidate_queries
from tlab_warmup.representations import build_dictionary, manifest, DAYS
from tlab_warmup.evaluation import per_query, paired_interval
from tlab_warmup.experiment_io import atomic_json, outputs_manifest, valid_output


class WeightedTests(unittest.TestCase):
    def test_exact_weighted_reference_and_strict_cutoff(self):
        events = np.array([[u % 3, item, t] for item in range(30) for u, t in enumerate([0, 5 + item, 90 + item])])
        events = events[np.argsort(events[:, 2], kind="stable")]
        q = {"uid": 1, "timestamp": 80, "positive": 2, "query_id": 1, "split": "train"}
        result = candidate_queries(events, [q], 100, 77, "P20")[0]
        ids = np.arange(30)
        weights = np.array([np.sum((events[:, 1] == it) & (events[:, 2] < 80)) + 1 for it in ids], dtype=float)
        weights[2] = 0
        rng = np.random.default_rng(np.random.SeedSequence([77, 1, 80, 0x503230]))
        expected = []
        for _ in range(20):
            index = int(np.searchsorted(np.cumsum(weights), rng.random()*weights.sum(), side="right"))
            expected.append(index)
            weights[index] = 0
        self.assertEqual(result["candidates"], sorted([2]+expected))
        modified = np.vstack([events, [1, 2, 80], [2, 28, 80], [3, 29, 1000]])
        modified = modified[np.argsort(modified[:, 2], kind="stable")]
        self.assertEqual(result, candidate_queries(modified, [q], 100, 77, "P20")[0])
        self.assertEqual(len(set(result["candidates"])), 21)

    def test_activation_catalog_and_target_restoration(self):
        events = np.array([[1, it, it] for it in range(35)])
        qs = [{"uid": 1, "timestamp": t, "positive": 2} for t in [25, 30, 50]]
        result = candidate_queries(events, qs, 28, 4, "P20")
        self.assertTrue(all(it < 25 for it in result[0]["candidates"]))
        self.assertTrue(all(it < 28 for q in result for it in q["candidates"]))
        self.assertEqual(result, candidate_queries(events, qs, 28, 4, "P20"))
        tree = Fenwick(3)
        for i, w in enumerate([1, 2, 3]): tree.add(i, w)
        before = tree.values.copy()
        tree.draw(np.random.default_rng(3), 2, 1)
        np.testing.assert_array_equal(before, tree.values)
        self.assertEqual(tree.total(), 6)


class RepresentationTests(unittest.TestCase):
    def test_interrupted_extraction_metadata_rebuilds(self):
        from fixture import make_fixture
        from tlab_warmup.data import extract
        with tempfile.TemporaryDirectory() as d:
            raw, c = make_fixture(d)
            table, info = extract(raw,c,Path(d),lambda _:None)
            meta = next(Path(d).glob("users-*.json"))
            meta.write_text('{"interrupted":')
            rebuilt, info2 = extract(raw,c,Path(d),lambda _:None)
            self.assertEqual(table.to_pydict(),rebuilt.to_pydict())
            self.assertEqual(info["selected_sha256"],info2["selected_sha256"])
            self.assertFalse(list(Path(d).glob("*.partial")))

    def test_dictionary_independent_reference_and_anchor(self):
        events = np.array([[1, 10, 0], [2, 10, 50], [1, 11, 100], [1, 10, 700], [2, 12, 700], [1, 10, 800]])
        qs = [{"uid": 1, "timestamp": t, "candidates": [10, 11, 12, 99]} for t in [700, 800, 1000]]
        spec = manifest()
        with tempfile.TemporaryDirectory() as d:
            build_dictionary(events, qs, Path(d)/"f.npy", day_seconds=100)
            f = np.load(Path(d)/"f.npy")
            original = [spec["names"].index(x) for x in spec["original"]]
            for j, (q, expected) in enumerate(zip(qs, build_features(events, qs, 700))):
                np.testing.assert_allclose(f[j*4:(j+1)*4, original], expected, rtol=1e-12)
                for k, item in enumerate(q["candidates"]):
                    past = events[events[:, 2] < q["timestamp"]]
                    for relation in ["user", "item", "user_item"]:
                        selected = past[(past[:, 0] == 1) if relation=="user" else (past[:, 1] == item) if relation=="item" else ((past[:, 0]==1)&(past[:, 1]==item))]
                        for day in DAYS:
                            ages = q["timestamp"]-selected[:, 2]
                            expected_hard = np.sum(ages <= day*100)
                            expected_decay = np.exp(-np.log(2)*ages/(day*100)).sum()
                            self.assertAlmostEqual(f[j*4+k, spec["names"].index(f"{relation}_hard_{day}d")], expected_hard)
                            self.assertAlmostEqual(f[j*4+k, spec["names"].index(f"{relation}_decay_{day}d")], expected_decay, places=11)
            modified = np.vstack([events, [1, 99, 1000], [1, 10, 2000]])
            build_dictionary(modified, qs, Path(d)/"later.npy", day_seconds=100)
            np.testing.assert_allclose(f, np.load(Path(d)/"later.npy"), rtol=1e-12, atol=1e-12)
        self.assertEqual(len(spec["manual"]), len(spec["decay"]))
        self.assertLessEqual(len(spec["names"]), 64)

    def test_metrics_reference_and_cluster_uncertainty(self):
        qs = [{"uid": j%5, "timestamp": j, "positive": 7, "candidates": list(range(21))} for j in range(30)]
        scores = np.random.default_rng(4).random((30, 21))
        actual = per_query(qs, scores, 3)
        for j, q in enumerate(qs):
            ref = ranked_metrics(q["candidates"], scores[j], 7, 3, q["uid"], q["timestamp"])
            np.testing.assert_allclose(actual[j], [ref["ndcg10"], ref["recall10"], ref["rank"]])
        ci = paired_interval(qs, actual, actual, np.ones(30,dtype=bool), 4, 100)
        self.assertEqual(ci["percentile_95"], [0, 0])

    def test_atomic_completion_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            atomic_json(p/"result.json", {"x":1})
            self.assertFalse(valid_output(p))
            atomic_json(p/"COMPLETE.json", {"files": outputs_manifest(p)})
            self.assertTrue(valid_output(p))
            atomic_json(p/"result.json", {"x":2})
            self.assertFalse(valid_output(p))
