import math
import unittest
import numpy as np
from tlab_warmup.core import build_features, brute_features, sample_candidates, ranked_metrics


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.events = np.array([[1, 10, 0], [2, 10, 5], [1, 11, 10],
                                [1, 10, 15], [2, 12, 15], [1, 12, 20], [1, 10, 25]])
        self.q = {"uid": 1, "timestamp": 20, "candidates": np.array([10, 11, 12, 99])}

    def test_manual_counts_and_missing(self):
        actual = build_features(self.events, [self.q], 10)[0]
        np.testing.assert_allclose(actual[0], [3, 2, 3, 1, 2/5, 2, 5, 5, 0])
        np.testing.assert_allclose(actual[2], [3, 2, 1, 1, 2/3, 0, 5, -1, 1])
        np.testing.assert_allclose(actual[3], [3, 2, 0, 0, 1/2, 0, 5, -1, 1])

    def test_independent_reference_and_unsorted_queries(self):
        queries = [self.q, {**self.q, "timestamp": 10}, {**self.q, "timestamp": 30}]
        for actual, q in zip(build_features(self.events, queries, 10), queries):
            np.testing.assert_allclose(actual, brute_features(self.events, q, 10))

    def test_future_invariance(self):
        modified = np.vstack([self.events, [1, 99, 100], [2, 10, 101]])
        np.testing.assert_array_equal(build_features(self.events, [self.q], 10)[0],
                                      build_features(modified, [self.q], 10)[0])

    def test_target_invariance_and_same_timestamp_isolation(self):
        past = self.events[self.events[:, 2] != 20]
        changed = np.vstack([past, [1, 10, 20], [2, 11, 20], [1, 99, 20]])
        changed = changed[np.argsort(changed[:, 2], kind="stable")]
        np.testing.assert_array_equal(build_features(past, [self.q], 10)[0],
                                      build_features(changed, [self.q], 10)[0])

    def test_randomized_reference(self):
        rng = np.random.default_rng(78)
        events = np.column_stack([rng.integers(0, 4, 200), rng.integers(0, 15, 200),
                                  np.sort(rng.integers(0, 60, 200))])
        queries = [{"uid": int(rng.integers(4)), "timestamp": int(t), "candidates": np.arange(17)}
                   for t in rng.integers(0, 90, 40)]
        for x, q in zip(build_features(events, queries, 12, 2), queries):
            np.testing.assert_allclose(x, brute_features(events, q, 12, 2), atol=1e-12)

    def test_unsorted_events_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.events[::-1], [self.q], 10)


class RankingTests(unittest.TestCase):
    def test_candidates_historical_unique_reproducible_and_keep_repeats(self):
        items, seen = np.arange(40), np.arange(40) * 5
        a = sample_candidates(items, seen, 150, 2, 20, 7, 9)
        b = sample_candidates(items, seen, 150, 2, 20, 7, 9)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 21)
        self.assertEqual(len(set(a)), 21)
        self.assertIn(2, a)
        self.assertTrue(np.all(a < 30))
        with self.assertRaises(ValueError):
            sample_candidates(items, seen, 150, 30, 20, 7, 9)
        with self.assertRaises(ValueError):
            sample_candidates(items, seen, 50, 2, 20, 7, 9)

    def test_future_universe_items_do_not_change_candidates(self):
        a = sample_candidates(np.arange(30), np.arange(30), 30, 2, 20, 7, 9)
        b = sample_candidates(np.arange(50), np.arange(50), 30, 2, 20, 7, 9)
        np.testing.assert_array_equal(a, b)

    def test_metrics_reference(self):
        items = np.arange(21)
        scores = -np.arange(21, dtype=float)
        for positive in [0, 1, 9, 10, 20]:
            m = ranked_metrics(items, scores, positive, 7, 1, 50)
            rank = positive + 1
            self.assertEqual(m["rank"], rank)
            self.assertEqual(m["recall10"], int(rank <= 10))
            self.assertAlmostEqual(m["ndcg10"], 1/math.log2(rank+1) if rank <= 10 else 0)

    def test_ties_ignore_candidate_position_and_label(self):
        items = np.arange(21)
        for positive in [0, 8, 20]:
            a = ranked_metrics(items, np.zeros(21), positive, 7, 1, 50)
            b = ranked_metrics(items[::-1], np.zeros(21), positive, 7, 1, 50)
            self.assertEqual(a, b)
        ranks = [ranked_metrics(items, np.zeros(21), int(p), 7, 1, 50)["rank"] for p in items]
        self.assertEqual(sorted(ranks), list(range(1, 22)))
