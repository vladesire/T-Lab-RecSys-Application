import copy
import gzip
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tlab_warmup.data import acquire, extract
from tlab_warmup.pipeline import prepare, run
from fixture import make_fixture


class IntegrationTests(unittest.TestCase):
    def test_failed_preflight_stops_before_acquisition(self):
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            with patch("subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="failed test", stderr="")):
                with patch("tlab_warmup.pipeline.acquire") as get:
                    record, bundle = run(c, path, Path(d) / "blocked")
                    self.assertEqual(record["status"], "failed")
                    self.assertIn("Preflight", record["error"])
                    get.assert_not_called()
                    self.assertTrue(bundle.exists())

    def test_complete_histories_including_reserved_future_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            table, info = extract(path, c, Path(d), lambda _: None)
            raw = pq.read_table(path)
            for u in info["selected_users"]:
                expected = int(np.sum(raw["uid"].to_numpy() == u))
                self.assertEqual(int(np.sum(table["uid"].to_numpy() == u)), expected)
            self.assertEqual(info["source_rows"], info["pass2_scanned_rows"])
            events, queries, audit = prepare(table, info, c)
            self.assertGreater(audit["reserved_test_rows_not_scored"], 0)
            self.assertTrue(all(q["timestamp"] < c["resolved_boundaries"]["development_end_exclusive"] for q in queries))
            table2, info2 = extract(path, c, Path(d), lambda _: None)
            self.assertEqual(table.to_pydict(), table2.to_pydict())
            self.assertEqual(info["selected_sha256"], info2["selected_sha256"])
            with self.assertRaises(ValueError):
                extract(path, {**c, "maximum_selected_rows": 1}, Path(d), lambda _: None)

    def test_unknown_development_item_and_tie_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            c["users"] = 4
            table, info = extract(path, c, Path(d), lambda _: None)
            rows = table.to_pylist()
            for row in rows:
                if 4800 <= row["timestamp"] < 5300:
                    row["item_id"] = 99999
            table = table.set_column(table.schema.get_field_index("item_id"), "item_id",
                                     pa.array([r["item_id"] for r in rows], type=pa.uint32()))
            _, qs, audit = prepare(table, info, c)
            self.assertGreater(audit["exclusions"]["development"]["target_outside_training_universe"], 0)
            self.assertGreater(audit["exclusions"]["train"]["ambiguous_user_timestamp"], 0)
            self.assertTrue(all(99999 not in q["candidates"] for q in qs))

    def test_response_values_are_not_features_or_target_filters(self):
        from tlab_warmup.core import build_features
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            table, info = extract(path, c, Path(d), lambda _: None)
            events, queries, _ = prepare(table, info, copy.deepcopy(c))
            changed = table.set_column(table.schema.get_field_index("played_ratio_pct"),
                                       "played_ratio_pct", pa.array([0] * len(table), type=pa.uint16()))
            changed = changed.set_column(changed.schema.get_field_index("track_length_seconds"),
                                         "track_length_seconds", pa.array([9999] * len(table), type=pa.uint32()))
            events2, queries2, _ = prepare(changed, info, copy.deepcopy(c))
            np.testing.assert_array_equal(events, events2)
            self.assertEqual([q["query_id"] for q in queries], [q["query_id"] for q in queries2])
            for a, b in zip(build_features(events, queries, 500), build_features(events2, queries2, 500)):
                np.testing.assert_array_equal(a, b)

    def test_hash_mismatch_rejected_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            c["source"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                acquire(c, Path(d), path, lambda _: None)

    def test_end_to_end_bundle_and_failure_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            path, c = make_fixture(d)
            c["run_no_pair_ablation"] = True
            record, bundle = run(copy.deepcopy(c), path, Path(d) / "runs", skip_checks=True)
            self.assertEqual(record["status"], "success", record.get("error"))
            self.assertEqual(record["data_kind"], "synthetic fixture")
            self.assertIn("boosting_no_pair", record["metrics"])
            with tarfile.open(bundle) as tar:
                names = tar.getnames()
                self.assertTrue(any(n.endswith("development_predictions.csv.gz") for n in names))
                self.assertFalse(any(n.endswith(".parquet") for n in names))
                config = json.load(tar.extractfile(next(n for n in names if n.endswith("resolved_config.json"))))
                predictions = tar.extractfile(next(n for n in names if n.endswith("development_predictions.csv.gz")))
                import csv, io
                rows = list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=predictions))))
                self.assertTrue(all(int(r["timestamp"]) < config["resolved_boundaries"]["development_end_exclusive"] for r in rows))
                self.assertEqual(len(rows), record["development_queries"] * 21)
            c["source"]["sha256"] = "0" * 64
            failed, bundle = run(c, path, Path(d) / "failed", skip_checks=True)
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(bundle.exists())
