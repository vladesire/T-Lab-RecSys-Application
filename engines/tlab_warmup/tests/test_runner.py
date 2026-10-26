import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from fixture import make_fixture
from tlab_warmup.batch import execute_job, run_batch
from tlab_warmup.experiment_io import atomic_json
from tlab_warmup.experiment_jobs import search_job
from tlab_warmup.pipeline import run, ROOT


class RunnerTests(unittest.TestCase):
    def fixture_config(self, directory):
        raw, c = make_fixture(directory)
        c["window_seconds"] = 700
        _, anchor = run(c, raw, Path(directory)/"anchor", skip_checks=True)
        cfg = json.loads((ROOT/"configs/experiments.json").read_text())
        cfg.update(data=str(raw), bootstrap_replicates=50, threads=1,
                   minimum_available_gib=.01, maximum_selected_rows=10000)
        cfg["panels"]["iteration1"] = {"users":3,"train":30,"development":24}
        return cfg, anchor

    def test_stop_resume_corruption_invalidation_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, anchor = self.fixture_config(d)
            out = Path(d)/"batch"
            out.mkdir()
            (out/"STOP").touch()
            self.assertFalse(run_batch("iteration1", cfg, anchor, out, check=False))
            self.assertTrue(all(not j["attempts"] for j in json.loads((out/"ledger.json").read_text()).values()))
            (out/"STOP").unlink()
            self.assertTrue(run_batch("iteration1", cfg, anchor, out, check=False))
            first = json.loads((out/"ledger.json").read_text())
            self.assertTrue(all(j["status"] not in {"running", "pending", "interrupted"} for j in first.values()))
            self.assertTrue(run_batch("iteration1", cfg, anchor, out, check=False))
            same = json.loads((out/"ledger.json").read_text())
            self.assertEqual({k:len(v["attempts"]) for k,v in first.items()}, {k:len(v["attempts"]) for k,v in same.items()})
            # Corrupt just U20 cache: its descendants rerun, P20 remains committed.
            p = Path(first["prepare_U20"]["output"])/"features.npy"
            with p.open("ab") as f: f.write(b"bad")
            self.assertTrue(run_batch("iteration1", cfg, anchor, out, check=False))
            repaired = json.loads((out/"ledger.json").read_text())
            self.assertEqual(len(repaired["U20_original"]["attempts"]), 2)
            self.assertEqual(len(repaired["P20_original"]["attempts"]), 1)
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                run_batch("iteration1", {**cfg,"threads":2}, anchor, out, check=False)

    def test_memory_guard_stops_and_records_interrupted_job(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            cfg = json.loads((ROOT/"configs/experiments.json").read_text())
            cfg["stop_rss_gib"] = .001
            job = {"id":"guarded", "output":str(p/"output"), "attempts":[],
                   "request":{"kind":"source","config":cfg,"anchor":"missing"}}
            atomic_json(p/"ledger.json", {"guarded":job})
            with patch("tlab_warmup.batch.observe_tree", return_value=(1024**3,10*1024**3)):
                execute_job(job, p, cfg, time.monotonic()+60, {"reason":None})
            self.assertEqual(job["status"], "interrupted")
            self.assertIn("RSS guard", job["reason"])
            self.assertEqual(len(job["attempts"]), 1)
            self.assertFalse((p/"output/COMPLETE.json").exists())

    def test_search_three_round_budget_and_matched_random_without_check_peeking(self):
        # Artificial scoring checks search bookkeeping, not recommendation quality.
        from tlab_warmup.representations import manifest
        dev = [{"uid":j%2,"timestamp":t,"query_id":j,"positive":0,"candidates":list(range(21)),"split":"development"}
               for j,t in enumerate([10,11,20,21])]
        meta = {"config":{"seed":7,"threads":1},"midpoint":15,"protocol":"P20","query_candidate_hash":"fixture"}
        cfg = {"threads":1,"search_min_queries":1,"search_min_users":1,"search_min_novel":0,
               "search_rounds":3,"search_fit_cap":24,"search_min_gain":.0001}
        calls = []
        def predict(model, f, ntrain, names, ndev, threads, indices=None):
            calls.append(None if indices is None else list(indices))
            return np.full((ndev if indices is None else len(indices),21),len(names)/100)
        def fake_per_query(q,s,seed):
            return np.column_stack([s[:,0],np.ones(len(q)),np.ones(len(q))])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d);manual=p/"manual";manual.mkdir()
            atomic_json(manual/"result.json", {"metrics":{"V_search":{"ndcg10":0}}})
            forward=p/"forward";forward.mkdir();random=p/"random";random.mkdir()
            request={"config":cfg,"seed":7,"method":"forward","manual_result":str(manual)}
            data=(meta,dev,np.zeros((84,47)),0,dev,np.ones(4,dtype=bool))
            with patch("tlab_warmup.experiment_jobs.data_for_model", return_value=data), \
                 patch("tlab_warmup.experiment_jobs.train_model", side_effect=lambda *a: ({"n":len(a[3])},.001)), \
                 patch("tlab_warmup.experiment_jobs.model_scores", side_effect=predict), \
                 patch("tlab_warmup.experiment_jobs.per_query", side_effect=fake_per_query):
                search_job(request,forward)
                result=json.loads((forward/"result.json").read_text())
                self.assertEqual(result["candidate_fits"],15)
                self.assertEqual(len(result["selected_groups"]),3)
                self.assertTrue(all(c==[0,1] for c in calls[:-1]));self.assertIsNone(calls[-1])
                request.update(method="random_search",forward_result=str(forward))
                search_job(request,random)
                other=json.loads((random/"result.json").read_text())
                self.assertEqual(result["trial_addition_counts"],other["trial_addition_counts"])
                self.assertEqual(other["candidate_fits"],15)
