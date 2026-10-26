"""Scientific boundary checks for the separate frozen holdout evaluator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'engines/tlab_warmup/src'))
spec=importlib.util.spec_from_file_location('holdout',ROOT/'tools/evaluate_frozen_final.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class HoldoutBoundaryTests(unittest.TestCase):
    def population(self):
        rows=[(1,10,1),(1,11,2),(1,12,3),(1,10,20),(1,13,21),
              (2,10,22),(1,10,25),(1,11,25),(1,10,31)]
        return np.array(rows,dtype=np.int64),np.array(['listen']*len(rows))

    def test_eligibility_preserves_training_catalog_and_user_and_tie_boundaries(self):
        events,types=self.population()
        pool,excluded=h.eligible_holdout(events,types,{'train_end_timestamp':10,
            'minimum_prior_user_events':1,'alternatives':2,'target_event_type':'listen'},20,30)
        self.assertEqual([(q['uid'],q['timestamp'],q['positive']) for q in pool],[(1,20,10)])
        self.assertEqual(excluded['outside_training_catalog'],1)
        self.assertEqual(excluded['insufficient_prior_history'],1)
        self.assertEqual(excluded['ambiguous_user_timestamp'],2)

    def test_interval_and_target_type_are_not_relabeled_development(self):
        events,types=self.population();types[3]='like'
        pool,_=h.eligible_holdout(events,types,{'train_end_timestamp':10,
            'minimum_prior_user_events':1,'alternatives':2,'target_event_type':'listen'},20,30)
        self.assertEqual(pool,[])

    def test_unsorted_or_overlapping_chronology_refused(self):
        events,types=self.population();config={'train_end_timestamp':10}
        with self.assertRaises(ValueError):h.eligible_holdout(events,types,config,10,30)
        with self.assertRaises(ValueError):h.eligible_holdout(events[::-1],types,config,20,30)

    def test_dictionary_is_invariant_to_equal_time_and_future_events(self):
        from tlab_warmup.representations import build_dictionary
        events=np.array([(1,10,1),(1,11,2),(1,12,3),(1,10,20),(1,11,21)],dtype=np.int64)
        queries=[{'uid':1,'timestamp':20,'positive':10,'candidates':[10,11,12]}]
        changed=events.copy();changed[changed[:,2]>=20,0:2]=999
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);build_dictionary(events,queries,p/'a.npy');build_dictionary(changed,queries,p/'b.npy')
            np.testing.assert_allclose(np.load(p/'a.npy'),np.load(p/'b.npy'),rtol=0,atol=1e-12)

    def test_model_tamper_refused_before_holdout_started(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'final';out.mkdir();run=root/'run';run.mkdir()
            (run/'manifest.json').write_text('{}');(root/'model.joblib').write_bytes(b'changed')
            f={'output':'final','engine_code_sha256':'code','evaluator_sha256':h.sha(Path(h.__file__)),
               'parent_run':'run','parent_run_manifest_sha256':h.sha(run/'manifest.json'),
               'models':[{'model':'model.joblib','model_sha256':'0'*64}]}
            (out/'freeze.json').write_text(json.dumps(f))
            with patch.object(h,'setup',return_value='code'):
                with self.assertRaisesRegex(ValueError,'Fitted model changed'):
                    h.execute(root,out/'freeze.json')
            self.assertFalse((out/'STARTED.json').exists())

    def test_repeated_execution_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'final';out.mkdir();(out/'STARTED.json').write_text('{}')
            (out/'freeze.json').write_text(json.dumps({'output':'final'}))
            with self.assertRaisesRegex(ValueError,'already started'):h.execute(root,out/'freeze.json')

if __name__=='__main__':unittest.main()
