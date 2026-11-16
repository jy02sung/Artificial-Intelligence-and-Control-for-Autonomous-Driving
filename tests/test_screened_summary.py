import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from utils.summarize_live_trials import read_screened

class ScreenedSummaryTest(unittest.TestCase):
    def prepare(self,root):
        d=root/'early-evaluation/51200/combined';d.mkdir(parents=True)
        cp=root/'runs/combined/policy-51200.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'frozen')
        rows=[dict(condition='combined',checkpoint=cp.name,route=f'route_{i:02d}',seed=79101,traffic='reference',outcome='collision',route_overrides=0) for i in range(1,4)]
        partial=d/'policy-51200-partial.json';partial.write_text(json.dumps(rows))
        note=d/'policy-51200-screen-rejected.json';note.write_text(json.dumps(dict(checkpoint=cp.name,checkpoint_sha=hashlib.sha256(cp.read_bytes()).hexdigest(),partial_sha=hashlib.sha256(partial.read_bytes()).hexdigest(),observed_traffic=3,planned_traffic=48,counts={'collision':3})))
        return note,partial

    def test_partial_keeps_observed_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);note,_=self.prepare(root)
            result=read_screened(root,note)
            self.assertEqual(result['observed'],3)
            self.assertEqual(result['counts'],{'collision':3})
            self.assertNotIn('completion_rate',result)

    def test_changed_partial_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);note,partial=self.prepare(root);partial.write_text('[]')
            with self.assertRaises(AssertionError):read_screened(root,note)

if __name__=='__main__':unittest.main()
