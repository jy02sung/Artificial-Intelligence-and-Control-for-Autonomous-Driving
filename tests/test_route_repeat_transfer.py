import unittest
from unittest.mock import patch
from utils import run_route_repeat_transfer as r
class TransferSetup(unittest.TestCase):
    def test_training_empty_evaluation_respects_traffic(self):
        with patch.object(r.x,'original_make',return_value=object()) as factory:
            r.make_env('baseline',evaluation=False,none=False)
            self.assertTrue(factory.call_args.kwargs['none'])
            r.make_env('baseline',evaluation=True,none=False)
            self.assertFalse(factory.call_args.kwargs['none'])
            r.make_env('baseline',evaluation=True,none=True)
            self.assertTrue(factory.call_args.kwargs['none'])
    def test_holdout_seeds_and_control(self):
        self.assertEqual(r.SEEDS,(22,33))
        self.assertEqual(r.CONDITIONS,('baseline','guidance','dwell'))
        self.assertTrue(set(r.EVAL_SEEDS).isdisjoint({19101,19102,19103}))
if __name__=='__main__':unittest.main()
