import unittest
from env.prepare_reward import PrepareCredit, PreparePressure, lane_change_risk, approach_risk


class PrepareTests(unittest.TestCase):
    def test_progress_only_and_no_oscillation_farming(self):
        r = PrepareCredit()
        self.assertEqual(r.award('a', 'b', [0], 3, 2, 250), .1)
        self.assertEqual(r.award('a', 'b', [0], 2, 3, 240), 0)
        self.assertEqual(r.award('a', 'b', [0], 3, 2, 230), 0)
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 220), .1)

    def test_range_and_compatible_lanes(self):
        r = PrepareCredit()
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 301), 0)
        self.assertEqual(r.award('a', 'b', [], 2, 1, 100), 0)
        self.assertEqual(r.award('a', 'b', [0,1], 0, 1, 100), 0)
        self.assertEqual(r.award('a', 'b', [0], 2, 1, 300), .1)


class PressureTests(unittest.TestCase):
    def test_zero_when_compatible_or_out_of_range(self):
        p = PreparePressure()
        self.assertEqual(p.penalty([1, 2], 2, 50), 0)
        self.assertEqual(p.penalty([1], 0, 301), 0)
        self.assertEqual(p.penalty([1], 0, None), 0)
        self.assertEqual(p.penalty([], 0, 50), 0)

    def test_grows_with_proximity_and_remaining_changes(self):
        p = PreparePressure(weight=.02)  # explicit weight for readable numbers
        self.assertAlmostEqual(p.penalty([3], 2, 300), 0)
        self.assertAlmostEqual(p.penalty([3], 2, 150), .01)
        self.assertAlmostEqual(p.penalty([3], 2, 0), .02)
        self.assertAlmostEqual(p.penalty([3], 1, 0), .04)
        self.assertAlmostEqual(p.penalty([0, 1, 2], 3, 0), .02)

    def test_capped_below_speed_reward(self):
        # Default: one owed change at the split costs 0.01/step, below the
        # 0.1 max speed reward, so it cannot outweigh driving normally.
        self.assertAlmostEqual(PreparePressure().penalty([0], 1, 0), .01)


class LaneChangeRiskTests(unittest.TestCase):
    def test_empty_lane_and_safe_gaps_are_free(self):
        self.assertEqual(lane_change_risk(None, None, 20), 0)
        # far leader slower, far follower slower: no risk
        self.assertEqual(lane_change_risk((80, 15), (60, 18), 20), 0)

    def test_fast_follower_closing_in(self):
        # follower 10 m behind, 10 m/s faster: TTC 1 s of 3 s horizon
        self.assertAlmostEqual(lane_change_risk(None, (10, 30), 20), 2 / 3)

    def test_slow_leader_ahead(self):
        # leader 20 m ahead, ego 10 m/s faster: TTC 2 s
        self.assertAlmostEqual(lane_change_risk((20, 10), None, 20), 1 / 3)

    def test_tiny_gap_is_full_risk_even_without_closing(self):
        self.assertEqual(lane_change_risk((0, 20), None, 20), 1)
        self.assertAlmostEqual(lane_change_risk(None, (2.5, 10), 20), .5)

    def test_worse_side_wins(self):
        self.assertAlmostEqual(lane_change_risk((20, 10), (10, 30), 20), 2 / 3)


class ApproachRiskTests(unittest.TestCase):
    def test_no_leader_or_not_closing_is_free(self):
        self.assertEqual(approach_risk(None, 25), 0)
        self.assertEqual(approach_risk((5, 20), 20), 0)
        self.assertEqual(approach_risk((5, 25), 20), 0)

    def test_closing_on_slow_leader(self):
        # 30 m gap, 15 m/s closing: TTC 2 s of 3 s horizon
        self.assertAlmostEqual(approach_risk((30, 10), 25), 1 / 3)
        # meeting-log case: 4.73 m gap, ~27.8 vs ~2.2 m/s -> near-certain contact
        self.assertGreater(approach_risk((4.73, 2.2), 27.8), .9)

    def test_far_enough_is_free(self):
        self.assertEqual(approach_risk((45, 20), 30), 0)  # TTC 4.5 s


if __name__ == '__main__':
    unittest.main()
