import unittest
from env.safety_strength_trials import safety_extra

class SafetyStrengthTest(unittest.TestCase):
    def test_collision_only_terminal_penalty(self):
        self.assertEqual(safety_extra('collision20', True, True, 1, 1, 1), dict(collision=-15., lane=0., merge=0.))
        self.assertEqual(sum(safety_extra('collision20', False, True, 1, 1, 1).values()), 0.)
        self.assertEqual(sum(safety_extra('collision20', False, False, 1, 1, 1).values()), 0.)

    def test_lane_reward_is_independent_and_nonterminal(self):
        self.assertAlmostEqual(safety_extra('lane4', False, False, .5, 1, 1)['lane'], -.15)
        self.assertEqual(sum(safety_extra('lane4', True, True, 1, 1, 1).values()), 0.)
        self.assertEqual(sum(safety_extra('lane4', False, False, 0, 1, 1).values()), 0.)

    def test_combination_and_zero_risk(self):
        parts=safety_extra('anticipatory', False, False, .5, .8, .5)
        self.assertAlmostEqual(parts['lane'], -.15)
        self.assertAlmostEqual(parts['merge'], -.04)
        self.assertEqual(parts['collision'],0.)
        self.assertEqual(sum(safety_extra('anticipatory', False, False, 0, 0, 1).values()),0.)
        self.assertEqual(sum(safety_extra('anticipatory', True, True, 1, 1, 1).values()),-15.)

if __name__=='__main__': unittest.main()
