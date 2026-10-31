import unittest
from rule_policy import emergency_needed, safe_gap, is_violator

class RuleTests(unittest.TestCase):
    def test_no_emergency_for_queue_or_safe_braking(self):
        self.assertFalse(emergency_needed(0,0,2,9,.2,1.5))
        self.assertFalse(emergency_needed(10,5,20,9,.2,1.5))
    def test_emergency_requires_both_closing_risk_and_stopping_deficit(self):
        self.assertTrue(emergency_needed(20,0,8,9,.2,1.5))
        self.assertFalse(emergency_needed(5,20,1,9,.2,1.5))
    def test_target_gap_rejects_fast_follower(self):
        self.assertFalse(safe_gap(15,25,1.5,5))
        self.assertTrue(safe_gap(50,25,1.5,5))
    def test_assignment_is_reproducible_and_bus_excluded(self):
        self.assertEqual(is_violator('v',200,.05),is_violator('v',200,.05))
        self.assertFalse(is_violator('v',200,1,bus=True))

if __name__=='__main__':unittest.main()
