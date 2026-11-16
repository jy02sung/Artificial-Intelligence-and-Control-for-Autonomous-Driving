import unittest
from env.risk_action_trial import acceleration_risk_penalty
class RiskActionTests(unittest.TestCase):
    def test_only_risky_positive_acceleration_costs(self):
        self.assertEqual(acceleration_risk_penalty(0,1,False),0)
        self.assertEqual(acceleration_risk_penalty(1,-1,False),0)
        self.assertAlmostEqual(acceleration_risk_penalty(.5,.5,False),-.1)
    def test_applied_acceleration_clipped(self):
        self.assertAlmostEqual(acceleration_risk_penalty(1,3,False),-.4)
    def test_terminal_keeps_original_terminal_reward(self):
        self.assertEqual(acceleration_risk_penalty(1,1,True),0)
if __name__=='__main__':unittest.main()
