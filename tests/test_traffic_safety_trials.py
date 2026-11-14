import unittest
from env.traffic_safety_trials import gap_deficit,closing_risk
class Safety(unittest.TestCase):
 def test_gap(self):
  self.assertEqual(gap_deficit(None,20,2.5),0)
  self.assertEqual(gap_deficit((30,10),20,2.5),0)
  self.assertAlmostEqual(gap_deficit((0,10),20,2.5),1-2.5/32.5)
 def test_risk_direction_and_overlap(self):
  self.assertEqual(closing_risk((20,30),20,True,2.5),0)
  self.assertEqual(closing_risk((20,30),20,False,2.5),1)
  self.assertEqual(closing_risk((0,0),0,True,2.5),1)
class Isolation(unittest.TestCase):
 def test_reward_branches_do_not_replace_action(self):
  import collections
  import numpy as np
  from unittest.mock import patch
  from env.traffic_safety_trials import TrafficSafetyEnv
  from env.route_learning_trials import RouteLearningEnv
  for condition,expected in [('baseline',.5),('gap',.45),('lane',.41),('combo',.36)]:
   env=object.__new__(TrafficSafetyEnv);env.safety_condition=condition;env.reward_counts=collections.Counter()
   env.safety_context=lambda a:dict(gap=.5,lane=.9,ttc=2.)
   action=np.array([-.5,1.])
   info=dict(route_selection_reward=0.,route_reward_parts={})
   with patch.object(RouteLearningEnv,'step',return_value=(np.zeros(63),.5,False,False,info)) as parent:
    result=env.step(action)
    self.assertIs(parent.call_args.args[0],action)
    self.assertAlmostEqual(result[1],expected)
if __name__=='__main__':unittest.main()
