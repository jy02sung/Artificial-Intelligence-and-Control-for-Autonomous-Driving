import unittest
from unittest.mock import patch
import numpy as np
from env.observation_reward_trials import TrialEnv, approach_risk, lane_deficit
from env.navigation_clarity import ClarityEnv

class TrialFeatures(unittest.TestCase):
    def test_lane_reward_never_changes_action_or_terminal_reward(self):
        for done,base,expected in ((False,.04,.02),(True,-5.,-5.),(True,2.,2.)):
            env=object.__new__(TrialEnv);env.condition='reward_lane';env.ego_id='ego';env.step_length=.2;env.lane_change_index=1
            action=np.array([.7,1.],dtype=np.float32)
            with self.subTest(done=done,base=base),patch('env.observation_reward_trials.traci.vehicle.getIDList',return_value=['ego']),patch('env.observation_reward_trials.traci.vehicle.getSpeed',return_value=20.),patch.object(env,'_quantize_lane_change',return_value=1),patch.object(env,'_neighbor',return_value=(0.,20.)),patch.object(ClarityEnv,'step',return_value=(np.zeros(63),base,done,False,{})) as parent:
                result=env.step(action)
            self.assertAlmostEqual(result[1],expected)
            self.assertIs(parent.call_args.args[0],action)
            np.testing.assert_array_equal(action,np.array([.7,1.],dtype=np.float32))

    def test_closing_risk_and_absence(self):
        self.assertEqual(approach_risk(None,20,True),0.)
        self.assertEqual(approach_risk((0,25),20,True),1.)
        self.assertEqual(approach_risk((30,10),20,True),1.)
        self.assertEqual(approach_risk((60,10),20,True),.5)
        self.assertEqual(approach_risk((10,25),20,True),0.)
        self.assertEqual(approach_risk((60,40),20,False),1.)

    def test_requested_lane_deficit_uses_front_and_rear_speeds(self):
        self.assertEqual(lane_deficit(None,None,20),0.)
        self.assertEqual(lane_deficit((0,10),None,20),1.)
        self.assertAlmostEqual(lane_deficit((16.25,10),None,20),.5)
        self.assertAlmostEqual(lane_deficit(None,(31.25,40),20),.5)
        self.assertEqual(lane_deficit((100,10),(100,40),20),0.)

    def test_conditions_have_fixed_shape_and_disjoint_channels(self):
        snapshots={}
        for condition in ('baseline','obs_presence','obs_risk','reward_gap','reward_block','reward_lane'):
            env=object.__new__(TrialEnv);env.condition=condition;env.ego_id='ego';env.obs_offsets=[2,1,0,-1,-2]
            with patch.object(ClarityEnv,'_get_obs',return_value=np.zeros(47,dtype=np.float32)),patch('env.observation_reward_trials.traci.vehicle.getIDList',return_value=['ego']),patch('env.observation_reward_trials.traci.vehicle.getSpeed',return_value=20.),patch.object(env,'_neighbor',side_effect=lambda offset,leader: (0.,10.) if leader else None):
                snapshots[condition]=env._get_obs()
            self.assertEqual(snapshots[condition].shape,(63,))
            self.assertTrue(np.isfinite(snapshots[condition]).all())
        for condition in ('baseline','reward_gap','reward_block','reward_lane'):
            np.testing.assert_array_equal(snapshots[condition],np.zeros(63))
        np.testing.assert_array_equal(snapshots['obs_presence'][47:57],[1,0]*5)
        np.testing.assert_array_equal(snapshots['obs_presence'][57:],np.zeros(6))
        np.testing.assert_array_equal(snapshots['obs_risk'][47:57],np.zeros(10))
        np.testing.assert_array_equal(snapshots['obs_risk'][57:],[1,0]*3)

if __name__=='__main__':unittest.main()
