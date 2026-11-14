import unittest
from unittest.mock import patch
import numpy as np
from env.route_following_trials import PreparedNavigation, selection_reward, RouteTrialEnv
from env.observation_reward_trials import TrialEnv

class RouteFollowing(unittest.TestCase):
    def test_baseline_and_reward_condition_preserve_observation(self):
        original=np.arange(63,dtype=np.float32)
        for condition in ('baseline','route_reward'):
            env=object.__new__(RouteTrialEnv);env.route_condition=condition
            with patch.object(TrialEnv,'_get_obs',return_value=original):
                np.testing.assert_array_equal(env._get_obs(),original)

    def test_guidance_does_not_alter_action_or_reward(self):
        env=object.__new__(RouteTrialEnv);env.route_condition='guidance'
        action=np.array([.2,-1],dtype=np.float32)
        with patch.object(TrialEnv,'step',return_value=(np.zeros(63),.5,False,False,{})) as parent:
            result=env.step(action)
        self.assertIs(parent.call_args.args[0],action)
        self.assertEqual(result[1],.5)

    def test_maintenance_bonus_bounded_and_progress_required(self):
        state={}
        self.assertAlmostEqual(selection_reward(state,('a',0),0,0,1000,False),.5)
        self.assertEqual(selection_reward(state,('a',0),0,0,1000,False),0)
        self.assertEqual(selection_reward({},('b',0),0,0,-1,False),0)

    def test_prepare_after_single_lane_merge_is_explicit(self):
        nav=PreparedNavigation({'a':1,'b':4,'c':3},[('a','b',0,3),('b','c',0,0),('b','c',1,1),('b','c',2,2)],{'a':100,'b':50,'c':200},horizon=300)
        old=nav.query(['a','b','c'],0,0,100)
        self.assertTrue(old['ambiguous'])
        new=nav.prepared(['a','b','c'],0,0,100)
        self.assertEqual(new['prepare_targets'],[0])
        self.assertEqual(new['future_direction'],-1)
        self.assertEqual(new['future_targets'],[0,1,2])
        self.assertEqual(new['distance'],150)
        self.assertTrue(new['valid'])

    def test_prepare_prefers_upstream_lane_that_avoids_future_changes(self):
        nav=PreparedNavigation({'a':2,'b':2,'c':1},[('a','b',0,0),('a','b',1,1),('b','c',1,0)],{'a':100,'b':50,'c':100})
        new=nav.prepared(['a','b','c'],0,0,100)
        self.assertEqual(new['prepare_targets'],[1])

    def test_missing_connection_not_valid_and_internal_not_guessed(self):
        nav=PreparedNavigation({'a':1,'b':2,'c':1},[('b','c',1,0)],{'a':100,'b':50,'c':100})
        self.assertFalse(nav.prepared(['a','b','c'],0,0,100)['valid'])
        self.assertFalse(nav.prepared(['a','b','c'],0,0,100,True)['valid'])

    def test_reward_has_no_stationary_or_oscillation_bonus(self):
        state={}
        # Entry on wrong lane establishes initial distance, earns nothing.
        self.assertEqual(selection_reward(state,('a',0),2,2,0,False),0)
        self.assertAlmostEqual(selection_reward(state,('a',0),2,1,0,False),.1)
        self.assertLess(selection_reward(state,('a',0),0,1,0,False),0)
        # Revisiting already achieved lane distance does not repay progress.
        self.assertEqual(selection_reward(state,('a',0),2,1,0,False),0)
        self.assertEqual(selection_reward(state,('a',0),0,0,0,False),0)
        self.assertEqual(selection_reward(state,('a',0),1,0,10,True),0)

if __name__=='__main__':unittest.main()
