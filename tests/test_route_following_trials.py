import unittest
from env.route_following_trials import PreparedNavigation, selection_reward

class RouteFollowing(unittest.TestCase):
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
