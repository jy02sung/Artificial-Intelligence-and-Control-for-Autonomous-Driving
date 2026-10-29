import unittest
import numpy as np
from collect import observation, applied_action

def car(edge='road',lane=0,speed=10,x=0):
    return dict(edge=edge,lane=lane,speed=speed,x=x,y=0,angle=0,position=10)

class CollectionTests(unittest.TestCase):
    def test_nearby_only_and_fixed_shape(self):
        obs,xy=observation({'ego':car(),'near':car(x=20),'far':car(x=80)},'nw_to_east',{'road_0':100},{'road':2})
        self.assertEqual(obs.shape,(54,))
        self.assertEqual(xy.shape,(1,2))
        self.assertTrue(np.isfinite(obs).all())
    def test_edge_transition_is_not_lane_change(self):
        action,valid=applied_action(car(lane=3),car(edge='next',lane=0,speed=11),.2)
        self.assertAlmostEqual(action[0],5)
        self.assertFalse(valid[1])
        self.assertEqual(action[1],0)
    def test_terminal_missing_state_mask(self):
        action,valid=applied_action(car(),None,.2)
        self.assertFalse(any(valid))
        self.assertTrue(np.array_equal(action,[0,0]))

if __name__=='__main__':unittest.main()
