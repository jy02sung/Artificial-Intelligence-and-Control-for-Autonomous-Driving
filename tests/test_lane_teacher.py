import unittest
import numpy as np
from utils.probe_lane_teacher import candidate_action

class LaneTeacherTests(unittest.TestCase):
    def test_waits_for_unsafe_target(self):
        x=candidate_action(np.array([.5,1.]),dict(needed=1),[0,0,0],{1:.9,-1:0},[-1,1])
        self.assertEqual(x[1],0)
    def test_allows_safe_required_change(self):
        x=candidate_action(np.array([.5,0.]),dict(needed=-1),[0,0,0],{-1:0},[-1])
        self.assertEqual(x[1],-1)
    def test_no_invalid_lane_or_hidden_clock(self):
        x=candidate_action(np.array([.5,1.]),dict(needed=1),[.5,.8,.2],{},[])
        self.assertEqual(x[1],0);self.assertEqual(x[0],-.8)

if __name__=='__main__':unittest.main()
