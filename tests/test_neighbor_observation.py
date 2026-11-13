"""Regression cases from actual adjacent-vehicle collision traces."""
import unittest
from unittest.mock import patch
import numpy as np
from env import sumo_env as module


class NeighborObservation(unittest.TestCase):
    def setUp(self):
        self.env=object.__new__(module.SumoHighwayEnv)
        self.env.ego_id='ego';self.env.visibility=50.
        self.env.vmax=27.78;self.env.obs_dim=31
        self.env.obs_offsets=[2,1,0,-1,-2]

    def test_adjacent_real_vehicle_negative_gap_is_present(self):
        for offset in (-1,1):
            for leader in (False,True):
                with self.subTest(offset=offset,leader=leader),patch.object(module.traci.vehicle,'getNeighbors',return_value=(('near',-6.704188),('far',20.))),patch.object(module.traci.vehicle,'getSpeed',return_value=24.58474):
                    self.assertEqual(self.env._neighbor(offset,leader),(0.,24.58474))

    def test_current_lane_negative_gap_with_id_is_present(self):
        with patch.object(module.traci.vehicle,'getLeader',return_value=('near',-1.)),patch.object(module.traci.vehicle,'getFollower',return_value=('near',-1.)),patch.object(module.traci.vehicle,'getSpeed',return_value=10.):
            self.assertEqual(self.env._neighbor(0,True),(0.,10.))
            self.assertEqual(self.env._neighbor(0,False),(0.,10.))

    def test_two_lanes_away_overlap_is_present(self):
        with patch.object(self.env,'_scan_lane_neighbors',return_value=((-2.,12.),(-1.,30.))):
            for offset in (-2,2):
                self.assertEqual(self.env._neighbor(offset,True),(0.,12.))
                self.assertEqual(self.env._neighbor(offset,False),(0.,30.))

    def test_absent_ids_and_out_of_range_remain_absent(self):
        with patch.object(module.traci.vehicle,'getLeader',return_value=('',-1.)),patch.object(module.traci.vehicle,'getFollower',return_value=('',-1.)),patch.object(module.traci.vehicle,'getNeighbors',return_value=()):
            self.assertIsNone(self.env._neighbor(0,True))
            self.assertIsNone(self.env._neighbor(0,False))
            self.assertIsNone(self.env._neighbor(1,True))
        with patch.object(module.traci.vehicle,'getNeighbors',return_value=(('far',50.01),)):
            self.assertIsNone(self.env._neighbor(-1,False))

    def test_positive_gaps_and_relative_speed_reach_policy_unchanged(self):
        for speed in (10.,35.):
            with self.subTest(speed=speed),patch.object(module.traci.vehicle,'getNeighbors',return_value=(('other',15.),)),patch.object(module.traci.vehicle,'getSpeed',return_value=speed):
                self.assertEqual(self.env._neighbor(1,True),(15.,speed))

    def test_overlap_is_not_encoded_as_empty_in_policy_input(self):
        def speed(vid):return 25. if vid=='ego' else 10.
        with patch.object(module.traci.vehicle,'getIDList',return_value=['ego','other']),patch.object(module.traci.vehicle,'getSpeed',side_effect=speed),patch.object(module.traci.vehicle,'getLeader',return_value=None),patch.object(module.traci.vehicle,'getFollower',return_value=('',-1.)),patch.object(module.traci.vehicle,'getNeighbors',return_value=(('other',-3.),)),patch.object(self.env,'_scan_lane_neighbors',return_value=(None,None)),patch.object(self.env,'_lane_connectivity',return_value=1.),patch.object(self.env,'_lane_density',return_value=0.):
            obs=self.env._get_obs()
        self.assertEqual(obs.shape,(31,))
        self.assertTrue(np.isfinite(obs).all())
        # Offset +1 occupies indices 5..8; approaching front, slower rear.
        np.testing.assert_allclose(obs[5:9],[0.,-15/27.78,0.,-15/27.78],atol=1e-7)
        np.testing.assert_equal(obs[9:13],[1.,1.,-1.,-1.])

if __name__=='__main__':unittest.main()
