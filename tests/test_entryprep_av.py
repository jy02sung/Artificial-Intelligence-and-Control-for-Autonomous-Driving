"""Release AV observation and score regressions using controlled simulator readings."""
import unittest
from unittest.mock import Mock, patch

from env.entryprep_av import EntryEnv, ROUTES, road_inputs
from utils.entryprep_score import score


class EntryprepTests(unittest.TestCase):
    def sensor(self):
        env = EntryEnv.__new__(EntryEnv)
        env.condition = "risk"
        env.ego_id = "ego"
        env.visibility = 50.
        return env

    @patch("env.entryprep_av.traci.vehicle")
    def test_real_overlapping_vehicle_is_visible(self, vehicle):
        env = self.sensor()
        vehicle.getLeader.return_value = ("hv", -1.5)
        vehicle.getSpeed.return_value = 16.
        self.assertEqual(env._neighbor(0, True), (0., 16.))
        vehicle.getLeader.return_value = ("", -1.)
        self.assertIsNone(env._neighbor(0, True))
        vehicle.getLeader.return_value = ("outside", 60.)
        self.assertIsNone(env._neighbor(0, True))

    @patch("env.entryprep_av.traci.vehicle")
    def test_adjacent_lane_scan_sees_newly_inserted_follower(self, vehicle):
        env = self.sensor()
        env._scan_lane_neighbors = Mock(return_value=(None, (-.5, 23.)))
        vehicle.getNeighbors.return_value = ()
        self.assertEqual(env._neighbor(1, False), (0., 23.))

    def test_all_released_routes_have_connected_edges(self):
        road, navigation, _ = road_inputs()
        self.assertEqual(set(road["routes"]), set(ROUTES))
        for route in road["routes"].values():
            for edge, following in zip(route, route[1:]):
                self.assertTrue(navigation.links[edge, following])

    def test_five_collisions_in_a_hundred_fails_even_with_95_arrivals(self):
        rows = [dict(route=route, seed=i, outcome="arrival") for route in ROUTES for i in range(100)]
        for row in rows[:5]:
            row["outcome"] = "collision"
        report = score(rows, range(100))
        self.assertFalse(report[0]["rates_pass"])
        rows[4]["outcome"] = "insertion_failure"
        report = score(rows, range(100))
        self.assertTrue(report[0]["rates_pass"])
        self.assertEqual(report[0]["arrival_pct"], 95.)
        self.assertEqual(report[0]["collision_pct"], 4.)
        with self.assertRaises(ValueError):
            score(rows[1:], range(100))
        with self.assertRaises(ValueError):
            score(rows + rows[:1], range(100))


if __name__ == "__main__":
    unittest.main()
