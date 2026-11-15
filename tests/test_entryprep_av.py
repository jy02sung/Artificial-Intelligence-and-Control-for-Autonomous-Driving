"""Release AV observation and score regressions using controlled simulator readings."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from env.entryprep_av import EntryEnv, ROUTES, occupied_crossing, road_inputs
from env.gap_reward import gap_reward, opportunity
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

    def test_crossing_observation_uses_local_vehicle_occupancy(self):
        self.assertTrue(occupied_crossing(20., 15., 9., 18., 12., 5.))
        self.assertFalse(occupied_crossing(20., 15., 74., 89., 12., 5.))
        self.assertFalse(occupied_crossing(20., 15., -18., -10., 12., 5.))
        self.assertFalse(occupied_crossing(20., 15., 30., 40., 0., 5.))

    def test_gap_reward_depends_on_actual_lane_change_and_preceding_gap(self):
        obs = np.zeros(57)
        obs[0], obs[31], obs[51], obs[52] = 10. / 27.78, 1. / 3., 1., .12
        obs[5], obs[7] = .4, -.4  # Both neighbours 20 m away, travelling at ego speed.
        free = opportunity(obs, 27.78, 50.)
        self.assertTrue(free.available)
        self.assertGreater(gap_reward(free, 10., 1, 1, .2), 0.)
        self.assertLessEqual(gap_reward(free, 10., 1, 0, .2), 0.)
        obs[7], obs[8] = -.08, .3  # Rear car is now 4 m away and closing.
        occupied = opportunity(obs, 27.78, 50.)
        self.assertFalse(occupied.available)
        self.assertLess(gap_reward(occupied, 10., 1, 1, .2), 0.)
        self.assertLess(gap_reward(occupied, 10., 0, 0, .2),
                        gap_reward(occupied, occupied.approach_speed, 0, 0, .2))
        obs[51] = 0.1
        no_turn = opportunity(obs, 27.78, 50.)
        self.assertIsNone(no_turn)
        self.assertLess(gap_reward(no_turn, 10., -1, -1, .2), gap_reward(no_turn, 10., 0, 0, .2))

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

    @patch("env.entryprep_av.traci.vehicle")
    @patch("env.entryprep_av.ClarityEnv.step")
    def test_gap_bonus_cannot_be_earned_again_by_returning_to_the_same_lane(self, parent_step, vehicle):
        env = self.sensor()
        env.condition = "gap"
        env.vmax = 27.78
        env.step_count = 0
        obs = np.zeros(57, dtype=np.float32)
        obs[0], obs[31], obs[51], obs[52] = 10. / env.vmax, 1. / 3., 1., .12
        obs[5], obs[7] = .4, -.4
        env.last_observation = obs
        vehicle.getRoadID.return_value = "weave"
        vehicle.getRouteIndex.return_value = 2
        parent_step.return_value = (obs, 0., False, False,
                                    dict(lane_change=1, lane_before=1, lane_after=2))

        first_reward = env.step(np.array([0., 1.]))[1]
        env.step_count = 2  # Same episode, after moving back to the previous lane.
        repeated_reward = env.step(np.array([0., 1.]))[1]
        self.assertAlmostEqual(first_reward - repeated_reward, 1.25)

        env.step_count = 0  # A new episode can earn its own progress bonus.
        self.assertAlmostEqual(env.step(np.array([0., 1.]))[1], first_reward)

    @patch("env.entryprep_av.traci.lane.getLength", return_value=55.)
    @patch("env.entryprep_av.traci.vehicle")
    @patch("env.entryprep_av.ClarityEnv._get_obs")
    def test_gap_context_inputs_preserve_existing_model_observations(self, parent_obs, vehicle, lane_length):
        env = self.sensor()
        env.condition = "gap_context"
        env.vmax, env.max_decel = 27.78, 5.4
        env._neighbor = Mock(return_value=None)
        base = np.zeros(47, dtype=np.float32)
        base[0], base[31], base[32], base[24] = 10. / env.vmax, -1. / 3., .1, 1.
        base[13], base[15] = .5, -.5
        parent_obs.return_value = base
        vehicle.getIDList.return_value = ["ego"]
        vehicle.getSpeed.return_value = 10.
        vehicle.getLeader.return_value = ("", -1.)
        vehicle.getRoadID.return_value = "weave"
        vehicle.getLaneID.return_value = "weave_1"
        vehicle.getLaneIndex.return_value = 1
        vehicle.getBestLanes.return_value = [(None, None, None, 0)] * 2
        vehicle.getLanePosition.return_value = 5.
        vehicle.getDistance.return_value = 100.
        vehicle.getJunctionFoes.return_value = ()

        obs = env._get_obs()
        self.assertEqual(obs.shape, (57,))
        np.testing.assert_allclose(obs[37:41], [-1., 1., -2. / 5.4, 1.], atol=1e-6)
        np.testing.assert_array_equal(env.last_observation, obs)
        env.condition = "merge"
        previous = env._get_obs()
        np.testing.assert_array_equal(previous[37:41], np.zeros(4))
        np.testing.assert_array_equal(obs[:37], previous[:37])
        np.testing.assert_array_equal(obs[41:], previous[41:])

    def test_gap_input_training_preserves_actor_outputs_without_a_required_lane_change(self):
        import torch
        from utils.entryprep_av import train_gap_input_weights
        from utils.networks import HybridActorCritic

        torch.manual_seed(22)
        policy = HybridActorCritic(57, hidden_sizes=(16, 16), activation="relu")
        with torch.no_grad():
            policy.trunk[0].weight[:, 37:41].zero_()
        ordinary = torch.rand(8, 57)
        ordinary[:, 37:41] = 0.
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        with torch.no_grad():
            original_accel, _, original_lanes, _ = policy(ordinary)
        train_gap_input_weights(policy)
        examples = ordinary.clone()
        examples[:, 37:41] = torch.tensor([1., -1., -.8, 1.])
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        for _ in range(3):
            accel, _, lanes, value = policy(examples)
            loss = accel.sum() + lanes.square().sum() + value.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for name, weight in policy.state_dict().items():
            if name == "trunk.0.weight":
                self.assertTrue(torch.equal(weight[:, :37], before[name][:, :37]))
                self.assertTrue(torch.equal(weight[:, 41:], before[name][:, 41:]))
                self.assertFalse(torch.equal(weight[:, 37:41], before[name][:, 37:41]))
            elif not name.startswith("v_head."):
                self.assertTrue(torch.equal(weight, before[name]), name)
        with torch.no_grad():
            accel, _, lanes, _ = policy(ordinary)
        self.assertTrue(torch.equal(accel, original_accel))
        self.assertTrue(torch.equal(lanes, original_lanes))

    def test_lane_head_training_preserves_ordinary_acceleration(self):
        import torch
        from utils.entryprep_av import train_gap_input_weights
        from utils.networks import HybridActorCritic

        torch.manual_seed(22)
        policy = HybridActorCritic(57, hidden_sizes=(16, 16), activation="relu")
        ordinary = torch.rand(8, 57)
        ordinary[:, 37:41] = 0.
        with torch.no_grad():
            original_accel = policy(ordinary)[0].clone()
            original_lanes = policy.lane_head.weight.clone()
        train_gap_input_weights(policy, train_lanes=True)
        examples = ordinary.clone()
        examples[:, 37:41] = torch.tensor([-1., 1., -.4, 1.])
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        _, _, logits, value = policy(examples)
        loss = -logits[:, 1].mean() + value.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.assertTrue(torch.equal(policy(ordinary)[0], original_accel))
            self.assertFalse(torch.equal(policy.lane_head.weight, original_lanes))


if __name__ == "__main__":
    unittest.main()
