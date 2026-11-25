"""SUMO integration checks for lane-driven branch traversal."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_physical_forks as experiment


class PhysicalForksTest(unittest.TestCase):
    def test_first_fork(self):
        with TemporaryDirectory() as temp:
            env = experiment.Env(experiment.ARMS[0], Path(temp), 11)
            try:
                for route, lane, switch, expected, correct in (
                    (13, 0, 0, "37927508", True),
                    (13, 1, 0, "86198725", False),
                    (16, 0, 0, "37927508", False),
                    (16, 1, 0, "86198725", True),
                    (16, 0, 1, "86198725", True),
                    (13, 0, 1, "86198725", False),
                ):
                    with self.subTest(route=route, lane=lane, switch=switch):
                        env.ego_depart_lane = str(lane)
                        obs, _ = env.reset(seed=41110016, options={"route": f"route_{route:02d}"})
                        mission = tuple(env.routes[env.episode_route])
                        veh = experiment.j.traci.vehicle
                        for other in veh.getIDList():
                            if other != env.ego_id:
                                veh.remove(other)
                        lid = veh.getLaneID(env.ego_id)
                        veh.moveTo(env.ego_id, lid, experiment.j.traci.lane.getLength(lid) - 75.)
                        obs = env.observe(env.sim._get_obs())
                        switched = False
                        for _ in range(400):
                            edge, actual_lane, path, idx, fork, _, _, _, remaining = env.snapshot
                            self.assertEqual(tuple(path), mission)
                            if edge == mission[0] and fork == 0:
                                self.assertEqual(obs[34], env.signs[mission[0], mission[1]])
                            command = switch if not switched and remaining < 60 and edge == mission[0] else 0
                            switched |= bool(command)
                            obs, reward, done, cut, info = env.step([.05, command])
                            if info["fork_passed"] or done or cut:
                                break
                        self.assertEqual(env.snapshot[0], expected)
                        self.assertEqual(info["fork_passed"], correct)
                        self.assertEqual(info["off_route"], not correct)
                        self.assertEqual(done, not correct)
                        self.assertFalse(cut)
                        if correct:
                            self.assertAlmostEqual(reward - info["progress_bonus"], 1.)
                        else:
                            self.assertEqual(reward, -5.)
                            self.assertEqual(info["off_route_detail"]["expected"], mission[1])
            finally:
                env.close()


    def test_network_forks(self):
        # Isolate geometry at each fork, including later forks and shared lanes.
        with TemporaryDirectory() as temp:
            env = experiment.Env(experiment.ARMS[0], Path(temp), 11)
            checked = set()
            try:
                for name, mission in env.routes.items():
                    for index, (src, expected) in enumerate(zip(mission, mission[1:])):
                        exits = env.links[src]
                        if len(exits) < 2:
                            continue
                        for lane in sorted(set.union(*map(set, exits.values()))):
                            key = src, expected, lane
                            if key in checked:
                                continue
                            checked.add(key)
                            with self.subTest(src=src, mission_next=expected, lane=lane):
                                env.reset(seed=41110016, options={"route":name})
                                veh = experiment.j.traci.vehicle
                                for other in veh.getIDList():
                                    if other != env.ego_id:
                                        veh.remove(other)
                                lid = f"{src}_{lane}"
                                veh.moveTo(env.ego_id, lid, experiment.j.traci.lane.getLength(lid) - 20.)
                                # Prepare the matching mission history for this isolated approach.
                                env.sim.last_external = index, src
                                env.sim._get_obs()
                                choices = [dst for dst, lanes in exits.items() if lane in lanes]
                                actual = expected if expected in choices else choices[0]
                                for _ in range(180):
                                    _, reward, done, cut, info = env.step([.3, 0.])
                                    if done or cut or info["fork_passed"]:
                                        break
                                self.assertEqual(env.snapshot[0], actual)
                                self.assertEqual(info["off_route"], actual != expected)
                                self.assertEqual(info["fork_passed"], actual == expected)
                                self.assertEqual(done, actual != expected)
                                if actual != expected:
                                    self.assertEqual(reward, -5.)
                                    self.assertEqual(info["off_route_detail"]["src"], src)
                                else:
                                    self.assertEqual(env.snapshot[3], index+1)
            finally:
                env.close()
            self.assertEqual(len({src for src, _, _ in checked}), 7)
            self.assertEqual(len(checked), 46)


if __name__ == "__main__":
    unittest.main()
