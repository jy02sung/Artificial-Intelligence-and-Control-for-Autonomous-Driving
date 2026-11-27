import os
os.environ.setdefault("IC_LIBSUMO", "1")
from utils import fork_geometry as v2
import utils.driving_env as j
import utils.route_progress as core
from utils.route_progress import FactorEnv as BaseEnv
ARMS = ("S1_dir_col5","S2_dir_col7")
from collections import Counter
import json
import math
import xml.etree.ElementTree as ET


def setup(arm):
    j.ARRIVAL_REWARD, j.PASS_REWARD, j.TIMEOUT_REWARD = 5., 1., -5.
    j.COLLISION_REWARD = dict(zip(ARMS, (-5., -7.)))[arm]
    core.FactorEnv = Env


class Env(BaseEnv):
    progress_cap = 2.
    off_route_penalty = -5.
    idle_clear = pot = 0.

    def __init__(self, arm, folder, seed):
        setup(arm)
        super().__init__("F1", folder, seed)
        self.arm = arm
        self.sums = Counter()
        self.outcome_log = None
        net = folder / "high_real_ratio" / "network.net.xml"
        v2.configure_roads(self.sim, net)
        self.road_types = {e.get("id"): e.get("type","") for e in ET.parse(net).findall("edge")}
        assert self.progress_enabled and not self.nav_enabled and self.sim.max_steps == 1000
        assert self.visibility == 120 and self.obs_noise and self.occlusion and not self.lc_safety_enabled

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_off_route = None
        self.assigned_path = self.routes[self.episode_route]
        self.current_seed = kwargs.get("seed")
        assert len(obs) == 52 and tuple(obs[-2:]) == (0., 0.)
        return obs, info

    def departure(self, previous, previously_intact, info):
        if self.snapshot is None or info["collided"] or not previously_intact:
            return None
        actual = self.snapshot[0]
        index, src = previous
        if actual == src or actual.startswith(":"):
            return None
        path = self.assigned_path
        expected = path[index+1] if index+1 < len(path) else None
        if actual == expected:
            return None
        if (self.ego_id in j.traci.simulation.getStartingTeleportIDList()
                or self.ego_id in j.traci.simulation.getEndingTeleportIDList()):
            return None
        expected_kind = self.road_types.get(expected)
        actual_kind = self.road_types.get(actual)
        kind = "other_external"
        if expected_kind == "highway.motorway_link" and actual_kind == "highway.motorway":
            kind = "missed_exit"
        if expected_kind == "highway.motorway" and actual_kind == "highway.motorway_link":
            kind = "unwanted_exit"
        return dict(src=src, expected=expected, actual=actual, kind=kind)

    def step(self, action):
        # The inner simulator updates last_external on this very step.
        previous, intact = self.sim.last_external, self.sim.route_intact
        obs, reward, done, cut, info = super().step(action)
        self.last_off_route = self.departure(previous, intact, info)
        off_route = self.last_off_route is not None
        if off_route:
            reward, done, cut = self.off_route_penalty, True, False
            info.update(arrived=False, fork_passed=False, stalled=False, timeout_reward=0.,
                        progress_bonus=0., route_intact=False)
        outcome = None
        if done or cut:
            outcome = ("collision" if info["collided"] else "off_route" if off_route
                       else "arrival" if info["arrived"] else "stall" if info["stalled"]
                       else "timeout" if cut else "other")
        info.update(off_route=off_route, off_route_detail=self.last_off_route, outcome=outcome)
        parts = dict(progress=info["progress_bonus"], arrival=j.ARRIVAL_REWARD*info["arrived"],
                     fork=j.PASS_REWARD*info["fork_passed"], collision=j.COLLISION_REWARD*info["collided"],
                     timeout=info["timeout_reward"], off_route=self.off_route_penalty*off_route,
                     idle_clear=0., pot=0.)
        assert math.isclose(reward, sum(parts.values()), abs_tol=1e-7)
        self.sums.update(parts)
        if outcome and self.outcome_log is not None:
            record = dict(route=self.episode_route, traffic_seed=self.current_seed, outcome=outcome,
                          steps=self.sim.step_count, off_route=self.last_off_route)
            with self.outcome_log.open("a", encoding="utf8") as f:
                print(json.dumps(record), file=f)
        return obs, reward, done, cut, info
