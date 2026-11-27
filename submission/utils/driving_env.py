import argparse
import numpy as np
from gymnasium import spaces
from env.sumo_env import SumoHighwayEnv
from utils.scenario import directions
from env.sumo_env import traci
from env.jys3_dagger_sensors import DaggerSensors
from utils import scenario as m
PASS_REWARD = 0.25
ARRIVAL_REWARD = 5.0
COLLISION_REWARD = -15.0
TIMEOUT_REWARD = -5.0
STALL_SECONDS = 30.0
MAX_LANES = 5
OBS_DIM = 40 + 2 * MAX_LANES


class SparseRouteEnv(DaggerSensors):
    def split(self,path,idx,remaining):
        for j in range(idx,len(path)-1):
            src,dst = path[j:j+2]
            if remaining>400:
                break
            if (src,dst) in self.signs:
                return j,remaining
            remaining += self.navigation.link_lengths.get((src,dst),0)+self.navigation.lengths[dst]
        return None,None

    def _get_obs(self):
        base = super()._get_obs()
        vehicle = traci.vehicle
        self.snapshot = None
        if self.ego_id not in vehicle.getIDList():
            return np.zeros(OBS_DIM, dtype=np.float32)
        edge = vehicle.getRoadID(self.ego_id)
        lane = vehicle.getLaneIndex(self.ego_id)
        if getattr(self, "physical_forks", False):
            # SUMO may replace its executable route. Navigation always uses the mission.
            path = self.routes[self.episode_route]
            index = self.last_external[0] if self.last_external is not None else 0
            self.crossed_short_link = False
            if index + 1 < len(path) and edge == path[index + 1]:
                index += 1
            elif not edge.startswith(":") and edge != path[index] and edge in path[index+2:]:
                # A short edge may be crossed between two 0.2 s observations.
                # Verify every intermediate edge against SUMO's actual executed route.
                end = path.index(edge, index+2)
                runtime_end = vehicle.getRouteIndex(self.ego_id) + 1
                runtime_start = runtime_end - (end-index+1)
                runtime = vehicle.getRoute(self.ego_id)
                if runtime_start >= 0 and tuple(runtime[runtime_start:runtime_end]) == tuple(path[index:end+1]):
                    self.crossed_short_link = True
                    index = end
        else:
            path = vehicle.getRoute(self.ego_id)
            index = vehicle.getRouteIndex(self.ego_id)
        remaining = traci.lane.getLength(vehicle.getLaneID(self.ego_id)) - vehicle.getLanePosition(self.ego_id)
        fork, distance = self.split(path, index, remaining) if not edge.startswith(":") else (None, None)
        # Use the same visibility mask for the two extra overlap cues as for the base observations.
        self._occ_sensor_mask = True
        try:
            overlap = [float(any(x and x[0] <= 0 for x in self._scan_lane_neighbors(direction)))
                       for direction in (-1, 1)]
        finally:
            self._occ_sensor_mask = False
        self.snapshot = (edge, lane, path, index, fork, distance, None, vehicle.getSpeed(self.ego_id), remaining)
        lane_count = traci.edge.getLaneNumber(edge)
        limit = traci.lane.getMaxSpeed(vehicle.getLaneID(self.ego_id))
        extra = [min(1., limit / self.vmax), lane / MAX_LANES, lane_count / MAX_LANES,
                 self.signs[path[fork], path[fork+1]] if fork is not None else 0.,
                 distance / 400. if distance is not None else -1., *overlap,
                 float(edge in self.solid or edge.startswith(":")), min(1.,remaining / 400.)]
        flags = self.road_flags.get(edge, self.no_links)
        return np.concatenate((base, np.asarray(extra, dtype=np.float32), flags)).clip(-1, 1)

    def reset(self, *args, **kwargs):
        self.last_external = None
        obs, info = super().reset(*args, **kwargs)
        self.last_external = self.snapshot[3], self.snapshot[0]
        self.route_intact = self.last_external == (0, self.routes[self.episode_route][0])
        self.paid_forks = set()
        self.stopped_steps = 0
        return obs, info

    def step(self, action):
        # Retain jys2 vehicle actuation. Skip all inherited reward-shaping wrappers.
        obs, _, done, cut, info = SumoHighwayEnv.step(self, action)
        previous_index, previous_edge = self.last_external
        path = self.routes[self.episode_route]
        if self.ego_id in traci.simulation.getStartingTeleportIDList():
            self.route_intact = False
        arrived = bool(self.route_intact and info["arrived"] and not info["collided"]
                       and previous_index == len(path) - 1 and previous_edge == path[-1])
        info["arrived"] = arrived
        passed = False
        stranded = False
        if self.snapshot and not done:
            edge, lane, _, index, _, _, _, speed, remaining = self.snapshot
            if not edge.startswith(":"):
                follows_route = (0 <= index < len(path) and edge == path[index]
                                 and (index in (previous_index, previous_index + 1)
                                      or getattr(self, "crossed_short_link", False)))
                self.route_intact = self.route_intact and follows_route
                passed = (self.route_intact and index == previous_index + 1 and edge == path[index]
                          and previous_edge == path[previous_index]
                          and (previous_edge, edge) in self.signs
                          and previous_index not in self.paid_forks)
                if passed:
                    self.paid_forks.add(previous_index)
                if self.route_intact and getattr(self, "crossed_short_link", False):
                    for k in range(previous_index, index):
                        if (path[k], path[k+1]) in self.signs and k not in self.paid_forks:
                            self.paid_forks.add(k)
                            info.setdefault("passed_edges", []).append(path[k+1])
                            passed = True
                self.last_external = index, edge
                if follows_route and index + 1 < len(path) and remaining <= 5.0:
                    limit = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(self.ego_id))
                    stranded = (speed <= min(0.5, 0.05 * limit)
                                and lane not in self.links[edge][path[index + 1]]
                                and (info["gap"] is None or info["gap"] > 25.0))
        self.stopped_steps = self.stopped_steps + 1 if stranded else 0
        stalled = self.stopped_steps * self.step_length >= STALL_SECONDS
        reward = COLLISION_REWARD if info["collided"] else ARRIVAL_REWARD if arrived else PASS_REWARD if passed else 0.0
        timed_out = (cut or stalled) and not done
        if timed_out:
            reward += TIMEOUT_REWARD
        info.update(fork_passed=passed, stalled=stalled, route_intact=self.route_intact,
                    timeout_reward=TIMEOUT_REWARD if timed_out else 0.)
        return obs, reward, done, cut or stalled, info


def environment(stage, seed, folder):
    cfg = m.prepare_scenario(stage, folder)
    net = cfg.parent / "network.net.xml"
    road, nav = m.load_network_metadata(net, cfg.parent / "highway.rou.xml")
    args = argparse.Namespace(stage=stage, safety=False, smoke=False, min_front_gap=3.5,
                              min_rear_gap=3.5, rear_tau=.35, approach_horizon=1.2,
                              collision_penalty=15., arrival_bonus=5., lc_penalty=0.)
    action, reward, sim, ego, traffic = m.build_configs(args)
    sim["max_steps"] = 1000
    reward.update(speed_weight=0., close_gap_penalty=0., blocked_penalty=0.)
    ego["depart_lane"] = "0"
    env = SparseRouteEnv(str(cfg), road, ego, sim, dict(m.OBSERVATION, visibility=120.),
                         action, reward, traffic, (), seed, navigation=nav, clarity="baseline",
                         obs_noise=True, occlusion=True)
    env.signs = directions(net, road["connections"])
    env.no_links = np.zeros(2 * MAX_LANES, dtype=np.float32)
    env.road_flags = {}
    if max(env.counts.values()) > MAX_LANES:
        raise ValueError("Road exceeds the lane schema")
    for edge, exits in env.links.items():
        if len(exits) > 2:
            raise ValueError("Road exceeds the two-branch schema")
        flags = env.no_links.copy().reshape(MAX_LANES, 2)
        for dest, lanes in exits.items():
            sign = env.signs.get((edge, dest))
            for lane_id in lanes:
                if sign is None:
                    flags[lane_id, :] = 1.
                else:
                    flags[lane_id, int(sign > 0)] = 1.
        env.road_flags[edge] = flags.ravel()
    env.observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
    assert not env.lc_safety_enabled
    return env
