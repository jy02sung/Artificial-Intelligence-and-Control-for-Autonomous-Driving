"""jys3: jys2 driving controls, route-aware lane reach, sparse event rewards.

Training is opt-in: python -m utils.jys3 --train --steps N --seed S
"""
import argparse
import hashlib
import numpy as np
from gymnasium import spaces
from pathlib import Path

from env.sumo_env import SumoHighwayEnv
from utils.experiment_a import RoadEnv, directions, traci
from env.jys3_dagger_sensors import DaggerSensors
from utils.experiment_b import HP, m, torch
from utils.experiment_c_train import ScheduledEpisodes, atomic_torch

PASS_REWARD = 0.25
ARRIVAL_REWARD = 5.0
COLLISION_REWARD = -15.0
TIMEOUT_REWARD = -5.0
STALL_SECONDS = 30.0


CHECKPOINT = m.ROOT / "models" / "jys3" / "dagger2.pt"
CHECKSUM = "fa5d26c9d42fad47217b9e10f560771fc01697435c4546c22a53e1e277f33514"
OBS_DIM = 52


def warm_start(seed):
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("Unexpected DAgger checkpoint")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # Retain the original 47 inputs and make the five new jys2 cues neutral at step zero.
    state["trunk.0.weight"] = torch.nn.functional.pad(state["trunk.0.weight"], (0, 5))
    agent = m.PPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=seed, **HP)
    agent.policy.load_state_dict(state, strict=True)
    return agent


class SparseRouteEnv(DaggerSensors):
    split = RoadEnv.split

    def _get_obs(self):
        base = super()._get_obs()
        vehicle = traci.vehicle
        self.snapshot = None
        if self.ego_id not in vehicle.getIDList():
            return np.zeros(OBS_DIM, dtype=np.float32)
        edge = vehicle.getRoadID(self.ego_id)
        path = vehicle.getRoute(self.ego_id)
        lane = vehicle.getLaneIndex(self.ego_id)
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
        extra = [lane / max(1, self.counts.get(edge, 1)-1),
                 self.signs[path[fork], path[fork+1]] if fork is not None else 0.,
                 distance / 400. if distance is not None else -1., *overlap]
        return np.concatenate((base, np.asarray(extra, dtype=np.float32))).clip(-1, 1)

    def reset(self, *args, **kwargs):
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
                                 and index in (previous_index, previous_index + 1))
                self.route_intact = self.route_intact and follows_route
                passed = (self.route_intact and index == previous_index + 1 and edge == path[index]
                          and previous_edge == path[previous_index]
                          and (previous_edge, edge) in self.signs
                          and previous_index not in self.paid_forks)
                if passed:
                    self.paid_forks.add(previous_index)
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
                         action, reward, traffic, (), seed, navigation=nav, clarity="lookahead",
                         route_conn=True, agenda2=True, obs_noise=True, occlusion=True)
    env.signs = directions(net, road["connections"])
    env.observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
    assert not env.lc_safety_enabled
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    if not args.steps or args.steps < 2:
        parser.error("--train requires --steps >= 2")
    torch.set_num_threads(1)
    folder = Path(m.ROOT) / "results" / "jys3" / f"seed_{args.seed}_{args.steps}"
    folder.mkdir(parents=True, exist_ok=False)
    agent = warm_start(args.seed)
    env = environment("high_real_ratio", args.seed, folder / "map")
    try:
        episodes = ScheduledEpisodes(env, args.seed, folder / "episodes.jsonl")
        obs, _ = episodes.reset()
        stats = [0., 0]
        atomic_torch(agent.policy.state_dict(), folder / "policy_0.pt")
        while agent.num_timesteps < args.steps:
            steps = min(2048, args.steps - agent.num_timesteps)
            if args.steps - agent.num_timesteps == 2049:
                steps = 2047
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats,
                                                  steps=steps, timeout_is_failure=True)
            advantages, returns = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            agent.update(advantages, returns)
        atomic_torch(agent.policy.state_dict(), folder / f"policy_{args.steps}.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
