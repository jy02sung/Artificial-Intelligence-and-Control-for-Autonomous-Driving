"""One learned driver on the release road and traffic. Both actions belong to the policy."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium import spaces

from env.mdp_config import ACTION, OBSERVATION, REWARD, SIMULATION
from env.gap_reward import gap_reward, opportunity
from env.navigation_clarity import ClarityEnv
from env.navigation_observation import RouteNavigation
from env.road_config import EGO, ROAD, TRAFFIC
import env.ic_experiment as ic
from env.sumo_env import traci


SCENARIO = Path(__file__).resolve().parents[1] / "scenarios/hv_high15000_entryprep"
ROUTES = tuple(f"route_{i:02}" for i in range(1, 17))
STEP = 0.2
WARMUP = 900
DRIVE_LIMIT = 10800
RANGE = 120.
# Original 47 inputs followed by ten policy-visible, normalized quantities.
OBS_SIZE = 57
GAP_CONDITIONS = ("gap", "gap_context", "gap_adapter", "gap_lane")


def input_hashes():
    expected = json.loads((SCENARIO / "manifest.json").read_text())["files"]
    actual = {name: hashlib.sha256((SCENARIO / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError("Reference scenario differs from its released manifest")
    return actual


def road_inputs():
    input_hashes()
    net = ET.parse(SCENARIO / "network.net.xml")
    edges = {e.get("id"): e for e in net.findall("edge") if not e.get("function")}
    lanes = {key: len(edge.findall("lane")) for key, edge in edges.items()}
    routes = {r.get("id"): r.get("edges").split()
              for r in ET.parse(SCENARIO / "highway.rou.xml").findall("route") if r.get("id") in ROUTES}
    if set(routes) != set(ROUTES):
        raise ValueError("Incomplete release route set")
    connections = [(c.get("from"), c.get("to"), int(c.get("fromLane")), int(c.get("toLane")))
                   for c in net.findall("connection") if c.get("from") in edges and c.get("to") in edges]
    lengths = {k: max(float(l.get("length")) for l in e.findall("lane")) for k, e in edges.items()}
    internal = {l.get("id"): float(l.get("length")) for e in net.findall("edge")
                if e.get("function") == "internal" for l in e.findall("lane")}
    links = {}
    for c in net.findall("connection"):
        pair = (c.get("from"), c.get("to"))
        links[pair] = max(links.get(pair, 0.), internal.get(c.get("via"), 0.))
    road = dict(ROAD, routes=routes, connections=connections,
                edges=[(k, e.get("from"), e.get("to"), lanes[k]) for k, e in edges.items()],
                solid_lane_edges=[k for k, e in edges.items() if any(l.get("changeLeft") or l.get("changeRight")
                                                                     for l in e.findall("lane"))])
    nav = RouteNavigation(lanes, connections, lengths, horizon=300., link_lengths=links)
    ramps = {k for k, e in edges.items() if e.get("type", "").endswith("_link")}
    return road, nav, ramps


def occupied_crossing(ego_distance, ego_speed, other_distance, other_exit, other_speed, length):
    """Estimated overlap at a mapped conflict point from current positions and speeds."""
    if not 0. < ego_distance <= 80. or abs(other_distance) > 80. or other_exit + length < 0.:
        return False
    if other_speed < 1. and other_distance > 6.:
        return False
    ego_eta = ego_distance / max(4., ego_speed)
    other_entry = max(0., other_distance) / max(1., other_speed)
    other_clear = max(0., other_exit + length) / max(1., other_speed)
    return other_entry - 1.25 <= ego_eta <= other_clear + 1.25


class EntryEnv(ClarityEnv):
    def __init__(self, *, condition, seed, training=False, gui=False):
        if condition not in ("baseline", "risk", "trip", "merge", *GAP_CONDITIONS):
            raise ValueError(condition)
        road, nav, ramps = road_inputs()
        # ICExperimentEnv uses these map-specific sets when it calculates IC features.
        ic.RAMPS, ic.ZIP = ramps, ()
        self.condition = condition
        self.training = training
        self.snapshot = None
        simulation = dict(SIMULATION, step_length=STEP, warmup_steps=int(WARMUP / STEP),
                          max_steps=int((240 if training else DRIVE_LIMIT) / STEP))
        reward = dict(REWARD, collision_penalty=15., arrival_bonus=8.) if condition in GAP_CONDITIONS else REWARD
        super().__init__(str(SCENARIO / "run.sumocfg"), road, dict(EGO, depart_lane="best", depart_speed="max"),
                         simulation, OBSERVATION, ACTION, reward, dict(TRAFFIC, prefill={"enabled": False}),
                         (), seed, navigation=nav, clarity="baseline")
        self.gui = gui
        self.gui_delay = 0.
        self.gui_autostart = True
        self.insert_wait_steps = int(DRIVE_LIMIT / STEP)
        self.observation_space = spaces.Box(-1., 1., shape=(OBS_SIZE,), dtype=np.float32)

    def _warmup_traffic(self):
        if self.snapshot is None:
            traci.simulationStep(WARMUP)
        else:
            traci.simulation.loadState(str(self.snapshot))
        if abs(traci.simulation.getTime() - WARMUP) > 1e-6 or traci.vehicle.getIDCount() == 0:
            raise RuntimeError("Invalid traffic warmup")

    def _neighbor(self, offset, leader):
        if self.condition == "baseline":
            return super()._neighbor(offset, leader)
        # A negative SUMO gap with a real ID denotes a near/overlapping vehicle.
        # Neighbours on an adjacent lane can be missing in the lane-change cache
        # immediately after insertion. Use the same-edge lane scan as a second source.
        query = traci.vehicle
        if offset == 0:
            raw = query.getLeader(self.ego_id, self.visibility) if leader else query.getFollower(self.ego_id, self.visibility)
            if not raw or not raw[0] or raw[1] > self.visibility:
                return None
            return max(0., raw[1]), query.getSpeed(raw[0])
        scanned = self._scan_lane_neighbors(offset)[0 if leader else 1]
        candidates = []
        if scanned is not None:
            candidates.append(scanned)
        if abs(offset) == 1:
            mode = self._NEIGHBOR_MODES[offset][0 if leader else 1]
            for ident, gap in query.getNeighbors(self.ego_id, mode):
                if ident and gap <= self.visibility:
                    candidates.append((gap, query.getSpeed(ident)))
        if not candidates:
            return None
        gap, speed = min(candidates, key=lambda p: p[0])
        return (max(0., gap), speed) if gap <= self.visibility else None

    def _get_obs(self):
        base = super()._get_obs()
        extra = np.zeros(OBS_SIZE - len(base), dtype=np.float32)
        if self.condition != "baseline" and self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle
            speed = v.getSpeed(self.ego_id)
            lead = v.getLeader(self.ego_id, RANGE)
            if lead and lead[0] and lead[1] <= RANGE:
                extra[:3] = (1., max(0., lead[1]) / RANGE,
                             np.clip((v.getSpeed(lead[0]) - speed) / self.vmax, -1., 1.))
            # Signed lane distance from SUMO's route guidance. No lane commands are generated.
            road = v.getRoadID(self.ego_id)
            if not road.startswith(":"):
                idx = v.getLaneIndex(self.ego_id)
                best = v.getBestLanes(self.ego_id)[idx]
                extra[3] = np.clip(best[3] / 4., -1., 1.)
                extra[4] = min(1., v.getLanePosition(self.ego_id) / 100.)
                extra[5] = min(1., (traci.lane.getLength(v.getLaneID(self.ego_id)) - v.getLanePosition(self.ego_id)) / 300.)
            if self.condition in ("trip", "merge", *GAP_CONDITIONS):
                # Only the first metres of the trip are an insertion. Entering an
                # ordinary new edge must not look like spawning a second time.
                extra[4] = np.clip(v.getDistance(self.ego_id) / 100., 0., 1.)
            if self.condition in ("merge", *GAP_CONDITIONS):
                # A current crossing vehicle can occupy the road before the physical
                # leader does. Encode its conflict point as a stationary obstacle.
                for foe in v.getJunctionFoes(self.ego_id, 80.):
                    if hasattr(foe, "foeId"):
                        ident, mine, theirs, leave = foe.foeId, foe.egoDist, foe.foeDist, foe.foeExitDist
                    else:
                        ident, mine, theirs, _, leave = foe[:5]
                    if (not extra[0] or mine < extra[1] * RANGE) and occupied_crossing(
                        mine, speed, theirs, leave, v.getSpeed(ident), v.getLength(ident)
                    ):
                        extra[:3] = (1., mine / RANGE, -speed / self.vmax)
            for i, (offset, front) in enumerate(((1, True), (1, False), (-1, True), (-1, False))):
                car = self._neighbor(offset, front)
                if car is not None:
                    gap, other = car
                    closing = max(0., (speed-other) if front else (other-speed))
                    required = 2.5 + .7 * (speed if front else other) + closing
                    extra[6+i] = max(0., 1. - gap / required)
        obs = np.concatenate((base, extra))
        if self.condition in ("gap_context", "gap_adapter", "gap_lane"):
            gap = opportunity(obs, self.vmax, self.visibility)
            if gap is not None:
                # Four formerly zero clarity inputs describe the currently observable gap.
                # They are policy inputs, including the speed difference needed to align.
                speed_difference = (gap.approach_speed - obs[0] * self.vmax) / self.max_decel
                obs[37:41] = (gap.direction, 1. if gap.available else -1.,
                              np.clip(speed_difference, -1., 1.), float(gap.immediate))
        if self.condition in GAP_CONDITIONS:
            self.last_observation = obs
        return obs

    def step(self, action):
        if self.condition in GAP_CONDITIONS:
            before = self.last_observation
            previous_edge = traci.vehicle.getRoadID(self.ego_id)
            route_index = traci.vehicle.getRouteIndex(self.ego_id)
            if self.step_count == 0:
                self.rewarded_gap_moves = set()
        obs, reward, done, cut, info = super().step(action)
        if self.condition != "baseline" and not done:
            # Train the policy to brake early and to progress along the requested route.
            # These are learning signals only; they cannot replace either commanded action.
            speed = obs[0] * self.vmax
            if obs[47] > 0:
                gap = obs[48] * RANGE
                closing = max(0., -obs[49] * self.vmax)
                desired_gap = 3. + speed + 1.5 * closing
                reward -= .45 * max(0., 1. - gap / desired_gap) * STEP
            if self.condition not in GAP_CONDITIONS and info["lane_change"]:
                side = 53 if info["lane_change"] > 0 else 55
                reward -= .12 * max(obs[side:side+2]) * STEP
            reward -= .08 * min(1., abs(obs[50]) * 4.) * STEP
        if self.condition in GAP_CONDITIONS and not done:
            actual = 0
            if previous_edge == traci.vehicle.getRoadID(self.ego_id):
                actual = info["lane_after"] - info["lane_before"]
            gap = opportunity(before, self.vmax, self.visibility)
            shaping = gap_reward(gap, before[0] * self.vmax, info["lane_change"], actual, STEP)
            if shaping > 0:
                # A reverse move and a return to the same lane must not earn a second bonus.
                move = (route_index, info["lane_after"])
                if move in self.rewarded_gap_moves:
                    shaping = 0.
                self.rewarded_gap_moves.add(move)
            reward += shaping
            if cut:
                reward -= 6.
        return obs, float(reward), done, cut, info
