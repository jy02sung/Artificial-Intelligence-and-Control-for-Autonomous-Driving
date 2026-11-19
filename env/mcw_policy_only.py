"""MCW perception with road-level goals; no target-lane oracle or action shield."""
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from gymnasium import spaces
from env.sumo_env import SumoHighwayEnv, traci
from utils.mcw_train import McwEnv


class PolicyOnlyEnv(McwEnv):
    def __init__(self, *args, side_risk_penalty=.3, acceleration_risk_penalty=0., future_goal_coef=0., invalid_command_penalty=.01, **kwargs):
        super().__init__(*args, route_conn=False, prep_reward=False, **kwargs)
        self.side_risk_penalty = float(side_risk_penalty)
        self.acceleration_risk_penalty = float(acceleration_risk_penalty)
        self.future_goal_coef = float(future_goal_coef)
        self.invalid_command_penalty = float(invalid_command_penalty)
        if self.lc_safety_enabled:
            raise ValueError('Policy-only experiments require safety=False')
        self.road_keys = sorted({e for p in self.routes.values() for e in p})
        self.road_ids = {e: i for i, e in enumerate(self.road_keys)}
        self.policy_dim = 47 + 3 * len(self.road_keys) + 6 + 36
        net = ET.parse(Path(self.cfg_path).parent/'network.net.xml')
        # A destination ROAD's entrance centre, averaged across ALL its lanes.
        # This is GPS/map geometry, not a selected lane or a feasible-lane mask.
        self.road_centres = {}
        for edge in net.findall('edge'):
            lanes = edge.findall('lane')
            if lanes and edge.get('id') in self.road_ids:
                self.road_centres[edge.get('id')] = np.mean([
                    [float(c) for c in lane.get('shape').split()[0].split(',')]
                    for lane in lanes], axis=0)
        self.observation_space = spaces.Box(-1., 1., (self.policy_dim,), np.float32)
        self.episode_outcomes = []

    def observation_schema(self):
        """Half-open index ranges, also written to each run's manifest."""
        n = len(self.road_keys)
        s = 47+3*n
        return dict(physical_neighbors=[0,31], removed_lane_hint=[31,34],
            speed_limit=[34,35], reserved_zeros=[35,47],
            current_road=[47,47+n], next_road=[47+n,47+2*n],
            following_road=[47+2*n,s], current_lane=[s,s+6],
            road_physics=[s+6,s+10], side_risk=[s+10,s+14],
            recent_action_and_time=[s+14,s+18], next_road_geometry=[s+18,s+22],
            front_risk=[s+22,s+24], traffic_neighbors=[s+24,s+42])

    def traffic_input_columns(self):
        s = 47+3*len(self.road_keys)
        return list(range(s+10,s+14))+list(range(s+22,s+42))

    def navigation_context(self):
        raise AssertionError('Target-lane navigation must never run in policy-only mode')

    def _lane_change_is_safe(self, direction):
        raise AssertionError('Action safety gate must never run')

    def _lane_connectivity(self, offset):
        # Road geometry only: any outgoing road counts, not the requested next road.
        return SumoHighwayEnv._lane_connectivity(self, offset)

    def _clear_observation_cache(self):
        current=traci.simulation.getTime()
        if getattr(self,'_observation_time',None)!=current:
            self._observation_time=current
            self._neighbor_cache={}
            self._scan_cache={}

    def _scan_lane_neighbors(self, direction):
        self._clear_observation_cache()
        if direction not in self._scan_cache:
            self._scan_cache[direction]=super()._scan_lane_neighbors(direction)
        return self._scan_cache[direction]

    def _neighbor(self, offset, leader):
        self._clear_observation_cache()
        key=(offset,leader)
        if key not in self._neighbor_cache:
            self._neighbor_cache[key]=self._neighbor_uncached(offset,leader)
        return self._neighbor_cache[key]

    def _neighbor_uncached(self, offset, leader):
        # Keep MCW junction/merge perception. Fix omitted close/overlapping cars
        # and the insertion-time neighbour cache for all adjacent lanes.
        candidates = []
        old = super()._neighbor(offset, leader)
        if old is not None:
            candidates.append(old)
        if offset:
            scan = self._scan_lane_neighbors(offset)[0 if leader else 1]
            if scan is not None:
                candidates.append(scan)
            if abs(offset) == 1:
                mode = self._NEIGHBOR_MODES[offset][0 if leader else 1]
                for ident, gap in traci.vehicle.getNeighbors(self.ego_id, mode):
                    if ident and gap <= self.visibility:
                        candidates.append((gap, traci.vehicle.getSpeed(ident)))
        else:
            raw = (traci.vehicle.getLeader if leader else traci.vehicle.getFollower)(self.ego_id, self.visibility)
            if raw and raw[0] and raw[1] <= self.visibility:
                candidates.append((raw[1], traci.vehicle.getSpeed(raw[0])))
        if not candidates:
            return None
        gap, speed = min(candidates, key=lambda x: x[0])
        return (max(0., gap), speed) if gap <= self.visibility else None

    def _get_obs(self):
        out = np.zeros(self.policy_dim, np.float32)
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        # Bypass Clarity/Navigation/IC observations altogether. In particular,
        # slots 31:34 and 37:47 (target offsets/masks) remain exactly zero.
        out[:31] = SumoHighwayEnv._get_obs(self)[:31]
        v = traci.vehicle
        lane_id = v.getLaneID(self.ego_id)
        out[34] = min(1., traci.lane.getMaxSpeed(lane_id) / self.vmax)
        edge, idx = v.getRoadID(self.ego_id), v.getRouteIndex(self.ego_id)
        path = self.routes[self.episode_route]
        n = len(self.road_keys)
        # Current and next two ROAD ids, never target lane ids/masks/offsets.
        for j in range(3):
            if 0 <= idx+j < len(path):
                out[47 + j*n + self.road_ids[path[idx+j]]] = 1.
        start = 47 + 3*n
        lane = v.getLaneIndex(self.ego_id)
        out[start + min(5, lane)] = 1.
        speed = v.getSpeed(self.ego_id)
        out[start+6:start+10] = [min(1., v.getDistance(self.ego_id)/100.),
            min(1., (traci.lane.getLength(lane_id)-v.getLanePosition(self.ego_id))/400.),
            float(edge.startswith(':')), min(1., traci.edge.getLaneNumber(edge)/6.)]
        for i, (offset, front) in enumerate(((1, True), (1, False), (-1, True), (-1, False))):
            car = self._neighbor(offset, front)
            if car:
                gap, other = car
                closing = max(0., speed-other if front else other-speed)
                required = 3. + .5*(speed if front else other) + closing
                out[start+10+i] = max(0., 1.-gap/required)
        out[start+14:start+18] = [self.previous_action[0], self.previous_action[1],
            min(1., self.stationary_seconds/10.),
            min(1., max(0., self.step_count*self.step_length-self.previous_change_time)/10.)]
        position = np.asarray(v.getPosition(self.ego_id))
        angle = np.deg2rad(v.getAngle(self.ego_id))
        forward = np.array([np.sin(angle), np.cos(angle)])
        right = np.array([np.cos(angle), -np.sin(angle)])
        for j in range(2):
            if 0 <= idx+j+1 < len(path):
                delta = self.road_centres[path[idx+j+1]] - position
                out[start+18+2*j:start+20+2*j] = [np.clip(delta@forward/400., -1., 1.),
                    np.clip(delta@right/30., -1., 1.)]
        lead = self._neighbor(0, True)
        if lead:
            gap, other = lead
            closing = max(0., speed-other)
            out[start+22] = max(0., 1.-gap/(3.+.7*speed+1.5*closing))
            out[start+23] = max(0., 1.-gap/(2.5*closing)) if closing>0 else 0.
        # Presence, proximity and relative speed for each immediately adjacent
        # front/rear car. All three are zero if absent, so traffic adaptation
        # cannot rewrite the no-traffic road policy through these columns.
        for i, (offset, front) in enumerate(((1,True),(1,False),(0,True),
                (0,False),(-1,True),(-1,False))):
            car = self._neighbor(offset, front)
            if car:
                gap, other = car
                at = start+24+3*i
                out[at:at+3] = [1., max(0.,1.-gap/self.visibility),
                    np.clip((other-speed)/self.vmax,-1.,1.)]
        self.last_observation = out
        return out

    def reset(self, *args, **kwargs):
        self._observation_time=None
        self.previous_change_time = -100.
        self.previous_change = 0
        self.previous_action = [0., 0.]
        self.stationary_seconds = 0.
        self.best_route_index = 0
        result = super().reset(*args, **kwargs)
        assert not self.lc_safety_enabled
        assert traci.vehicle.getLaneChangeMode(self.ego_id) == 0
        assert traci.vehicle.getSpeedMode(self.ego_id) == 0
        return result

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).copy()
        v = traci.vehicle
        edge = v.getRoadID(self.ego_id)
        old_lane = v.getLaneIndex(self.ego_id)
        idx = v.getRouteIndex(self.ego_id)
        path = self.routes[self.episode_route]
        goal = self.road_centres[path[idx+1]] if 0 <= idx < len(path)-1 else None
        goal_distance = min(200., float(np.linalg.norm(goal-np.asarray(v.getPosition(self.ego_id))))) if goal is not None else 0.
        future_goal = self.road_centres[path[idx+2]] if 0 <= idx < len(path)-2 else None
        future_distance = min(300., float(np.linalg.norm(future_goal-np.asarray(v.getPosition(self.ego_id))))) if future_goal is not None else 0.
        before = self.last_observation.copy()
        self.previous_action = np.clip(raw, -1., 1.).tolist()
        self.stationary_seconds = self.stationary_seconds+self.step_length if v.getSpeed(self.ego_id)<.5 else 0.
        # This executes the PPO action directly through the base physical actuator.
        obs, _, done, cut, info = SumoHighwayEnv.step(self, raw)
        assert np.isclose(info['accel_raw'], np.clip(raw[0], -1., 1.))
        assert np.isclose(info['lane_change_raw'], np.clip(raw[1], -1., 1.))
        arrived = info['arrived'] and not info['collided'] and edge == self.routes[self.episode_route][-1]
        info['arrived'] = bool(arrived)
        info['policy_action'] = raw.tolist()
        info['safety_overrides'] = 0
        info['lane_guidance'] = False
        components = dict(progress=0., road_goal=0., future_goal=0., safety=0., lane=0., terminal=0., time=-.002, stall=0.)
        if info['collided']:
            components['terminal'] = -30.
        elif arrived:
            components['terminal'] = 20.
        elif done:
            components['terminal'] = -20.
        elif cut:
            components['terminal'] = -20.
        else:
            speed = v.getSpeed(self.ego_id)
            components['progress'] = .035 * speed/self.vmax
            new_idx = v.getRouteIndex(self.ego_id)
            if goal is not None and idx == new_idx:
                after_distance = min(200., float(np.linalg.norm(goal-np.asarray(v.getPosition(self.ego_id)))))
                components['road_goal'] = .015*(goal_distance-after_distance)
            if future_goal is not None and idx == new_idx:
                after_future = min(300., float(np.linalg.norm(future_goal-np.asarray(v.getPosition(self.ego_id)))))
                components['future_goal'] = self.future_goal_coef*(future_distance-after_future)
            if new_idx > self.best_route_index:
                components['progress'] += .6*(new_idx-self.best_route_index)
                self.best_route_index = new_idx
            lead = self._neighbor(0, True)
            if speed < 3. and (lead is None or lead[0] > 25.):
                components['stall'] -= .08 * (1.-speed/3.)
            if lead:
                gap, other = lead
                closing = max(0., speed-other)
                required = 3. + .7*speed + 1.5*closing
                components['safety'] -= .08 * max(0., 1.-gap/required)
            start = 47 + 3*len(self.road_keys)
            command = info['lane_change']
            if command:
                a = start+10 if command > 0 else start+12
                components['safety'] -= self.side_risk_penalty*max(before[a:a+2])
                if not info['lane_change_applied']:
                    components['lane'] -= self.invalid_command_penalty
            actual = v.getLaneIndex(self.ego_id)-old_lane if v.getRoadID(self.ego_id)==edge else 0
            if actual:
                components['lane'] -= .01
                if actual == -self.previous_change and self.step_count*self.step_length-self.previous_change_time < 2.:
                    components['lane'] -= .2
                self.previous_change, self.previous_change_time = actual, self.step_count*self.step_length
            # A training cost for accelerating while closing on a nearby leader.
            # It never edits the requested acceleration or lane command.
            components['safety'] -= self.acceleration_risk_penalty*before[start+22]*max(0., info['accel_raw'])
        if done or cut:
            outcome = 'collision' if info['collided'] else 'arrival' if arrived else 'timeout' if cut else 'other'
            info['outcome'] = outcome
            self.episode_outcomes.append(outcome)
        info['reward_components'] = components
        return obs, float(sum(components.values())), done, cut, info
