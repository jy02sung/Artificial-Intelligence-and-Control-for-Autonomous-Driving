"""Sensor path of the 47-input DAgger teacher-data environment (mcw2 ac2f9bc)."""
import math
import zlib
import numpy as np
from env.navigation_clarity import ClarityEnv
from env.sumo_env import SumoHighwayEnv, traci

def _segment_intersects_obb(
    sx: float, sy: float, tx: float, ty: float,
    cx: float, cy: float, ux: float, uy: float,
    rx: float, ry: float, half_l: float, half_w: float,
) -> bool:
    """Return True if segment S->T enters an oriented box [+-half_l, +-half_w] before reaching T."""
    dsx, dsy = sx - cx, sy - cy
    dtx, dty = tx - cx, ty - cy
    s_u = dsx * ux + dsy * uy
    s_r = dsx * rx + dsy * ry
    du = (dtx * ux + dty * uy) - s_u
    dr = (dtx * rx + dty * ry) - s_r
    tmin, tmax = 0.0, 0.999
    for p0, dp, bound in ((s_u, du, half_l), (s_r, dr, half_w)):
        if abs(dp) < 1e-9:
            if p0 < -bound or p0 > bound:
                return False
        else:
            t0 = (-bound - p0) / dp
            t1 = (bound - p0) / dp
            if t0 > t1:
                t0, t1 = t1, t0
            tmin = max(tmin, t0)
            tmax = min(tmax, t1)
            if tmin > tmax:
                return False
    return tmin <= tmax

class DaggerSensors(ClarityEnv):
    """Local traffic sensor with occlusion and distance-dependent noise."""

    # Zipper merge foe lanes: ego_lane -> list of (foe_incoming_lane, foe_via_lane)
    MERGE_FOES = {
        "169728849_0": [("544295537_0", ":1808903166_1_0")],
        "544295537_0": [("169728849_0", ":1808903166_0_0")],
        "169929503#0_0": [("170059992_0", ":1810880137_0_0")],
        "170059992_0": [("169929503#0_0", ":1810880137_1_0")],
        "206426298#1_0": [("544292616_1", ":1000493465_0_1")],
        "544292616_1": [("206426298#1_0", ":1000493465_2_0")],
    }
    INTERNAL_MERGE_FOES = {
        ":1808903166_0_0": (":1808903166_1_0", "169929488#0_0"),
        ":1808903166_1_0": (":1808903166_0_0", "169929488#0_0"),
        ":1810880137_0_0": (":1810880137_1_0", "169929503#1_0"),
        ":1810880137_1_0": (":1810880137_0_0", "169929503#1_0"),
        ":1000493465_0_1": (":1000493465_2_0", "844484877_1"),
        ":1000493465_2_0": (":1000493465_0_1", "844484877_1"),
    }

    def __init__(self, *args,
                 obs_noise: bool = False, noise_gap_base: float = 0.2,
                 noise_gap_slope: float = 3.6, noise_vel_base: float = 0.15,
                 noise_vel_slope: float = 1.35,
                 occlusion: bool = False, occ_truck_rate: float = 1.0,
                 occ_bus_rate: float = 1.0, occ_delivery_rate: float = 0.5,
                 gui: bool = False, **kwargs):
        self.obs_noise = bool(obs_noise)
        self.noise_gap_base = float(noise_gap_base)
        self.noise_gap_slope = float(noise_gap_slope)
        self.noise_vel_base = float(noise_vel_base)
        self.noise_vel_slope = float(noise_vel_slope)
        self.occlusion = bool(occlusion)
        self.occ_truck_rate = float(occ_truck_rate)
        self.occ_bus_rate = float(occ_bus_rate)
        self.occ_delivery_rate = float(occ_delivery_rate)
        self._noise_rng = np.random.default_rng(11)
        self._occ_sensor_mask = False
        self._occ_subscribed = False
        self._occ_sensor = (0.0, 0.0)
        self._occ_boxes = []
        self._ctx_vehicles = {}
        self._occluded_cache = {}
        self._colored_delivery_ids = set()
        super().__init__(*args, **kwargs)
        self.gui = bool(gui)

    def reset(self, *args, **kwargs):
        self._occ_sensor_mask = False
        self._occ_subscribed = False
        self._occ_boxes = []
        self._ctx_vehicles = {}
        self._occluded_cache = {}
        self._colored_delivery_ids = set()
        seed = kwargs.get("seed") if kwargs else None
        if seed is None and len(args) >= 1:
            seed = args[0]
        if seed is not None:
            self._noise_rng = np.random.default_rng(int(seed) + 777)
        return super().reset(*args, **kwargs)

    def _occluder_rate(self, vclass: str) -> float:
        if vclass == "truck":
            return self.occ_truck_rate
        if vclass == "bus":
            return self.occ_bus_rate
        if vclass == "delivery":
            return self.occ_delivery_rate
        return 0.0

    def _is_opaque_occluder(self, vid: str, vclass: str) -> bool:
        rate = self._occluder_rate(vclass)
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return (zlib.crc32(vid.encode("utf-8")) % 1000) < round(rate * 1000.0)

    def _update_occlusion_state(self):
        self._occ_boxes = []
        self._ctx_vehicles = {}
        self._occluded_cache = {}
        try:
            tc = traci.constants
            if not self._occ_subscribed:
                traci.vehicle.subscribeContext(
                    self.ego_id,
                    tc.CMD_GET_VEHICLE_VARIABLE,
                    self.visibility + 25.0,
                    [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_LENGTH, tc.VAR_WIDTH, tc.VAR_VEHICLECLASS],
                )
                self._occ_subscribed = True
            ctx = traci.vehicle.getContextSubscriptionResults(self.ego_id)
            if not ctx:
                return
            self._ctx_vehicles = ctx
            ex, ey = traci.vehicle.getPosition(self.ego_id)
            erad = math.radians(float(traci.vehicle.getAngle(self.ego_id)))
            # Driver windshield / front-roof sensor origin (~1.2m back from front bumper)
            sx = float(ex) - 1.2 * math.sin(erad)
            sy = float(ey) - 1.2 * math.cos(erad)
            self._occ_sensor = (sx, sy)
            for vid, vals in ctx.items():
                if vid == self.ego_id:
                    continue
                vclass = vals.get(tc.VAR_VEHICLECLASS, "")
                opaque = self._is_opaque_occluder(vid, vclass)
                if getattr(self, "gui", False) and vclass == "delivery" and vid not in self._colored_delivery_ids:
                    self._colored_delivery_ids.add(vid)
                    try:
                        # Box van (occluder): deep teal; open cargo (pass-through): pale sky
                        traci.vehicle.setColor(vid, (15, 130, 175, 255) if opaque else (130, 225, 245, 255))
                    except traci.TraCIException:
                        pass
                if not opaque:
                    continue
                fx, fy = vals[tc.VAR_POSITION]
                ang = math.radians(float(vals[tc.VAR_ANGLE]))
                vlen = float(vals[tc.VAR_LENGTH])
                vwid = float(vals[tc.VAR_WIDTH])
                ux, uy = math.sin(ang), math.cos(ang)
                rx, ry = math.cos(ang), -math.sin(ang)
                cx = float(fx) - 0.5 * vlen * ux
                cy = float(fy) - 0.5 * vlen * uy
                if math.hypot(cx - sx, cy - sy) > self.visibility + 12.0:
                    continue
                self._occ_boxes.append((vid, cx, cy, ux, uy, rx, ry, 0.5 * vlen, 0.5 * vwid))
        except traci.TraCIException:
            self._occ_boxes = []

    def _is_occluded(self, vid: str) -> bool:
        if not self._occ_sensor_mask or not self._occ_boxes or not vid or vid == self.ego_id:
            return False
        cached = self._occluded_cache.get(vid)
        if cached is not None:
            return cached
        try:
            tc = traci.constants
            vals = self._ctx_vehicles.get(vid)
            if vals is not None:
                fx, fy = vals[tc.VAR_POSITION]
                ang = math.radians(float(vals[tc.VAR_ANGLE]))
                vlen = float(vals[tc.VAR_LENGTH])
            else:
                fx, fy = traci.vehicle.getPosition(vid)
                ang = math.radians(float(traci.vehicle.getAngle(vid)))
                vlen = float(traci.vehicle.getLength(vid))
            tcx = float(fx) - 0.5 * vlen * math.sin(ang)
            tcy = float(fy) - 0.5 * vlen * math.cos(ang)
            sx, sy = self._occ_sensor
            occluded = any(
                vid != occ_id and _segment_intersects_obb(sx, sy, tcx, tcy, cx, cy, ux, uy, rx, ry, hl, hw)
                for occ_id, cx, cy, ux, uy, rx, ry, hl, hw in self._occ_boxes
            )
        except traci.TraCIException:
            occluded = False
        self._occluded_cache[vid] = occluded
        return occluded

    def _get_obs(self):
        if getattr(self, "occlusion", False) and self.ego_id in traci.vehicle.getIDList():
            self._update_occlusion_state()
            self._occ_sensor_mask = True
        try:
            obs = SumoHighwayEnv._get_obs(self)
        finally:
            self._occ_sensor_mask = False
        out = obs.copy()
        if self.obs_noise and self.ego_id in traci.vehicle.getIDList():
            for k in range(len(self.obs_offsets)):
                lg_idx, lv_idx = 1 + 4 * k, 2 + 4 * k
                if not (out[lg_idx] >= 1.0 - 1e-6 and out[lv_idx] >= 1.0 - 1e-6):
                    frac = float(np.clip(out[lg_idx], 0.0, 1.0))
                    sg = self.noise_gap_base + self.noise_gap_slope * frac
                    sv = self.noise_vel_base + self.noise_vel_slope * frac
                    out[lg_idx] = float(np.clip(out[lg_idx] + self._noise_rng.normal(0.0, sg) / self.visibility, 0.0, 1.0))
                    out[lv_idx] = float(np.clip(out[lv_idx] + self._noise_rng.normal(0.0, sv) / self.vmax, -1.0, 1.0))

                fg_idx, fv_idx = 3 + 4 * k, 4 + 4 * k
                if not (out[fg_idx] <= -1.0 + 1e-6 and out[fv_idx] <= -1.0 + 1e-6):
                    frac = float(np.clip(-out[fg_idx], 0.0, 1.0))
                    sg = self.noise_gap_base + self.noise_gap_slope * frac
                    sv = self.noise_vel_base + self.noise_vel_slope * frac
                    out[fg_idx] = float(np.clip(out[fg_idx] + self._noise_rng.normal(0.0, sg) / self.visibility, -1.0, 0.0))
                    out[fv_idx] = float(np.clip(out[fv_idx] + self._noise_rng.normal(0.0, sv) / self.vmax, -1.0, 1.0))
        return out

    def _scan_lane_neighbors(self, direction: int):
        """Scan target lane AND across junction boundary (via + outgoing lanes), respecting 1st-person occlusion."""
        front, rear = None, None
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            if edge.startswith(":") or li < 0 or li >= traci.edge.getLaneNumber(edge):
                return front, rear
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            my_len = traci.vehicle.getLength(self.ego_id)
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id or self._is_occluded(vid):
                    continue
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - my_len
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            remaining = traci.lane.getLength(lane_id) - my_pos
            if remaining < 35.0:
                for lk in traci.lane.getLinks(lane_id):
                    via_lane = lk[4] if len(lk) > 4 else ""
                    via_len = float(lk[7]) if len(lk) > 7 else 0.0
                    if via_lane:
                        for vid in traci.lane.getLastStepVehicleIDs(via_lane):
                            if self._is_occluded(vid):
                                continue
                            d = remaining + traci.vehicle.getLanePosition(vid)
                            g = d - traci.vehicle.getLength(vid)
                            if front is None or g < front[0]:
                                front = (g, traci.vehicle.getSpeed(vid))
                    next_lane = lk[0] if len(lk) > 0 else ""
                    if next_lane:
                        for vid in traci.lane.getLastStepVehicleIDs(next_lane):
                            if self._is_occluded(vid):
                                continue
                            vpos = traci.vehicle.getLanePosition(vid)
                            vlen = traci.vehicle.getLength(vid)
                            if vpos - vlen <= 5.0:
                                d = remaining + via_len + vpos
                                g = d - vlen
                                if front is None or g < front[0]:
                                    front = (g, traci.vehicle.getSpeed(vid))
        except traci.TraCIException:
            pass
        return front, rear

    def _neighbor(self, offset: int, leader: bool):
        """Fix zipper-merge foe blind spot on ego lane and lateral neighbor blind spots (gap < 0 / edge-end / spawn)."""
        if offset != 0:
            best = None

            def _consider(cand):
                nonlocal best
                if cand is None:
                    return
                g, v = float(cand[0]), float(cand[1])
                if g > self.visibility:
                    return
                g = max(0.0, g)
                if best is None or g < best[0]:
                    best = (g, v)

            try:
                if abs(offset) == 1:
                    mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
                    pairs = traci.vehicle.getNeighbors(
                        self.ego_id, mode_leader if leader else mode_follower
                    )
                    for vid, g in pairs or ():
                        if self._is_occluded(vid):
                            continue
                        _consider((g, traci.vehicle.getSpeed(vid)))
                front, rear = self._scan_lane_neighbors(offset)
                _consider(front if leader else rear)
                # If a vehicle in the target lane is physically overlapping side-by-side (gap <= 0),
                # the lateral slot itself is occupied in both front and rear directions.
                other = rear if leader else front
                if other is not None and float(other[0]) <= 0.0:
                    _consider((0.0, float(other[1])))
            except traci.TraCIException:
                pass
            return best

        res = super()._neighbor(offset, leader)
        try:
            if offset == 0 and leader:
                edge = traci.vehicle.getRoadID(self.ego_id)
                li = traci.vehicle.getLaneIndex(self.ego_id)
                lane_id = f"{edge}_{li}"
                if edge.startswith(":"):
                    pair = self.INTERNAL_MERGE_FOES.get(lane_id)
                    if pair:
                        foe_via, out_lane = pair
                        rem_ego = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(self.ego_id)
                        foe_vlen = traci.lane.getLength(foe_via)
                        for vid in traci.lane.getLastStepVehicleIDs(foe_via):
                            if self._is_occluded(vid):
                                continue
                            rem_foe = foe_vlen - traci.vehicle.getLanePosition(vid)
                            if rem_foe <= rem_ego + 8.0:
                                g = max(0.0, (rem_ego - rem_foe) - traci.vehicle.getLength(vid))
                                v = float(traci.vehicle.getSpeed(vid))
                                if res is None or g < res[0]:
                                    res = (g, v)
                        for vid in traci.lane.getLastStepVehicleIDs(out_lane):
                            if self._is_occluded(vid):
                                continue
                            g = max(0.0, rem_ego + traci.vehicle.getLanePosition(vid) - traci.vehicle.getLength(vid))
                            if g <= self.visibility:
                                v = float(traci.vehicle.getSpeed(vid))
                                if res is None or g < res[0]:
                                    res = (g, v)
                else:
                    foes = self.MERGE_FOES.get(lane_id)
                    if foes:
                        rem_ego = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(self.ego_id)
                        if rem_ego < 95.0:
                            ego_v = traci.vehicle.getSpeed(self.ego_id)
                            ego_eta = rem_ego / max(1.0, ego_v)
                            for foe_lane, foe_via in foes:
                                if foe_via:
                                    for vid in traci.lane.getLastStepVehicleIDs(foe_via):
                                        if self._is_occluded(vid):
                                            continue
                                        g = max(0.0, rem_ego + traci.vehicle.getLanePosition(vid) - traci.vehicle.getLength(vid))
                                        if g <= self.visibility:
                                            v = float(traci.vehicle.getSpeed(vid))
                                            if res is None or g < res[0]:
                                                res = (g, v)
                                foe_len = traci.lane.getLength(foe_lane)
                                for vid in traci.lane.getLastStepVehicleIDs(foe_lane):
                                    if self._is_occluded(vid):
                                        continue
                                    rem_foe = foe_len - traci.vehicle.getLanePosition(vid)
                                    fv = float(traci.vehicle.getSpeed(vid))
                                    foe_eta = rem_foe / max(1.0, fv)
                                    if rem_foe <= rem_ego + 5.0 or (foe_eta <= ego_eta + 0.6 and rem_foe <= rem_ego + 15.0):
                                        g = max(0.0, (rem_ego - rem_foe) - traci.vehicle.getLength(vid))
                                        if g <= self.visibility and (res is None or g < res[0]):
                                            res = (g, fv)
        except traci.TraCIException:
            pass
        return res

    def _lane_density(self, offset: int) -> float:
        if not (self._occ_sensor_mask and self._occ_boxes):
            return super()._lane_density(offset)
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return 1.0
            lane_id = f"{edge}_{lane_index}"
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = traci.lane.getLinks(lane_id)
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0
            count = 0
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id or self._is_occluded(vid):
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if self._is_occluded(vid):
                        continue
                    if traci.vehicle.getLanePosition(vid) <= self.visibility - remaining:
                        count += 1
            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0
