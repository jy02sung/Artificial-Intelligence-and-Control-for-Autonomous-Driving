"""Separate along-route merge lookahead from physical sensor visibility."""
import math
from env.junction_conflict_trials import ConflictEnv, conflict_risk
from env.route_following_trials import traci


def visible_conflict(ego_distance, physical_distance, planning_horizon, visibility):
    return ego_distance <= planning_horizon and physical_distance <= visibility


class JunctionLookaheadEnv(ConflictEnv):
    def __init__(self, *args, condition=None, **kwargs):
        if condition is None:
            args, condition = args[:-1], args[-1]
        assert condition in ('near50', 'ahead150')
        self.planning_horizon = 50 if condition == 'near50' else 150
        super().__init__(*args, condition='observation', **kwargs)

    def conflict_features(self):
        v = traci.vehicle
        if self.ego_id not in v.getIDList():
            return (0., 0., 0.)
        pos = v.getPosition(self.ego_id)
        speed = v.getSpeed(self.ego_id)
        best = (0., 0., 0.)
        for item in v.getJunctionFoes(self.ego_id, self.planning_horizon):
            vid, ed, fd, ex, fx = item[:5]
            if ex < -v.getLength(self.ego_id) or fx < -v.getLength(vid):
                continue
            q = v.getPosition(vid)
            distance = math.hypot(q[0] - pos[0], q[1] - pos[1])
            if not visible_conflict(ed, distance, self.planning_horizon, self.visibility):
                continue
            value = conflict_risk(ed, fd, speed, v.getSpeed(vid))
            if value[0] > best[0]:
                best = value
        return best
