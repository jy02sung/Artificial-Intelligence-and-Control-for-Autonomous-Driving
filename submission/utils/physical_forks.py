from utils import route_rewards as v3
ARMS, j = v3.ARMS, v3.j


class Env(v3.Env):
    """Score the original mission while SUMO traverses links of the occupied lane."""

    def __init__(self, arm, folder, seed):
        super().__init__(arm, folder, seed)
        self.sim.physical_forks = True
        # Keep driving through even a short wrong edge until the terminal observation.
        self.continuations = {}
        for path in self.routes.values():
            for i in range(len(path) - 1):
                key = path[i], path[i + 1]
                if key not in self.continuations or len(self.continuations[key]) < len(path) - i:
                    self.continuations[key] = path[i:]

    def reset(self, **kwargs):
        self.alternate = None
        return super().reset(**kwargs)

    def departure(self, previous, previously_intact, info):
        if self.sim.crossed_short_link and info["route_intact"]:
            return None
        return super().departure(previous, previously_intact, info)

    def step(self, action):
        edge, lane, mission, index, *_ = self.snapshot
        exits = self.links.get(edge, {})
        if len(exits) > 1 and index + 1 < len(mission) and edge == mission[index]:
            expected = mission[index + 1]
            available = [dst for dst, lanes in exits.items() if lane in lanes]
            # A shared lane can follow the mission. A lane with no link remains a real dead end.
            alternate = available[0] if available and expected not in available else None
            if alternate != self.alternate:
                path = mission[index:]
                if alternate is not None:
                    path = self.continuations.get((edge, alternate), [edge, alternate])
                j.traci.vehicle.setRoute(self.ego_id, path)
                self.alternate = alternate
        return super().step(action)
