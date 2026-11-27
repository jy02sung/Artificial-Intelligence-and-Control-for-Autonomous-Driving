"""Lane-driven SUMO forks for the jys3 simple-reward PPO environment."""
import argparse
import hashlib
from pathlib import Path

from utils import jys3_reward_simple_v3 as v3

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--train", action="store_true")
    task.add_argument("--evaluate", action="store_true")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--panel", type=int, default=v3.SCREEN)
    parser.add_argument("--case", type=int, choices=range(4))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    stage = v3.ROOT.parent / "physical_forks"
    if args.train:
        if args.checkpoint or args.case is not None or args.steps < 2:
            parser.error("Training uses its own resume state and needs at least two steps.")
        v3.train_job(args.arm, args.seed, args.steps, stage, env_type=Env)
    else:
        if args.checkpoint is not None:
            # Frozen historical policies get a separate output namespace from newly trained ones.
            digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            stage = stage / "frozen" / digest
        (stage / f"{args.arm}_{args.seed}").mkdir(parents=True, exist_ok=True)
        for rep in ([args.case] if args.case is not None else range(4)):
            v3.eval_job(stage, args.arm, args.seed, args.steps, rep, args.panel,
                        env_type=Env, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
