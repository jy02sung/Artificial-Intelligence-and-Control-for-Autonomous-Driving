"""Run a released team policy in the physical-fork SUMO environment."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))


def model_path(name):
    record = MODELS[name]
    path = ROOT / "models" / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="dagger5")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a newly trained checkpoint using --model observation settings.")
    parser.add_argument("--route", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--seed", type=int, default=104320008)
    parser.add_argument("--lane", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Run all 576 presentation cases.")
    args = parser.parse_args()
    os.environ["IC_LIBSUMO"] = "0" if args.gui else "1"

    import torch
    from utils import imitation as final

    torch.set_num_threads(1)
    policy = final.network(args.checkpoint or model_path(args.model))
    outcomes = Counter()
    cases = [(args.route, args.seed, args.lane)]
    if args.holdout:
        cases = [(route, base + rep * 100 + route, rep % 2)
                 for base, reps in ((81260100, 4), (91000000, 16), (104320000, 16))
                 for rep in range(reps) for route in range(1, 17)]
    with final.simulator() as env, torch.inference_mode():
        assert env.sim.physical_forks and env.sim.occlusion and not env.sim.lc_safety_enabled
        env.sim.gui = args.gui
        transform = None
        if args.model == "pure_ppo":
            from utils.ppo_rollout import lateral_transform
            transform = lateral_transform(env)
        for route, seed, lane in cases:
            route_id = f"route_{route:02d}"
            lane = min(lane, env.counts[env.routes[route_id][0]] - 1)
            env.ego_depart_lane = str(lane)
            obs, _ = env.reset(seed=seed, options={"route": route_id})
            for step in range(1, 1001):
                state = obs[:50]
                if transform:
                    state = transform(state)
                action = policy.det_action(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            assert (done or truncated) and info["outcome"] in final.OUTCOMES
            assert info["outcome"] != "arrival" or info["route_intact"]
            outcomes[info["outcome"]] += 1
            print(f"{args.model} route={route} lane={lane} seed={seed} steps={step} {info['outcome']}", flush=True)
    print(json.dumps({key: outcomes[key] for key in final.OUTCOMES}))


if __name__ == "__main__":
    main()
