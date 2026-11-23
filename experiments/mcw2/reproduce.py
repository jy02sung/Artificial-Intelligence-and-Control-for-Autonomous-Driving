"""Verify archived evidence, replay matched traffic seeds, or retrain the published policy."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/local_mcw2_driver_ppo"
INITIAL = ROOT / "results/mcw_runs/mcw2_agenda2_dagger_v2/best_model.pt"
FINAL = ARCHIVE / "merge_12.pt"


def read(name):
    return json.loads((ARCHIVE / name).read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    before = read("saved_paired_holdout.json")
    after = read("merge_12_original_holdout.json")
    plan = read("original_only_freeze.json")
    paired = read("paired_effect.json")
    hashes = [digest(path) for path in (INITIAL, ARCHIVE / "gap_4.pt", FINAL)]
    training_seeds = set()
    for i, (name, steps) in enumerate([("gap_4", 8192), ("merge_12", 24576)]):
        log = read(name + "_train.json")
        if (log["parent_sha256"], log["sha256"], log["steps"]) != (hashes[i], hashes[i+1], steps):
            raise ValueError(f"Checkpoint ancestry or training length differs for {name}")
        training_seeds.update(seed for _, seed in log["episodes"])
    if (before["sha256"], after["sha256"], plan["sha256"], plan["name"]) != (hashes[0], hashes[2], hashes[2], "merge_12"):
        raise ValueError("Evaluation weights differ from the frozen weights")
    seeds = {seed for _, seed in plan["seeds"]}
    if len(seeds) != 1600 or training_seeds & seeds:
        raise ValueError("The holdout must contain 1,600 distinct unused seeds")
    for stage in ("screen", "dev"):
        if any(row["seed"] in seeds for row in read(f"merge_12_{stage}.json")["rows"]):
            raise ValueError("Selection and holdout seeds overlap")
    for key, result in [("before", before), ("after", after)]:
        rows = result["rows"]
        if [[r["route"], r["seed"]] for r in rows] != plan["seeds"]:
            raise ValueError("Mismatch in route or traffic seed")
        if Counter(r["route"] for r in rows) != {f"route_{i:02}": 100 for i in range(1, 17)}:
            raise ValueError("Unexpected number of runs per route")
        counts = Counter(r["outcome"] for r in rows)
        if counts != result["counts"] or counts != paired[key]:
            raise ValueError("Episode counts differ from the published results")
    pairs = list(zip(before["rows"], after["rows"]))
    for outcome in ("arrival", "collision"):
        gained = sum(a["outcome"] != outcome and b["outcome"] == outcome for a, b in pairs)
        lost = sum(a["outcome"] == outcome and b["outcome"] != outcome for a, b in pairs)
        change = paired["changes"][outcome]
        if (gained,lost,gained-lost) != (change["gained"],change["lost"],change["net"]):
            raise ValueError(f"Paired changes differ for {outcome}")
    print(f"Verified {len(pairs)} pairs. Before {before['counts']}. After {after['counts']}.", flush=True)
    return pairs


def setup_worker(checkpoint, scratch):
    from . import local_driver_ppo as driver
    driver.D = Path(scratch)
    driver.setup(checkpoint)


def replay(per_route, workers):
    from .local_driver_ppo import drive
    pairs = records()
    seen = Counter()
    selected = []
    for pair in pairs:
        route = pair[0]["route"]
        if seen[route] < per_route:
            selected.append(pair)
            seen[route] += 1
    tasks = [(a["route"], a["seed"]) for a, _ in selected]
    with TemporaryDirectory(prefix="mcw2-replay-") as scratch:
        for i, checkpoint in enumerate((INITIAL, FINAL)):
            with ProcessPoolExecutor(workers, initializer=setup_worker, initargs=(checkpoint, scratch)) as pool:
                for actual, pair in zip(pool.map(drive, tasks), selected):
                    expected = pair[i]
                    if any(actual[field] != expected[field] for field in ("route","seed","steps","outcome")):
                        raise ValueError(f"Replay mismatch for {checkpoint.name}: {actual} versus {expected}")
            print(f"Reproduced {len(tasks)} episodes for {checkpoint.name}, including their ending steps.", flush=True)


def train(output):
    from . import local_driver_gap_ppo as training
    from . import local_driver_ppo as driver
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory for retraining: {output}")
    driver.D = training.D = output
    for name, updates in [("gap", 4), ("merge", 12)]:
        training.train(name, updates=updates)
        path = output / f"{name}_{updates}.pt"
        actual = digest(path)
        expected = digest(ARCHIVE / path.name)
        if actual != expected:
            raise ValueError(f"{path.name} SHA-256 mismatch: {actual}, expected {expected}")
        print(f"Reproduced {path.name}, SHA-256 {actual}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Recalculate paired outcomes and check weights without SUMO")
    sim = commands.add_parser("replay", help="Drive both policies and compare each episode to the archived trace")
    sim.add_argument("--per-route", type=int, default=1, metavar="1..100")
    sim.add_argument("--workers", type=int, default=4)
    fit = commands.add_parser("train", help="Reproduce the two selected PPO training stages")
    fit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        records()
    elif args.command == "replay":
        if not 1 <= args.per_route <= 100 or args.workers < 1:
            parser.error("Use between 1 and 100 episodes per route and a positive worker count")
        replay(args.per_route, args.workers)
    else:
        train(args.output)


if __name__ == "__main__":
    main()
