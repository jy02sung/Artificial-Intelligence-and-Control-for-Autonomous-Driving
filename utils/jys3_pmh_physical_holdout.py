"""Paired frozen replay of the pmh round-10 fixed policy on its three published panels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

os.environ.setdefault("IC_LIBSUMO", "1")
from utils.jys3_imitation_physical import dagger, drive, physical, v3

SOURCE = "2bf7d863c746bf9f25369f2dc6ab0bf6a6067577"
SOURCE_PATH = "models/pmh/jys3_dagger10_fixed.pt"
DIGEST = "502ce37d120f2c4a9dd55b6e2754e9d74ac486c698b87cf8c55b7e4fd421f2d2"
ROOT = dagger.BC.parent / "jys3_pmh_physical_holdout"
PANELS = {"final256": (30010000, 16), "reference64": (26010000, 4),
          "round_holdout64": (29010000, 4)}
PUBLISHED = {"final256": {"arrival": 245, "collision": 2, "stall": 9},
             "reference64": {"arrival": 60, "stall": 4},
             "round_holdout64": {"arrival": 63, "stall": 1}}


def replay(job):
    panel, rep, weights = job
    dagger.torch.set_num_threads(1)
    model = dagger.policy(weights)
    records, starts = [], []
    with TemporaryDirectory() as tmp:
        old = dagger.environment("high_real_ratio", 11, Path(tmp) / "old")
        try:
            for route in range(1, 17):
                lane = min(rep % 2, old.counts[old.routes[f"route_{route:02d}"][0]] - 1)
                seed = PANELS[panel][0] + 100 * rep + route
                initial, outcome = drive(old, model, route, seed, lane, False)
                starts.append(initial)
                records.append(dict(route=route, seed=seed, depart_lane=lane, old=outcome))
        finally:
            old.close()
        new = physical.Env(v3.ARMS[0], Path(tmp) / "new", 11)
        new.sim.signs, new.sim.road_flags = old.signs, old.road_flags
        try:
            for row, initial in zip(records, starts):
                obs, outcome = drive(new, model, row["route"], row["seed"], row["depart_lane"], True)
                assert (obs == initial).all() and row["old"]["start"] == outcome["start"]
                row["new"] = outcome
        finally:
            new.close()
    data = dict(panel=panel, rep=rep, sha256=DIGEST, initial_observations_equal=True, rows=records)
    dest = ROOT / f"{panel}_{rep:02d}.json"
    part = dest.with_suffix(".tmp")
    part.write_text(json.dumps(data, indent=2))
    part.replace(dest)
    return panel, rep


def summarize(panels):
    output = {}
    for panel in panels:
        rows = [row for rep in range(PANELS[panel][1])
                for row in json.loads((ROOT / f"{panel}_{rep:02d}.json").read_bytes())["rows"]]
        counts = {side: dict(Counter(r[side]["outcome"] for r in rows)) for side in ("old", "new")}
        assert counts["old"] == PUBLISHED[panel], (panel, counts)
        if panel == "final256":
            failures = Counter((r["old"]["outcome"], r["route"]) for r in rows
                               if r["old"]["outcome"] != "arrival")
            assert failures == {("collision", 6): 1, ("collision", 13): 1,
                                ("stall", 5): 2, ("stall", 13): 3, ("stall", 14): 4}, failures
        assert len(set(row["seed"] for row in rows)) == 16 * PANELS[panel][1]
        output[panel] = dict(episodes=len(rows), **counts,
                             transitions=dict(Counter(f'{r["old"]["outcome"]}->{r["new"]["outcome"]}'
                                                      for r in rows)))
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the frozen pmh Git blob")
    parser.add_argument("--panels", choices=PANELS, nargs="+", default=list(PANELS))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory() as tmp:
        weights = args.checkpoint
        if weights is None:
            weights = Path(tmp) / "dagger10.pt"
            weights.write_bytes(subprocess.check_output(["git", "show", f"{SOURCE}:{SOURCE_PATH}"],
                                                        cwd=dagger.m.ROOT))
        assert sha256(weights.read_bytes()).hexdigest() == DIGEST
        jobs = [(p, rep, weights) for p in args.panels for rep in range(PANELS[p][1])]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for panel, rep in pool.map(replay, jobs):
                print(f"finished {panel} case {rep}", flush=True)
    summarize(args.panels)


if __name__ == "__main__":
    main()
