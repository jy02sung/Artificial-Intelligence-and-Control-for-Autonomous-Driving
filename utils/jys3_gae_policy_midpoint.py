"""Evaluate one prespecified weight midpoint between sibling pure PPO policies."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import torch

from utils import jys3_m5ft_ablation as ft

L, C = ft.L, ft.C
SOURCE = ft.ROOT.parent / "jys3_m5ft_gae_trial"
DEST = ft.ROOT.parent / "jys3_gae_policy_midpoint"
STEP = 100000
FRESH = (1867032100, 16)
ARMS = {
    "lambda95": "0a645c7bf6e061cdacdcd631acb7823697472cbdaaf02c093da9607e9a3cc403",
    "lambda98": "c8e4d87c0225cb6adbfad7fd7c01eb04eaa8514d95bd1cba737fdab442392f3d",
}
POLICY = DEST / "midpoint.pt"


def make_policy():
    states = []
    starts = []
    for arm, digest in ARMS.items():
        directory = SOURCE / arm / "s33"
        path = directory / f"policy_{STEP}.pt"
        assert ft.sha(path) == digest
        states.append(torch.load(path, map_location="cpu", weights_only=True))
        starts.append(L.read(directory / "start.json"))
    assert starts[0] == starts[1]
    left, right = states
    for key in ("step", "obs_dim", "hidden_sizes", "activation"):
        assert left[key] == right[key]
    for key in ("policy", "value_trunk"):
        assert left[key].keys() == right[key].keys()
        left[key] = {k: torch.lerp(v, right[key][k], 0.5) for k, v in left[key].items()}
    if POLICY.exists():
        saved = torch.load(POLICY, weights_only=True)
        for key in ("policy", "value_trunk"):
            assert all(torch.equal(v, saved[key][k]) for k, v in left[key].items())
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        tmp = POLICY.with_suffix(".tmp")
        torch.save(left, tmp)
        tmp.replace(POLICY)


def shard(job):
    panel, rep = job
    if panel == "fresh":
        C.old.PANELS["fresh"] = FRESH
    C.evaluate("M5-FT", 33, STEP, panel, rep, directory=DEST, checkpoint=POLICY, lateral=True)


def evaluate(pool, panels):
    jobs = [(panel, rep) for panel in panels for rep in range(panels[panel][1])]
    list(pool.map(shard, jobs))
    rows = []
    for panel, rep in jobs:
        file = f"{STEP}_{panel}_{rep}.json"
        part = L.read(DEST / "parts" / file)
        base = panels[panel][0]
        assert [r["seed"] for r in part] == [base + 100*rep + i for i in range(1,17)]
        if panel != "fresh":
            for arm in ARMS:
                ref = L.read(SOURCE / arm / "s33" / "parts" / file)
                assert all((a["initial_sha256"], a["seed"], a["lane"]) ==
                           (b["initial_sha256"], b["seed"], b["lane"]) for a,b in zip(ref, part, strict=True))
        rows.extend(part)
    assert len(rows) == len({r["seed"] for r in rows})
    counts = Counter(r["outcome"] for r in rows)
    return {k: counts[k] for k in ("arrival", "collision", "off_route", "stall", "timeout")}


def main():
    make_policy()
    with ProcessPoolExecutor(max_workers=4) as pool:
        screen = evaluate(pool, {"screen": C.old.PANELS["screen"]})
        results = {"screen": screen}
        L.save(DEST / "summary.json", results)
        print(results, flush=True)
        if screen["arrival"] >= 116 and screen["collision"] <= 6:
            results["holdout"] = evaluate(pool, {k: C.old.PANELS[k] for k in L.HOLDS})
            results["fresh"] = evaluate(pool, {"fresh": FRESH})
            L.save(DEST / "summary.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    main()
