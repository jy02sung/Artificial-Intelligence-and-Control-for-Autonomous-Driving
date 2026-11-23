"""Local experiment X. Use train, baseline, candidate, or final from this checkout."""
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from . import local_audit as audit

m = audit.m
ROOT = audit.ROOT
OUT = ROOT / "results" / "experiment_X"
SOURCE = ROOT / "results/mcw_runs/mcw2_occlusion_ppo/manifest.json"
SPEC = json.loads(SOURCE.read_bytes())
HARD = ["route_02", "route_05", "route_13", "route_14"]
SEED = 112301
BASE_HASH = "4dab15658aa3f85880df9165d655f932be3e092c4997758e515b9cb6092e4f7e"
X = OUT / "model_32768.pt"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def plan():
    rng = random.Random(SEED)
    blocks = {}
    for stage, n, lo in [("dev", 25, 910000000), ("final", 100, 920000000)]:
        seeds = iter(rng.sample(range(lo, lo + 1000000), 16*n))
        blocks[stage] = [[r, next(seeds)] for _ in range(n) for r in m.ROUTES]
    return blocks

def initialize(model, training):
    global env, agent
    audit.torch.set_num_threads(1)
    args = SimpleNamespace(**(SPEC["args"] | {"seed": SEED}))
    cfg = m.prepare_scenario(args.stage, OUT/"maps"/str(os.getpid()), color_by_type=args.color_by_type)
    road, nav = m.load_network_metadata(cfg.parent/"network.net.xml", cfg.parent/"highway.rou.xml", args.nav_horizon)
    env = m.make_env(cfg, road, nav, args, *m.build_configs(args))
    env.eval_mode = not training
    hp = SPEC["hparams"] | {"device": "cpu", "seed": SEED}
    agent = m.PPO(obs_dim=47, act_dim=2, buffer=m.RolloutBuffer(), **hp)
    agent.load(str(model))
    agent.policy.train(training)

def train():
    if X.exists():
        raise RuntimeError("Run already has a frozen candidate")
    initialize(audit.MODEL, True)
    route_rng = random.Random(SEED + 1)
    seeds = iter(random.Random(SEED + 2).sample(range(930000000, 931000000), 8000))
    upcoming = []
    episodes = []
    native_reset = env.reset

    def reset():
        if not upcoming:
            upcoming.extend(list(m.ROUTES) + HARD*4)
            route_rng.shuffle(upcoming)
        route, seed = upcoming.pop(), next(seeds)
        obs, info = native_reset(seed=seed, options={"route": route})
        episodes.append([route, seed])
        return obs, info

    env.reset = reset
    start = time.monotonic()
    try:
        obs, _ = env.reset()
        ep_stats = [0., 0]
        updates = []
        for i in range(1, 17):
            obs, value, _ = agent.collect_rollout(env, obs, ep_stats=ep_stats)
            adv, ret = agent.buffer.compute_gae(value, agent.gamma, agent.gae_lambda)
            stats = agent.update(adv, ret)
            assert all(math.isfinite(float(v)) for v in stats.values())
            updates.append(dict(update=i, steps=int(agent.num_timesteps), elapsed=round(time.monotonic()-start,2), **stats))
            write_json(OUT/"training.json", dict(updates=updates, episodes_started=episodes))
            print("PPO", i, agent.num_timesteps, "episodes", len(episodes), flush=True)
        assert agent.num_timesteps == 32768
        tmp = X.with_suffix(".tmp")
        agent.save(str(tmp))
        tmp.replace(X)
    finally:
        env.close()

def drive(task):
    route, seed = task
    obs, _ = env.reset(seed=seed, options={"route":route})
    late = 0
    location = {"route_02":("844484877_0",1), "route_13":("170059994_3",-1),
                "route_14":("170059994_3",-1)}.get(route)
    v = audit.V
    while True:
        action = agent.predict(obs, deterministic=True)
        if location and int(action[1]) == location[1] and v.getSpeed(env.ego_id) < 0.1:
            lane = v.getLaneID(env.ego_id)
            if lane == location[0] and m.traci.lane.getLength(lane)-v.getLanePosition(env.ego_id) <= 0.1:
                late += 1
        obs, _, done, cut, info = env.step(action)
        if done or cut:
            outcome = ("collision" if info["collided"] else "arrival" if info["arrived"] else "timeout" if cut else "other")
            return dict(route=route,seed=seed,outcome=outcome,steps=env.step_count,late_requests=late)

def evaluate(stage):
    if stage == "final":
        gate = json.loads((OUT/"candidate.json").read_bytes())["counts"]
        assert gate.get("arrival",0)>=380 and gate.get("collision",0)<=19
    model = audit.MODEL if stage == "baseline" else X
    tasks = plan()["final" if stage=="final" else "dev"]
    start = time.monotonic()
    with ProcessPoolExecutor(5, initializer=initialize, initargs=(model,False)) as pool:
        rows = []
        for row in pool.map(drive, tasks):
            rows.append(row)
            if len(rows)%80==0:
                print(stage,len(rows),flush=True)
    assert [(r["route"],r["seed"]) for r in rows]==[tuple(t) for t in tasks]
    with (OUT/(stage+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(model_sha256=digest(model), counts=Counter(r["outcome"] for r in rows),
                   seconds=round(time.monotonic()-start,1))
    write_json(OUT/(stage+".json"),summary)
    print(stage,summary,flush=True)
    return summary["counts"]

if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    assert digest(audit.MODEL)==BASE_HASH
    stage = sys.argv[1]
    if stage == "train":
        write_json(OUT/"protocol.json",dict(seeds=plan(),train_source=SPEC,source_sha256=BASE_HASH,
                   training_seed=SEED,device="cpu",extra_routes=HARD*4,steps=32768,
                   dev_min_arrival=380,dev_max_collision=19,final_min_arrival=1520,final_max_collision=79))
        train()
    elif stage == "candidate":
        result = evaluate(stage)
        if result.get("arrival",0)>=380 and result.get("collision",0)<=19:
            evaluate("final")
    elif stage in ("baseline","final"):
        evaluate(stage)
    else:
        raise ValueError(stage)
