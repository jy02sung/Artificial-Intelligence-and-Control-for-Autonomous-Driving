"""Train or evaluate a new PPO on the frozen release scenario. Run with python -m utils.entryprep_av."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

# Enable in-process simulation where available. GUI replay always uses TraCI.
if "--gui" in sys.argv:
    os.environ.pop("IC_LIBSUMO", None)
else:
    try:
        import libsumo
    except ImportError:
        pass
    else:
        os.environ["IC_LIBSUMO"] = "1"

import numpy as np
import torch

from algorithms.ppo import PPO
from env.entryprep_av import DRIVE_LIMIT, EntryEnv, OBS_SIZE, RANGE, ROUTES, SCENARIO, WARMUP, input_hashes
from env.sumo_env import traci

ROOT = SCENARIO.parents[1]
DEFAULT_OUT = ROOT / "results/entryprep_av"
TRAIN_SEEDS = tuple(range(12001, 12009))
PARAMS = json.loads((ROOT / "docs/experiments/ic-navigation-plan-v1.json").read_text())["hparams"]
CHECK_FILES = ("env/entryprep_av.py", "utils/entryprep_av.py", "env/sumo_env.py", "env/ic_experiment.py",
               "env/navigation_observation.py", "env/navigation_clarity.py", "algorithms/ppo.py", "utils/networks.py",
               "utils/buffer.py", "env/mdp_config.py", "env/road_config.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(f.name, path)


def snapshot(cache, seed):
    cache.mkdir(exist_ok=True, parents=True)
    path = cache / f"warm-{seed}.xml.gz"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            new = Path(tmp) / path.name
            subprocess.run([shutil.which("sumo") or "sumo", "-c", str(SCENARIO / "run.sumocfg"), "--seed", str(seed),
                            "--save-state.times", str(WARMUP), "--save-state.files", str(new),
                            "--save-state.rng", "true", "--save-state.precision", "16", "--end", str(WARMUP+.2),
                            "--no-step-log", "true", "--no-warnings", "true"], check=True)
            os.replace(new, path)
    return path


class TrainingEpisodes:
    """Balance route visits and vary real traffic across a separate set of training seeds."""
    def __init__(self, env, cache, seed):
        self.env, self.cache = env, cache
        self.rng = np.random.default_rng(seed)
        self.routes = []
        self.counts = collections.Counter()
        self.finished = collections.Counter()

    def reset(self):
        if not self.routes:
            self.routes = list(self.rng.permutation(ROUTES))
        route = str(self.routes.pop())
        traffic_seed = int(self.rng.choice(TRAIN_SEEDS))
        self.env.snapshot = snapshot(self.cache, traffic_seed)
        self.counts[route] += 1
        return self.env.reset(seed=traffic_seed, options={"route": route})

    def step(self, action):
        obs, reward, done, cut, info = self.env.step(action)
        if done or cut:
            outcome = "collision" if info["collided"] else "arrival" if info["arrived"] else "other"
            self.finished[outcome] += 1
        return obs, reward, done, cut, info


def train(args):
    out = args.run
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps partial and completed experiments from being overwritten.
    with (out / "manifest.json").open("x", encoding="utf8") as fp:
        meta = dict(condition=args.condition, model_seed=args.model_seed, steps=args.steps,
                    inputs=input_hashes(), source={k: digest(ROOT / k) for k in CHECK_FILES},
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    params=dict(PARAMS, seed=args.model_seed), traffic_seeds=TRAIN_SEEDS, routes=ROUTES,
                    observations=OBS_SIZE, warmup_seconds=WARMUP, training_timeout_seconds=240,
                    evaluation_timeout_seconds=DRIVE_LIMIT, speedMode=0, laneChangeMode=0,
                    lane_change_safety=False, assisted_initialization=args.pretrain,
                    sumo=subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0])
        json.dump(meta, fp, indent=2)
    env = EntryEnv(condition=args.condition, seed=args.model_seed, training=True)
    driver = PPO(OBS_SIZE, 2, **meta["params"])
    if args.pretrain:
        pretrain(driver, args.model_seed, args.condition)
    episodes = TrainingEpisodes(env, args.cache, args.model_seed)
    try:
        obs, _ = episodes.reset()
        stats = [0., 0]
        start = time.monotonic()
        while driver.num_timesteps < args.steps:
            n = min(driver.n_steps, args.steps - driver.num_timesteps)
            obs, value, _ = driver.collect_rollout(episodes, obs, ep_stats=stats, steps=n)
            adv, ret = driver.buffer.compute_gae(value, driver.gamma, driver.gae_lambda)
            driver.update(adv, ret)
            temp = out / "policy.tmp"
            driver.save(temp)
            os.replace(temp, out / "policy.pt")
            progress = dict(steps=driver.num_timesteps, episodes=episodes.counts, outcomes=episodes.finished,
                            elapsed_seconds=round(time.monotonic()-start, 1))
            json_write(out / "progress.json", progress)
            print(progress, flush=True)
        json_write(out / "complete.json", dict(**progress, policy_sha256=digest(out / "policy.pt")))
    finally:
        env.close()


def pretrain(driver, seed, condition):
    """Optional supervised initialization from procedurally sampled car-following and lane-choice examples.
    Only the neural policy runs in SUMO. The demonstrator is never called on evaluation episodes.
    """
    rng = np.random.default_rng(seed)
    n = 32768
    obs = np.zeros((n, OBS_SIZE), np.float32)
    speed = rng.uniform(0, 27.78, n)
    limit = rng.choice([11.11, 16.67, 19.44, 22.22, 27.78], n)
    obs[:, 0] = speed / 27.78
    obs[:, 34] = limit / 27.78
    risk = {}
    for off_index, off in enumerate((2, 1, 0, -1, -2)):
        obs[:, 21+off_index] = rng.choice([-1., 1., 1., 1.], n) if off else 1.
        obs[:, 26+off_index] = rng.uniform(0, .4, n)
        for side, front in enumerate((True, False)):
            near = rng.random(n) < .68
            gap = rng.uniform(0., RANGE, n)
            other = rng.uniform(0, 27.78, n)
            visible = near & (gap <= 50.) & (obs[:,21+off_index] != -1)
            col = 1+off_index*4+side*2
            sign = 1. if front else -1.
            obs[:,col] = np.where(visible, sign*gap/50., sign)
            obs[:,col+1] = np.where(visible, (other-speed)/27.78, sign)
            closing = np.maximum(0., speed-other if front else other-speed)
            needed = 2.5+.7*(speed if front else other)+closing
            risk[off,front] = np.where(visible,np.maximum(0.,1.-gap/needed),0.)
            if off==0 and front:
                ahead = near if condition=="risk" else visible
                effective_gap = np.where(ahead, gap, 10000.)
                lead_speed = other
                if condition=="risk":
                    obs[:,47] = near
                    obs[:,48] = np.where(near,gap/RANGE,0.)
                    obs[:,49] = np.where(near,(other-speed)/27.78,0.)
    desired = rng.choice([-2,-1,0,0,0,1,2], n)
    desired = np.where((desired>0) & (obs[:,22]<0) | (desired<0) & (obs[:,24]<0), 0, desired)
    obs[:,31] = desired/3.
    obs[:,32] = rng.uniform(-.05,1.,n)
    obs[:,33] = desired/3.
    obs[:,35] = rng.integers(0,2,n)
    if condition=="risk":
        obs[:,50] = desired/4.
        obs[:,51:53] = rng.uniform(0,1,(n,2))
        for i,key in enumerate(((1,True),(1,False),(-1,True),(-1,False))):
            obs[:,53+i] = risk[key]
    d = np.maximum(0., speed*1.4 + speed*(speed-lead_speed)/(2*np.sqrt(2.*3.5)))
    accel = 2. * (1.-(speed/np.maximum(2.,limit*.85))**4 - ((3.+d)/np.maximum(.2,effective_gap))**2)
    lane = np.sign(desired)
    for side in (-1,1):
        unsafe = (risk[side,True] > .02) | (risk[side,False] > .02)
        lane[(lane==side) & unsafe] = 0
    if condition=="risk":
        lane[obs[:,51] < .20] = 0
    x = torch.from_numpy(obs).to(driver.device)
    y = torch.tensor(np.clip(accel/5.4,-1.,1.),dtype=torch.float32,device=driver.device)
    z = torch.tensor(1-lane,dtype=torch.int64,device=driver.device)
    for _ in range(16):
        for ix in rng.permutation(n).reshape(-1,256):
            mu, _, logits, _ = driver.policy(x[ix])
            loss = torch.nn.functional.smooth_l1_loss(mu[:,0],y[ix]) + torch.nn.functional.cross_entropy(logits,z[ix])
            driver.optimizer.zero_grad()
            loss.backward()
            driver.optimizer.step()


EVAL_FIELDS = ("seed", "route", "outcome", "seconds", "edge", "insert_wait_seconds",
               "route_overrides", "collision_partner", "last_speed_mps", "last_accel", "last_lane")


def csv_write(path, rows, fields):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(fp.name, path)


def evaluate_route(env, driver, seed, route):
    row = dict.fromkeys(EVAL_FIELDS, "")
    row.update(seed=seed, route=route, seconds=0., route_overrides=0)
    try:
        obs, _ = env.reset(seed=seed, options={"route": route})
    except RuntimeError as exc:
        if "삽입 실패" not in str(exc):
            raise
        row.update(outcome="insertion_failure", insert_wait_seconds=DRIVE_LIMIT)
        return row
    row["insert_wait_seconds"] = round(traci.simulation.getTime() - WARMUP - env.step_length, 1)
    with torch.inference_mode():
        while True:
            edge = traci.vehicle.getRoadID(env.ego_id)
            speed = float(obs[0]) * env.vmax
            action = driver.predict(obs, deterministic=True)
            obs, _, done, cut, info = env.step(action)
            if info["collided"]:
                row["outcome"] = "collision"
                partners = set()
                for contact in traci.simulation.getCollisions():
                    if env.ego_id in (contact.collider, contact.victim):
                        partners.update((contact.collider, contact.victim))
                row["collision_partner"] = ",".join(sorted(partners - {env.ego_id}))
            elif env.ego_id in traci.simulation.getStartingTeleportIDList():
                row["outcome"] = "teleport"
            elif done:
                row["outcome"] = "arrival" if info["arrived"] and edge == env.routes[route][-1] else "wrong_exit"
            elif cut:
                row["outcome"] = "timeout"
            else:
                continue
            row.update(seconds=round(env.step_count * env.step_length, 1), edge=edge,
                       route_overrides=env.route_override_count, last_speed_mps=round(speed, 3),
                       last_accel=round(float(action[0]), 5), last_lane=info["lane_change"])
            return row


def evaluate_seed(job):
    seed, routes, run, cache, cached, gui, limit, shard = job
    if shard.is_file() and not gui:
        with shard.open(newline="", encoding="utf8") as fp:
            rows = list(csv.DictReader(fp))
        if [r["route"] for r in rows] != list(routes) or any(int(r["seed"]) != seed for r in rows):
            raise ValueError(f"Unexpected seed results in {shard}")
        return rows
    meta = json.loads((run / "manifest.json").read_text(encoding="utf8"))
    torch.set_num_threads(1)
    driver = PPO(OBS_SIZE, 2, **meta["params"])
    driver.load(run / "policy.pt")
    driver.policy.eval()
    env = EntryEnv(condition=meta["condition"], seed=meta["model_seed"], gui=gui)
    env.max_steps = round(limit / env.step_length)
    try:
        env.snapshot = snapshot(cache, seed) if cached else None
        rows = []
        for route in routes:
            row = evaluate_route(env, driver, seed, route)
            rows.append(row)
            if len(routes) < len(ROUTES) or gui:
                print(seed, route, row["outcome"], row["seconds"], flush=True)
        csv_write(shard, rows, EVAL_FIELDS)
    finally:
        env.close()
    print(seed, dict(collections.Counter(r["outcome"] for r in rows)), flush=True)
    return rows


def evaluate(args):
    out = args.run
    meta = json.loads((out / "manifest.json").read_text(encoding="utf8"))
    complete = json.loads((out / "complete.json").read_text(encoding="utf8"))
    checkpoint = out / "policy.pt"
    if meta["inputs"] != input_hashes() or complete["policy_sha256"] != digest(checkpoint):
        raise ValueError("Model or frozen scenario differs from training run")
    seeds = list(range(args.seed, args.seed + args.seeds))
    if set(seeds).intersection(meta["traffic_seeds"]):
        raise ValueError("Evaluation and training seeds overlap")
    routes = [f"route_{i:02}" for i in args.routes] if args.routes else list(ROUTES)
    if len(routes) != len(set(routes)) or any(r not in ROUTES for r in routes):
        raise ValueError("Expected unique route IDs in 1..16")
    protocol = "snapshot" if args.cached_start else "fresh"
    route_label = "all" if routes == list(ROUTES) else "-".join(r[-2:] for r in routes)
    version = dict(policy=complete["policy_sha256"], inputs=meta["inputs"],
                   software={k: digest(ROOT / k) for k in CHECK_FILES},
                   sumo=subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0],
                   backend=traci.__name__, protocol=protocol, drive_limit=args.drive_limit, routes=routes)
    signature = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()[:16]
    shard_dir = out / "evaluations" / signature
    shard_dir.mkdir(exist_ok=True, parents=True)
    json_write(shard_dir / "protocol.json", version)
    jobs = [(seed, routes, out, args.cache, args.cached_start, args.gui, args.drive_limit,
             shard_dir / f"{seed}.csv") for seed in seeds]
    if args.workers == 1:
        results = [row for job in jobs for row in evaluate_seed(job)]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = [row for batch in pool.map(evaluate_seed, jobs) for row in batch]
    summary_path = out / f"eval-{protocol}-{seeds[0]}-{seeds[-1]}-{route_label}-{args.drive_limit}s.csv"
    csv_write(summary_path, results, EVAL_FIELDS)
    print(dict(collections.Counter(r["outcome"] for r in results)), summary_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--run", type=Path, required=True)
    train_parser.add_argument("--condition", choices=("baseline", "risk"), required=True)
    train_parser.add_argument("--steps", type=int, default=32768)
    train_parser.add_argument("--model-seed", type=int, default=22)
    train_parser.add_argument("--cache", type=Path, default=DEFAULT_OUT/"states")
    train_parser.add_argument("--pretrain", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--seed", type=int, default=9101)
    ev.add_argument("--seeds", type=int, default=1)
    ev.add_argument("--routes", type=int, nargs="+")
    ev.add_argument("--gui", action="store_true")
    ev.add_argument("--workers", type=int, default=1, help="Parallel independent traffic seeds")
    ev.add_argument("--drive-limit", type=int, default=DRIVE_LIMIT,
                    help="Seconds after insertion. The release uses 10800 and the 100-seed plan uses 900.")
    ev.add_argument("--cached-start", action="store_true",
                    help="Use a saved 900 s traffic state for faster development comparisons")
    ev.add_argument("--cache", type=Path, default=DEFAULT_OUT/"states")
    args=parser.parse_args()
    if min(getattr(args, k, 1) for k in ("steps", "seeds", "workers", "drive_limit")) <= 0:
        parser.error("Episode and step counts must be positive")
    if getattr(args, "gui", False) and args.workers != 1:
        parser.error("GUI replay uses one worker")
    torch.set_num_threads(1)
    (train if args.stage=="train" else evaluate)(args)


if __name__ == "__main__":
    main()
