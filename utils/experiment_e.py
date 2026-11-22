"""Experiment E: matched 11-route curriculum, route intent x early secured passage."""
import itertools
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from env import ic_experiment as ic
from utils.experiment_b import HP, digest, m, save, torch, traci
from utils.experiment_b_eval import evaluate_episode
from utils.experiment_c_train import atomic_torch
from utils.experiment_d_env import OUT as D, environment

OUT = D.parent / "experiment_e_v1"
ROUTES = (1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 13, 16)
STEPS = tuple(range(0, 200001, 50000))

def agent(seed):
    return m.PPO(obs_dim=41, act_dim=2, buffer=m.RolloutBuffer(), seed=seed, **HP)

def make(stage, arm, seed, folder):
    env = environment(stage, "D1", seed, folder, audit=True)
    reset, step = env.reset, env.step
    def encode(obs):
        cue = np.zeros(3, np.float32)
        if arm in ("E1", "E3") and env.snapshot:
            edge, _, path, _, j, *_ = env.snapshot
            k = 2
            if j is not None and path[j] not in ic.RAMPS:
                k = int(path[j+1] in ic.RAMPS)
            cue[k] = 1
        return np.concatenate((obs, cue))
    def start(*args, **kwargs):
        obs, info = reset(*args, **kwargs)
        return encode(obs), info
    def advance(action):
        obs, reward, done, cut, info = step(action)
        bonus = 0.
        if info["c_pass"] and arm in ("E2", "E3"):
            idx = env.snapshot[3]-1
            rec = env.decisions[idx]
            bonus = .25 * min(1., (rec["secured"] or 0.) / 300.)
        info["early_pass"] = bonus
        return encode(obs), reward + bonus, done, cut, info
    env.reset, env.step = start, advance
    from gymnasium.spaces import Box
    env.observation_space = Box(-1., 1., (41,), dtype=np.float32)
    return env

class Episodes:
    def __init__(self, env, seed, directory):
        self.env, self.seed, self.directory = env, seed, directory
        self.count = 0
    def reset(self):
        block, idx = divmod(self.count, len(ROUTES))
        order = list(ROUTES)
        random.Random(self.seed*100000 + block).shuffle(order)
        route = f"route_{order[idx]:02d}"
        lane = min(block % 2, self.env.counts[self.env.routes[route][0]]-1)
        sim = self.seed*1000000 + self.count
        self.env.ego_depart_lane = str(lane)
        obs, info = self.env.reset(seed=sim, options={"route":route})
        with (self.directory/"episodes.jsonl").open("a") as f:
            print(json.dumps([self.count,route,sim,self.env.snapshot[1],traci.vehicle.getLanePosition(self.env.ego_id),self.env.snapshot[7]]),file=f)
        self.count += 1
        return obs, info
    def step(self, action):
        return self.env.step(action)

def train(job):
    arm, seed = job
    torch.set_num_threads(1)
    folder = OUT/f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"policy_200000.pt").exists(): return
    if (folder/"episodes.jsonl").exists(): raise RuntimeError(f"Existing partial run {folder}")
    a = agent(seed)
    save(folder/"initial.json", dict(seed=seed, digest=digest(a)))
    atomic_torch(a.policy.state_dict(), folder/"policy_0.pt")
    env = make("high_real_ratio", arm, seed, folder/"map")
    try:
        schedule = Episodes(env, seed, folder)
        obs, _ = schedule.reset()
        stats = [0., 0]
        for target in STEPS[1:]:
            while a.num_timesteps < target:
                obs, last, _ = a.collect_rollout(schedule, obs, ep_stats=stats, steps=min(2048,target-a.num_timesteps), timeout_is_failure=True)
                adv, returns = a.buffer.compute_gae(last,a.gamma,a.gae_lambda)
                a.update(adv,returns)
                save(folder/"progress.json",dict(steps=a.num_timesteps,episodes=schedule.count))
            atomic_torch(a.policy.state_dict(),folder/f"policy_{target}.pt")
    finally:
        env.close()

def evaluate(job):
    arm, seed = job
    torch.set_num_threads(1)
    folder = OUT/f"{arm}_{seed}"
    a = agent(seed)
    scores = {}
    for stage in ("high_real_ratio","low","none"):
        env = make(stage,arm,seed,folder/f"eval_{stage}")
        try:
            def cases(step,sim_seeds,part):
                file = folder/f"{part}_{step}_{stage}.json"
                if file.exists(): return json.loads(file.read_text())
                a.load(str(folder/f"policy_{step}.pt")); a.policy.eval()
                rows = []
                for sim,lane,route in itertools.product(sim_seeds,(0,1),m.ROUTES):
                    r = evaluate_episode(env,a,route,sim,lane)
                    r.update(cycles=env.d_cycles,actions16=env.c_commands,reward_events=env.c_journal)
                    rows.append(r)
                save(file,rows)
                return rows
            for step in STEPS:
                rows = cases(step,(870000,870100),"select")
                if stage=="high_real_ratio":
                    correct={(r["sim_seed"],r["lane_case"],r["route"]):bool(r["first"] and r["first"]["actual"]==r["first"]["expected"] and not r["first"]["wrong_end_stops"]) for r in rows}
                    pairs=sum(correct[s,l,"route_13"] and correct[s,l,"route_16"] for s,l in itertools.product((870000,870100),(0,1)))
                    scores[step]=(pairs,-sum(r["outcome"]=="collision" for r in rows),sum(r["outcome"]=="arrival" for r in rows),-step)
            if stage=="high_real_ratio":
                best=max(scores,key=scores.get)
                save(folder/"selection.json",dict(selected=best,scores=scores))
            for step in sorted({best,STEPS[-1]}):
                cases(step,(980000,980100,980200,980300),"hold")
        finally:
            env.close()

if __name__=="__main__":
    jobs=list(itertools.product(("E0","E1","E2","E3"),(11,22,33)))
    jobs.sort(key=lambda x:x[1])
    OUT.mkdir(exist_ok=True)
    save(OUT/"design.json",dict(routes=ROUTES,steps=STEPS,arms=["D1","D1+intent","D1+early","D1+intent+early"],seeds=[11,22,33]))
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(train,jobs))
        list(pool.map(evaluate,jobs))

