"""Paired short PPO reward ablation, from the same BC + DAgger student."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
from utils import jys3
from utils.jys3_ppo_600k import ROOT as PREVIOUS, BASE, paired
from utils.jys3 import torch, ScheduledEpisodes, atomic_torch

ROOT = paired.OUT / "reward_ab"
MODELS = {"control": (5., .25), "A": (10., .25), "B": (10., 1.)}
START = paired.ck("bc_dagger", 0)

def weights(arm, n):
    if n == 0 or arm == "control":
        return paired.ck("bc_dagger", n)
    return ROOT / arm / f"policy_{n}.pt"

def configure(arm):
    jys3.ARRIVAL_REWARD, jys3.PASS_REWARD = MODELS[arm]
    assert (jys3.COLLISION_REWARD, jys3.TIMEOUT_REWARD) == (-15., -5.)
    torch.set_num_threads(1)

def train(arm):
    configure(arm)
    folder = ROOT / arm
    agent = paired.policy(START)
    agent.policy.train()
    env = jys3.environment("high_real_ratio", 11, folder / "map")
    try:
        episodes = ScheduledEpisodes(env, 11, folder / "episodes.jsonl")
        obs, _ = episodes.reset()
        stats = [0., 0]
        for target in paired.STEPS[1:]:
            while agent.num_timesteps < target:
                obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats, steps=2048, timeout_is_failure=True)
                advantages, returns = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
                agent.update(advantages, returns)
            atomic_torch(dict(schema=jys3.SCHEMA, policy=agent.policy.state_dict()), weights(arm,target))
    finally:
        env.close()

def evaluate(job):
    arm,n,rep = job
    configure(arm)
    agent = paired.policy(weights(arm,n))
    cases = json.loads((PREVIOUS/"bc_dagger"/f"baseline_rep_{rep}.json").read_bytes())
    rows=[]
    with TemporaryDirectory() as tmp:
        env=jys3.environment("high_real_ratio",11,Path(tmp))
        try:
            for case in cases:
                env.ego_depart_lane=str(case["lane"])
                obs,_=env.reset(seed=case["seed"],options={"route":f"route_{case['route']:02d}"})
                assert [env.snapshot[i] for i in (0,1,7,-1)] == case["start"]
                forks=[]
                for t in range(1,1001):
                    edge=env.last_external[1]
                    obs,_,done,cut,info=env.step(agent.predict(obs,deterministic=True))
                    if info["fork_passed"]:
                        forks.append([edge,env.last_external[1],t])
                    if done or cut:
                        break
                outcome="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout" if cut else "other"
                if n==0: assert (outcome,t)==(case["outcome"],case["steps"])
                rows.append(dict(**{k:case[k] for k in ("route","seed","case","lane")},baseline=case["outcome"],outcome=outcome,steps=t,forks=forks))
        finally:
            env.close()
    paired.save(ROOT/arm/f"eval_{n}_{rep}.json",rows)
    return arm,n,rows

def main():
    ROOT.mkdir(exist_ok=False)
    for arm in MODELS: (ROOT/arm).mkdir()
    paired.save(ROOT/"protocol.json",dict(rewards=MODELS,collision=-15,timeout=-5,seed=11,steps=paired.STEPS,hp=jys3.HP,initial=str(START),sha256=hashlib.sha256(START.read_bytes()).hexdigest(),eval_base=BASE))
    jobs=[("control",n,r) for n in paired.STEPS for r in range(4)]
    with ProcessPoolExecutor(2) as trainers, ProcessPoolExecutor(8) as evaluators:
        training={trainers.submit(train,a):a for a in ("A","B")}
        futures=[evaluators.submit(evaluate,job) for job in jobs]
        for f in as_completed(training):
            f.result()
            futures.extend(evaluators.submit(evaluate,(training[f],n,r)) for n in paired.STEPS[1:] for r in range(4))
        merged={}
        for f in as_completed(futures):
            a,n,rows=f.result()
            merged.setdefault(f"{a}_{n}",[]).extend(rows)
    summary={}
    for key,rows in merged.items():
        assert len({(r["route"],r["case"]) for r in rows})==len(rows)==64
        summary[key]=dict(counts=dict(Counter(r["outcome"] for r in rows)),retained=sum(r["outcome"]==r["baseline"]=="arrival" for r in rows),rescued=sum(r["baseline"]=="collision" and r["outcome"]=="arrival" for r in rows))
    paired.save(ROOT/"summary.json",summary)

if __name__=="__main__":main()
