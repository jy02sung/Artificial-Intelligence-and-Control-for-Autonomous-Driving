"""Reproduce the previous run, continue PPO, and evaluate matched checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import json
import time
from utils import jys3_terminal_ratio as ratio
from utils.jys3_pure_f_report import aggregate

e=ratio.exp
ROOT=ratio.ROOT/"long_600k_4"
ARMS=tuple(ratio.ARMS)
SEEDS=ratio.SEEDS
MARKS=tuple(range(50000,600001,50000))
SCREEN=ratio.FRESH
HOLD=43110000
STRICT_REFERENCE=True

def reference(a,s):
    return (ratio.REF if a=="F1" else ratio.ROOT)/f"{a}_{s}"/"policy_65536.pt"

def train(job):
    a,s=job
    ratio.setup(a)
    p=ROOT/f"{a}_{s}"
    p.mkdir()
    model=e.agent(s)
    env=ratio.Env(a,p/"map",s)
    try:
        env.training_16=[]
        env.training_log=p/"training_outcomes.jsonl"
        sched=e.j.ScheduledEpisodes(env,s,p/"episodes.jsonl")
        obs,_=sched.reset()
        stats=[0.,0]
        while model.num_timesteps<600000:
            old=model.num_timesteps
            stop=65536 if old<65536 else next(n for n in MARKS if n>old)
            batch=min(2048,stop-old)
            obs,last,_=model.collect_rollout(sched,obs,ep_stats=stats,steps=batch,timeout_is_failure=True)
            # No update occurs inside this rollout. Save the policy active at step 50,000.
            if old<50000<model.num_timesteps:
                e.j.atomic_torch(dict(policy=model.policy.state_dict()),p/"policy_50000.pt")
            adv,ret=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
            model.update(adv,ret)
            n=model.num_timesteps
            if n==65536:
                ref=e.j.torch.load(reference(a,s),weights_only=True)["policy"]
                matched=all(e.j.torch.equal(v,ref[k]) for k,v in model.policy.state_dict().items())
                e.save(p/"verified.json",dict(step=n,matched=matched))
                if STRICT_REFERENCE:
                    assert matched,(a,s,n)
            if n in MARKS or n==65536:
                e.j.atomic_torch(dict(policy=model.policy.state_dict()),p/f"policy_{n}.pt")
            e.save(p/"progress.json",dict(step=n,time=time.time()))
    finally:env.close()

def evaluate(job):
    a,s,n,r,base=job
    ratio.setup(a)
    e.ROOT=ROOT
    e.evaluate(job)

def results(a,s,n,base):
    p=ROOT/f"{a}_{s}"
    rs=[r for i in range(4) for r in json.loads((p/f"eval_{base}_{n}_{i}.json").read_bytes())]
    keys=[(r["route"],r["case"],r["traffic_seed"],r["start"]) for r in rs]
    assert len(set((r["route"],r["case"]) for r in rs))==len(rs)==64
    return keys,aggregate(rs)

def main():
    ROOT.mkdir(exist_ok=False)
    jobs=[(a,s) for s in SEEDS for a in ARMS]
    e.save(ROOT/"protocol.json",dict(arms={a:ratio.params(a) for a in ARMS},seeds=SEEDS,marks=MARKS,screen=SCREEN,hold=HOLD,
        timeout=-5,cap=1000,progress_bonus=2.,train_workers=6,eval_workers=4,started=time.time()))
    seen=set();pending={};partial={};ref=None
    with ProcessPoolExecutor(6) as tp,ProcessPoolExecutor(4) as ep:
        trains=[tp.submit(train,j) for j in jobs]
        while True:
            for f in trains:
                if f.done():f.result()
            for a,s in jobs:
                for n in MARKS:
                    key=(a,s,n)
                    if key in seen or not (ROOT/f"{a}_{s}"/f"policy_{n}.pt").exists():continue
                    seen.add(key)
                    for r in range(4):pending[ep.submit(evaluate,(a,s,n,r,SCREEN))]=key
            for f in list(pending):
                if f.done():
                    key=pending.pop(f);f.result()
                    a,s,n=key
                    rowkey=f"{a}_{s}_{n}"
                    if rowkey not in partial and all((ROOT/f"{a}_{s}"/f"eval_{SCREEN}_{n}_{i}.json").exists() for i in range(4)):
                        keys,partial[rowkey]=results(a,s,n,SCREEN)
                        if ref is None:ref=keys
                        assert ref==keys
                        e.save(ROOT/"screen.json",partial)
            if len(partial)==len(jobs)*len(MARKS) and not pending and all(f.done() for f in trains):break
            time.sleep(3)
    best={}
    for a in ARMS:
        def rank(n):
            c=Counter()
            for s in SEEDS:c.update(partial[f"{a}_{s}_{n}"]["outcomes"])
            return c["arrival"],-c["collision"],-n
        best[a]=max(MARKS,key=rank)
    e.save(ROOT/"selection.json",best)
    final={};ref=None
    with ProcessPoolExecutor(10) as ep:
        targets=[(a,s,n) for a,s in jobs for n in sorted({65536,best[a],600000})]
        list(ep.map(evaluate,[(a,s,n,r,HOLD) for a,s,n in targets for r in range(4)]))
        for a,s,n in targets:
            keys,final[f"{a}_{s}_{n}"]=results(a,s,n,HOLD)
            if ref is None:ref=keys
            assert ref==keys
    e.save(ROOT/"final.json",dict(best=best,hold=final))

if __name__=="__main__":main()
