"""Replay E training with passive route-16 event logging; verify archived models."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from utils import experiment_e as e

ORIGINAL=e.OUT
AUDIT=ORIGINAL/"training_replay"
base_make=e.make
base_save=e.atomic_torch

def replay(job):
    arm,seed=job
    name=f"{arm}_{seed}"
    folder=AUDIT/name
    original=ORIGINAL/name
    def make(stage,arm,seed,map_path):
        env=base_make(stage,arm,seed,map_path)
        reset,step=env.reset,env.step
        episode=-1
        tick=0
        active=False
        events=open(folder/"events.jsonl","w",buffering=1)
        def log(*args):
            print(json.dumps(args),file=events)
        def start(*args,**kw):
            nonlocal episode,active
            obs,info=reset(*args,**kw)
            episode+=1
            active=env.episode_route=="route_16"
            if active:log("start",episode,tick,env.snapshot[1])
            return obs,info
        def advance(action):
            nonlocal tick
            before=env.snapshot
            obs,r,done,cut,info=step(action)
            tick+=1
            if active:
                after=env.snapshot
                if before[0]=="86198728":
                    cmd=info["lane_change"]
                    if cmd==1 or (after and after[0]==before[0] and before[1]!=after[1]):
                        log("cmd",episode,tick,cmd,before[1],after[1] if after and after[0]==before[0] else None,info["c_prep"],before[8])
                if after and after[0]=="86198725" and before[0]!=after[0]:
                    log("pass",episode,tick,info["c_pass"],info["early_pass"])
                if done or cut:log("end",episode,tick,info["collided"],info["arrived"],cut)
            return obs,r,done,cut,info
        env.reset,env.step=start,advance
        return env
    def verify(obj,path):
        stored=torch.load(original/path.name,map_location="cpu",weights_only=True)
        assert obj.keys()==stored.keys() and all(torch.equal(v,stored[k]) for k,v in obj.items()),(name,path.name)
        if (folder/"episodes.jsonl").exists():
            observed=(folder/"episodes.jsonl").read_bytes().splitlines()
            truth=(original/"episodes.jsonl").read_bytes().splitlines()
            assert observed==truth[:len(observed)],name
            if path.name=="policy_200000.pt":assert len(observed)==len(truth)
        (folder/(path.stem+".verified")).touch()
    e.OUT=AUDIT
    e.make=make
    e.atomic_torch=verify
    e.train(job)
if __name__=="__main__":
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(replay,[(a,s) for s in (11,22,33) for a in ("E0","E1","E2","E3")]))
