"""Local mcw2 training and evaluation utilities, deterministic evaluation with native model."""
import csv
import json
import os
import sys
import random
from collections import deque, Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from . import local_experiment_x as x

a = x.audit
base = a.MODEL
D = x.ROOT/"results"/"local_improvement"
D.mkdir(exist_ok=True)
def init(model):
    x.OUT = D
    x.initialize(model, False)
def play(task):
    route,seed,expect = task
    env, agent = x.env,x.agent
    obs,_ = env.reset(seed=seed,options={"route":route})
    hist = deque(maxlen=6)
    while True:
        act = agent.predict(obs,deterministic=True)
        cmd=int(act[1])
        if expect and env.step_count >= expect-6:
            lane=a.V.getLaneID(env.ego_id)
            hist.append(dict(lane=lane,remain=round(x.m.traci.lane.getLength(lane)-a.V.getLanePosition(env.ego_id),2),
                             speed=round(a.V.getSpeed(env.ego_id),2),action=act.tolist(),
                             gap=round(float(120*min(obs[1+4*(2-cmd)],-obs[3+4*(2-cmd)])),2)))
        obs,_,done,cut,info=env.step(act)
        if done or cut:
            result="collision" if info["collided"] else "arrival" if info["arrived"] else "timeout" if cut else "other"
            if expect: assert env.step_count==expect and result=="collision"
            return dict(route=route,seed=seed,outcome=result,steps=env.step_count,
                        contacts=a.collisions() if result=="collision" else [],hist=list(hist))
def run(model,tasks,name,workers=6):
    output=D/(name+".json")
    if output.exists():
        data=json.loads(output.read_bytes())
        assert data["sha256"]==x.digest(model) and [(r["route"],r["seed"]) for r in data["rows"]]==[(t[0],t[1]) for t in tasks]
        return data["rows"]
    with ProcessPoolExecutor(workers,initializer=init,initargs=(model,)) as pool:
        rows=list(pool.map(play,tasks))
    x.write_json(output,dict(sha256=x.digest(model),rows=rows))
    print(name,dict(Counter(r["outcome"] for r in rows)),flush=True)
    return rows
def seeds(stage):
    return [(r,s,0) for r,s in x.plan()[stage]]
if __name__=="__main__":
    if sys.argv[1]=="replay":
        def read(name):
            with (x.OUT/(name+".csv")).open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
        tasks=[(v["route"],int(v["seed"]),int(v["steps"])) for u,v in zip(read("baseline"),read("candidate")) if u["outcome"]=="arrival" and v["outcome"]=="collision"]
        run(x.X,tasks,"X_regressions")
    else:
        model,stage=sys.argv[1:3]
        run(D/(model+".pt"),seeds(stage),model+"_"+stage)
