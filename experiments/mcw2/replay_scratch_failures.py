"""Replay every failed holdout and render selected local scenes from SUMO coordinates."""
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import local_scratch_driver as d

OUT = d.OUT.parent / "scratch_failure_video_20261124"
MODELS = ("dagger2", "ppo_16")

def read(name):
    return json.loads((d.OUT / (name + "_holdout.json")).read_bytes())["rows"]

def init(name):
    global V, ego
    d.init(name, False)
    V = d.m.traci.vehicle
    ego = d.env.ego_id

def car(vid):
    return dict(id=vid, xy=list(V.getPosition(vid)), edge=V.getRoadID(vid),
                lane=V.getLaneIndex(vid), pos=V.getLanePosition(vid), speed=V.getSpeed(vid),
                angle=V.getAngle(vid), length=V.getLength(vid), width=V.getWidth(vid))

def snap(step):
    av = car(ego)
    xy = av["xy"]
    av["remaining"] = d.m.traci.lane.getLength(V.getLaneID(ego)) - av["pos"]
    nav = d.env.navigation_context()
    near = [car(k) for k in V.getIDList() if k != ego and math.dist(xy, V.getPosition(k)) < 55]
    return dict(step=step, av=av, near=near, targets=nav["current_targets"], leader=V.getLeader(ego,120))

def trial(task):
    name,row,film,anchor = task
    rt,sd,end = row["route"],row["seed"],row["steps"]
    obs,_ = d.env.reset(seed=sd, options={"route":rt})
    hist=[]
    start_still=None
    while True:
        step=d.env.step_count
        a=d.agent.predict(obs)
        detail=step >= end-6 or film and (end-36<=step or anchor-36<=step<anchor)
        f=snap(step) if detail else None
        if f:
            f["accel"]=float(a[0])
        obs,_,done,cut,info=d.env.step(a)
        if info.get("speed") is None or info["speed"]>=0.5:
            start_still=None
        elif start_still is None:
            start_still=step
        if f:
            f.update(cmd=int(info["lane_change"]), after=info["lane_after"])
            hist.append(f)
        if done or cut:
            break
    out=d.outcome(info,cut)
    assert out==row["outcome"] and end==d.env.step_count,(row,out,d.env.step_count)
    contacts=[x for x in d.m.traci.simulation.getCollisions() if ego in (x.collider,x.victim)]
    last=hist[-1]; av=last["av"]
    lane=av["lane"]; edge=av["edge"]
    result=dict(model=name,route=rt,seed=sd,outcome=out,steps=end,edge=edge,lane=lane,
                pos=round(av["pos"],1),remaining=round(av["remaining"],1),speed=round(av["speed"],2),
                targets=str(last["targets"]),lead=str(last["leader"]),standstill_s=round((end-start_still)*d.env.step_length,1) if start_still is not None else 0.,
                cmd=last["cmd"],accel=round(last["accel"],3),kind=out)
    if out=="timeout":
        result["kind"]="wrong_lane_end" if last["targets"] and lane not in last["targets"] and av["remaining"]<10 else "other_timeout"
    if contacts:
        c=contacts[0]; foe=c.victim if c.collider==ego else c.collider
        result.update(contact_lane=c.lane,contact_pos=round(c.pos,1),role="collider" if c.collider==ego else "victim",contact_type=c.type)
        result["kind"]="unresolved"
        fcar=next((x for x in last["near"] if x["id"]==foe),None)
        if fcar:
            sides=[(f["av"],p) for f in hist[-3:] for p in f["near"] if p["id"]==foe]
            result.update(foe=foe,foe_edge=fcar["edge"],foe_lane=fcar["lane"],foe_speed=round(fcar["speed"],2))
            if any(v["edge"]==p["edge"] and v["lane"]!=p["lane"] for v,p in sides):
                result["kind"]="lateral_lane_convergence"
            elif all(v["edge"]==p["edge"] and v["lane"]==p["lane"] for v,p in sides):
                result["kind"]="av_follows_into_front" if fcar["pos"]>av["pos"] else "other_follows_into_av"
            elif edge!=fcar["edge"]:
                result["kind"]="different_edges"
    if film:
        scene=dict(model=name,route=rt,seed=sd,outcome=out,anchor=anchor,frames=hist,result=result)
        (OUT/f"{name}-{rt}-{sd}.json").write_text(json.dumps(scene))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    pairs=list(zip(read(MODELS[0]),read(MODELS[1])))
    chosen={}
    for rt,left,right in [(2,"collision","arrival"),(5,"arrival","collision"),(8,"arrival","collision"),(13,"timeout","arrival"),(14,"timeout","collision"),(5,"collision","collision"),(2,"collision","collision")]:
        a,b=next((a,b) for a,b in pairs if a["route"]==f"route_{rt:02}" and a["outcome"]==left and b["outcome"]==right)
        chosen[a["seed"]]=min(a["steps"],b["steps"])
    rows=[]
    for i,name in enumerate(MODELS):
        tasks=[(name,p[i],p[i]["seed"] in chosen,chosen.get(p[i]["seed"],0)) for p in pairs if p[i]["outcome"]!="arrival" or p[i]["seed"] in chosen]
        with ProcessPoolExecutor(8,initializer=init,initargs=(name,)) as pool:
            for j,result in enumerate(pool.map(trial,tasks,chunksize=1)):
                rows.append(result)
                if j%40==0: print(name,j+1,len(tasks),flush=True)
        with (OUT/"cases.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            w.writeheader();w.writerows(rows)
if __name__=="__main__":
    main()
