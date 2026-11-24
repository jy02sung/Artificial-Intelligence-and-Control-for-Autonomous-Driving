"""Identify route-required lane changes preceded by measured longitudinal adjustment."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from . import find_gap_example as g
from . import replay_scratch_failures as replay

OUT=g.curve.OUT/"long-gap-replays"
def nearby_gap(history):
    # A gap can be too narrow without the two vehicles actually overlapping.
    last=history[-1];av=last["av"];lane=last["after"]
    if lane not in last["targets"] or lane==av["lane"] or last["next_edge"]!=av["edge"]:return
    for p in last["near"]:
        if p["lane"]!=lane:continue
        for side,sign in ((1,1),(0,-1)):
            gap=g.measure(av,p)[side]
            if not 5<gap<=38:continue
            for i,f in enumerate(history[:-8]):
                window=history[i:];car=f["av"]
                if any(x["av"]["edge"]!=av["edge"] or x["av"]["lane"]!=av["lane"] for x in window):continue
                if any(x["cmd"] for x in window[:-1]):continue
                peers=[next((v for v in x["near"] if v["id"]==p["id"] and v["lane"]==lane),None) for x in window]
                if not all(peers):continue
                old=g.measure(car,peers[0])[side]
                if not -15<old<5 or gap-old<4:continue
                if sign*(av["speed"]-car["speed"])<0.8:continue
                n=sum(sign*x["accel"]>.08 for x in window[:-1])
                if n<4:continue
                yield dict(start=f["step"],change=last["step"],foe=p["id"],side=side,gap_start=old,gap_end=gap,
                           v_start=car["speed"],v_end=av["speed"],peer_start=peers[0]["speed"],peer_end=p["speed"],
                           lane=lane,edge=av["edge"],count_commands=n)
                break

def worker(model,alternate=False):
    g.curve.init_eval(model)
    env=g.d.env
    v=g.d.m.traci.vehicle
    replay.V=v
    replay.ego=env.ego_id
    rows=json.loads((g.curve.OUT/f"{model}_eval.json").read_bytes())["rows"]
    routes={"200000":["route_05","route_03","route_13"],"250000":["route_08","route_10","route_04"],"300000":["route_14","route_16","route_07"]}[model]
    if alternate:
        routes=["route_04","route_08","route_10","route_12"] if model=="300000" else ["route_04","route_05","route_13","route_14","route_10"]
        rows=rows[16*16:]+rows[:16*16]
    if alternate=="limited":
        routes=["route_11","route_02","route_08","route_10"]
        rows.sort(key=lambda r:r["seed"] not in (200493722,200763654,200263277))
    for row in rows:
        if (OUT/f"{model}.json").exists():return
        if row["route"] not in routes or row["outcome"]!="arrival":continue
        obs,_=env.reset(seed=row["seed"],options={"route":row["route"]})
        history=deque(maxlen=40); events=[]
        while True:
            step=env.step_count
            act=g.d.agent.predict(obs)
            ego=replay.ego
            edge=v.getRoadID(ego);lane=v.getLaneIndex(ego)
            nav=env.navigation_context()
            targets=nav["current_targets"]
            if targets and lane not in targets and not edge.startswith(":"):
                av=replay.car(ego)
                near=[]
                for idx in [lane-1,lane+1]:
                    if idx not in targets:continue
                    for vid in g.d.m.traci.lane.getLastStepVehicleIDs(f"{edge}_{idx}"):
                        if abs(v.getLanePosition(vid)-av["pos"])<45:near.append(replay.car(vid))
                f=dict(step=step,av=av,near=near,targets=targets,accel=float(act[0]))
            else:
                f=None;history.clear()
            obs,_,done,cut,info=env.step(act)
            if f:
                f.update(cmd=int(info["lane_change"]),after=info["lane_after"],next_edge=v.getRoadID(ego) if ego in v.getIDList() else "")
                history.append(f)
                if f["after"]!=lane:
                    for event in (nearby_gap(list(history)) if alternate else g.candidates(list(history))):
                        if event["change"]>=50 and event["change"]<=row["steps"]-41:
                            events.append(event)
            if done or cut:break
        assert g.d.outcome(info,cut)==row["outcome"] and env.step_count==row["steps"]
        for event in events:
            scene=capture(model,row,event)
            if scene:
                path=OUT/(f"{model}-extra.json" if alternate=="limited" else f"{model}-alt.json" if alternate else f"{model}.json")
                path.write_text(json.dumps(scene))
                print(model,row["route"],event,flush=True)
                env.close()
                return
    env.close()

def capture(model,row,e):
    env=g.d.env
    obs,_=env.reset(seed=row["seed"],options={"route":row["route"]})
    frames=[]
    direction=None
    while True:
        t=env.step_count; action=g.d.agent.predict(obs)
        if e["change"]-65<=t<=e["change"]+55:
            f=replay.snap(t); f["accel"]=float(action[0]); f["obs"]=obs.tolist()
            if t==e["start"]:direction=e["lane"]-f["av"]["lane"]
            if e["start"]<=t<=e["change"]:
                k=list(env.obs_offsets).index(direction)
                alt=obs.copy();alt[1+4*k:5+4*k]=[1,1,-1,-1]
                f["masked_action"]=g.d.agent.predict(alt).tolist()
            frames.append(f)
        obs,_,done,cut,info=env.step(action)
        if frames and frames[-1]["step"]==t:
            frames[-1].update(cmd=int(info["lane_change"]),after=info["lane_after"])
        if done or cut:break
    assert g.d.outcome(info,cut)=="arrival" and env.step_count==row["steps"]
    after=[f for f in frames if e["change"]<f["step"]<=e["change"]+10]
    if any(f["av"]["edge"]==e["edge"] and f["av"]["lane"]!=e["lane"] for f in after):return
    base=next(f for f in frames if f["step"]==e["start"])
    c=next(f for f in frames if f["step"]==e["change"])
    if int(c["masked_action"][1])==c["cmd"] and max(abs(x["masked_action"][0]-x["accel"]) for x in frames if "masked_action" in x)<.04:return
    return dict(model=model,row=row,event=e,direction=direction,frames=frames)
if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:
        worker(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else True)
        sys.exit(0)
    with ProcessPoolExecutor(3) as pool:list(pool.map(worker,["200000","250000","300000"]))
