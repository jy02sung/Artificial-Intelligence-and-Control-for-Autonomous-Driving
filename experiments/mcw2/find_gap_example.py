"""Find and replay a verified longitudinal gap-opening lane change in a completed PPO drive."""
import json
import math
from collections import deque
from pathlib import Path

from . import long_scratch_ppo as curve
from . import replay_scratch_failures as replay
from . import local_scratch_driver as d

MODEL="150000"
OUT=curve.OUT/"gap-example"
def measure(av,other):
    return (other["pos"]-other["length"]-av["pos"],av["pos"]-av["length"]-other["pos"])

def candidates(history, stronger=False):
    last=history[-1]
    av=last["av"]
    lane=last["after"]
    if lane is None or lane==av["lane"] or last["cmd"]!=lane-av["lane"] or last["next_edge"]!=av["edge"]:return
    if lane not in last["targets"] or av["lane"] in last["targets"]:return
    near=[p for p in last["near"] if p["edge"]==av["edge"] and p["lane"]==lane]
    for other in near:
        gaps=measure(av,other)
        for k,sign in [(0,-1),(1,1)]:
            endgap=gaps[k]
            if endgap<max(4.,2+.15*max(av["speed"],other["speed"])) or endgap>30:continue
            for i,first in enumerate(history[:-5]):
                seq=history[i:]
                if any(f["av"]["lane"]!=av["lane"] or f["av"]["edge"]!=av["edge"] for f in seq):continue
                peers=[next((p for p in f["near"] if p["id"]==other["id"] and p["edge"]==av["edge"] and p["lane"]==lane),None) for f in seq]
                if not all(peers):continue
                old=measure(first["av"],peers[0])
                if max(old)>0:continue
                if sign*(av["speed"]-first["av"]["speed"])<1:continue
                if sum(sign*f["accel"]>.1 for f in seq[:-1])<3:continue
                if any(f["cmd"]!=0 for f in seq[:-1]):continue
                # Reject a different vehicle already overlapping in the receiving lane.
                if any(max(measure(av,p))<2.0 for p in near):continue
                if stronger:
                    dv=av["speed"]-first["av"]["speed"]
                    if abs(dv)<2 or abs(peers[0]["speed"]-first["av"]["speed"])>3:continue
                    # Only screen candidates here. Verify with a second SUMO replay.
                    if abs(dv)*.5*(last["step"]-first["step"])*.2<2:continue
                yield dict(start=first["step"],change=last["step"],foe=other["id"],side=k,
                           gap_start=old[k],gap_end=endgap,v_start=first["av"]["speed"],v_end=av["speed"],
                           peer_start=peers[0]["speed"],peer_end=other["speed"],lane=lane,edge=av["edge"],
                           count_commands=sum(sign*f["accel"]>.1 for f in seq[:-1]))
                break

def run(row,chosen=None,neutral=False,stronger=False):
    env=d.env
    v=d.m.traci.vehicle
    obs,_=env.reset(seed=row["seed"],options={"route":row["route"]})
    h=deque(maxlen=36)
    result=None
    frames=[]
    while True:
        t=env.step_count
        a=d.agent.predict(obs)
        active=chosen is None or chosen["start"]-5<=t<=chosen["change"]+12
        if active:
            f=replay.snap(t)
            f["accel"]=float(a[0])
        if neutral and chosen["start"]<=t<chosen["change"]:
            a[0]=0.
            a[1]=0.  # Match the factual lateral hold until the sampled merge instant.
        obs,_,done,cut,info=env.step(a)
        if active:
            f.update(cmd=int(info["lane_change"]),after=info["lane_after"],next_edge=v.getRoadID(replay.ego) if replay.ego in v.getIDList() else "")
            if chosen:frames.append(f)
            else:
                h.append(f)
                if not result and len(h)>6:result=next(candidates(list(h),stronger),None)
        if done or cut:break
        if neutral and t>=chosen["change"]:break
    if not neutral:assert d.outcome(info,cut)==row["outcome"] and env.step_count==row["steps"]
    return result,frames

def main():
    OUT.mkdir(exist_ok=True)
    curve.init_eval(MODEL)
    replay.V=d.m.traci.vehicle
    replay.ego=d.env.ego_id
    rows=json.loads((curve.OUT/(MODEL+"_eval.json")).read_bytes())["rows"]
    preferred=("route_02","route_05","route_13","route_14","route_08","route_10")
    for row in rows:
        if row["outcome"]!="arrival" or row["route"] not in preferred:continue
        found,_=run(row)
        if found:
            _,frames=run(row,found)
            _,control=run(row,found,True)
            (OUT/"example.json").write_text(json.dumps(dict(model=MODEL,row=row,event=found,frames=frames,control=control)))
            print(row,found,flush=True)
            break
    else: print("No verified event",flush=True)
    d.env.close()
if __name__=="__main__":main()
