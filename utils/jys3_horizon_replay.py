"""Paired frozen-policy replay changing only the episode time limit."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from utils.jys3_ppo_600k import ROOT, ARMS, paired, environment, torch

def run(job):
    arm, rep = job
    torch.set_num_threads(1)
    folder = ROOT / arm
    model = paired.policy(folder / "policy_600000.pt")
    original = json.loads((folder / f"final_rep_{rep}.json").read_bytes())
    output = []
    with TemporaryDirectory() as tmp:
        env = environment("high_real_ratio",11,Path(tmp))
        env.max_steps = 3000
        try:
            for row in original:
                env.ego_depart_lane = str(row["lane"])
                obs,_ = env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"})
                assert [env.snapshot[i] for i in (0,1,7,-1)] == row["start"]
                marks = {}
                hist = []
                for t in range(1,3001):
                    edge,lane,path,idx,_,_,_,speed,rem = env.snapshot
                    targets = env.links.get(edge,{}).get(path[idx+1],()) if idx+1<len(path) else ()
                    act = model.predict(obs,deterministic=True)
                    obs,_,done,cut,info = env.step(act)
                    hist.append((speed,float(act[0]),int(act[1]),info["gap"]))
                    if done or cut or t in (1000,2000):
                        last = hist[-150:]
                        marks[t] = dict(edge=edge,lane=lane,remaining=round(rem,2),idx=idx,targets=list(targets),speed=round(speed,2),
                            slow=sum(v<0.5 for v,_,_,_ in last),brake=sum(a<0 for _,a,_,_ in last),mean_acc_raw=round(sum(a for _,a,_,_ in last)/len(last),3),
                            near=sum(g is not None and g<=25 for _,_,_,g in last),cmd=dict(Counter(c for _,_,c,_ in last)))
                    if done or cut: break
                outcome = "arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout" if cut else "other"
                assert (t>=1000 and 1000 in marks) if row["outcome"]=="timeout" else (outcome,t)==(row["outcome"],row["steps"])
                output.append(dict(arm=arm,case=rep,route=row["route"],original=row["outcome"],outcome=outcome,steps=t,marks=marks))
        finally: env.close()
    return output
if __name__=="__main__":
    with ProcessPoolExecutor(8) as pool:
        groups = list(pool.map(run,[(a,r) for r in range(4) for a in ARMS]))
    paired.save(ROOT/"horizon_replay.json",[r for group in groups for r in group])
