"""Summarize all completed D checkpoints. Use denominators from actual approaches."""
import csv,json
from collections import Counter
from utils.experiment_d_env import OUT
from utils.experiment_d_run import score

def write(name,rows):
    if not rows:return
    p=OUT/name
    tmp=p.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    tmp.replace(p)
def run():
    summaries=[]; cycles=[]; drops=[]; lanes=[]
    for p in sorted([*OUT.glob("D*_*/*select_*.json"),*OUT.glob("D*_*/*hold_*.json")]):
        arm,seed=p.parent.name.split("_");part,step,stage=p.stem.split("_",2)
        key=dict(condition=arm,seed=seed,step=step,traffic=stage,partition=part)
        rows=json.loads(p.read_text())
        sims={r["sim_seed"] for r in rows}
        assert len(rows)==32*len(sims)
        both,_,_=score(rows)
        outcomes=Counter(r["outcome"] for r in rows)
        events=[(r,z) for r in rows for z in r["cycles"]]
        summaries.append(dict(**key,n=len(rows),joint=both,joint_n=len(sims),arrival=outcomes["arrival"],collision=outcomes["collision"],timeout=outcomes["timeout"],
            cycles=len(events),paid_cycles=sum(z["prep"]>0 for r,z in events),cycle_prep=round(sum(z["prep"] for r,z in events),4)))
        for r,z in events:
            cycles.append(dict(**key,route=r["route"],sim=r["sim_seed"],start_lane=r["lane_case"],constraint=str(z["constraint"]),departure=z["departure"]["distance"],return_distance=z["distance"],payment=z["prep"],outcome=r["outcome"]))
        for route in ("route_13","route_16"):
            for lane in (0,1):
                g=[r for r in rows if r["route"]==route and r["lane_case"]==lane]
                passed=[r for r in g if r["first"]["actual"]==r["first"]["expected"]]
                later=Counter(r["outcome"] for r in passed)
                lanes.append(dict(**key,route=route,start_lane=lane,n=len(g),passed=len(passed),arrival=later["arrival"],after_collision=later["collision"],after_timeout=later["timeout"],
                    failed_left_commands=sum(r["actions16"].get("1",0) for r in g if r not in passed) if route=="route_16" else ""))
        for road in ("844484877","206426298","86198708"):
            g=[r["drops"][road] for r in rows if road in r["drops"]]
            drops.append(dict(**key,road=road,episodes=len(rows),approached=len(g),needed=sum(x["needed"] for x in g),needed_passed=sum(x["needed"] and x["passed"] for x in g),passed=sum(x["passed"] for x in g),wrong_end=sum(x["wrong_end"] for x in g)))
    write("checkpoint_metrics.csv",summaries);write("lane_cycles.csv",cycles);write("lane_drops.csv",drops);write("first_fork_lanes.csv",lanes)
    print(len(summaries))
if __name__=="__main__":run()
