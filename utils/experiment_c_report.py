"""Refresh experiment C checkpoint/holdout metrics from immutable episode files."""
import csv
import json
from collections import Counter
from utils.experiment_c_env import OUT,DROPS
from utils.experiment_b_report import measure
from utils.experiment_b_eval import score

def write(name,rows):
    if not rows:return
    p=OUT/name
    tmp=p.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)

def run():
    pairs=json.loads((OUT/"design.json").read_text())["pairs"]
    exits={p["exit"] for p in pairs}
    mains={p["main"] for p in pairs}
    summary,lanes,drops,cycles,scenes=[],[],[],[],[]
    for file in sorted([*OUT.glob("C*_*/*hold_*.json"),*OUT.glob("C*_*/*select_*.json")]):
        part,step,stage=file.stem.split("_",2)
        arm,seed=file.parent.name.split("_")
        key=dict(condition=arm,seed=seed,partition=part,step=step,traffic=stage)
        rows=json.loads(file.read_text())
        assert len(rows)==len({(r["route"],r["sim_seed"]) for r in rows})==(128 if part=="hold" else 32)
        p,n,*_=score(rows,pairs)
        pp,npair,*_=score(rows,[x for x in pairs if (x["exit"],x["main"])==("route_13","route_16")])
        summary.append(dict(**key,pair_success=p,pairs=n,pair_pct=100*p/n,pair13_16=pp,pair13_16_n=npair,**measure(rows,exits,mains)))
        for route in ("route_13","route_16"):
            for lane in (0,1):
                g=[r for r in rows if r["route"]==route and r["lane_case"]==lane]
                lanes.append(dict(**key,route=route,start_lane=lane,n=len(g),pass_count=sum(r["first"]["actual"]==r["first"]["expected"] for r in g),
                    arrival=sum(r["outcome"]=="arrival" for r in g),collision=sum(r["outcome"]=="collision" for r in g),
                    timeout=sum(r["outcome"]=="timeout" for r in g),left=sum(r["actions16"].get("1",0) for r in g) if route=="route_16" else None,
                    actual_changes=sum(r["actions16"].get("actual",0) for r in g) if route=="route_16" else None))
        for road in sorted(DROPS):
            g=[r for r in rows if road in r["drops"]]
            drops.append(dict(**key,road=road,total=len(rows),approached=len(g),needed=sum(r["drops"][road]["needed"] for r in g),
                passed=sum(r["drops"][road]["passed"] for r in g),wrong_end=sum(r["drops"][road]["wrong_end"] for r in g),
                pre_collision=sum(not r["drops"][road]["passed"] and r["outcome"]=="collision" for r in g),
                pre_timeout=sum(not r["drops"][road]["passed"] and r["outcome"]=="timeout" for r in g)))
        for r in rows:
            for f in r["forks"].values():
                if not f.get("initially_ready") or not f["wrong_entries"]:continue
                returned=[x for x in r["reward_events"] if x["edge"]==f["edge"] and x["lane"] not in f["targets"] and x["after"] in f["targets"]]
                cycles.append(dict(**key,route=r["route"],sim_seed=r["sim_seed"],edge=f["edge"],wrong_entries=f["wrong_entries"],returns=f["returns"],return_prep=sum(x["prep"] for x in returned)))
            if part=="hold" and r["outcome"]!="arrival":
                scenes.append(dict(**key,route=r["route"],sim_seed=r["sim_seed"],outcome=r["outcome"],last=json.dumps(r["last"]),commands16=json.dumps(r["actions16"]),forks=json.dumps(r["forks"])))
    write("checkpoint_metrics.csv",summary)
    write("first_fork_lanes.csv",lanes)
    write("lane_drops.csv",drops)
    write("ready_lane_excursions.csv",cycles)
    write("holdout_failures.csv",scenes)
    print(len(summary),"evaluations aggregated")

if __name__=="__main__":run()
