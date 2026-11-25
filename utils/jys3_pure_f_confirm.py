"""Confirm one pilot arm and the baseline on independent traffic and training seeds."""
import json
import sys
from collections import Counter
from utils.jys3_pure_f import ROOT,run,save,STEPS
from utils.jys3_pure_f_report import read,aggregate

if __name__=="__main__":
    arm=sys.argv[1]
    assert arm in ("F1","F2","F3")
    arms=["F0",arm]; seeds=[22,33]; endpoints=[32768,65536]; base=37110000
    save(ROOT/"confirmation_protocol.json",dict(arms=arms,seeds=seeds,endpoints=endpoints,traffic_base=base))
    run(arms,seeds,[*STEPS,65536],base,endpoints)
    results={}
    for seed in seeds:
        hs=[json.loads((ROOT/f"{a}_{seed}"/"preflight.json").read_bytes())["initial_hash"] for a in arms]
        assert len(set(hs))==1
        ref=None
        for a in arms:
            for n in endpoints:
                rows=read(a,seed,n,base)
                keys=[(r["case"],r["route"],r["traffic_seed"],r["start"]) for r in rows]
                if ref is None: ref=keys
                assert keys==ref and len(rows)==64
                results[f"{a}_{seed}_{n}"]=aggregate(rows)
    save(ROOT/"confirmation_summary.json",results)
