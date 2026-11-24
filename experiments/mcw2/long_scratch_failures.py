"""Apply the same contact classifier to failed 600k evaluation cases."""
import csv
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from . import replay_scratch_failures as replay
from .long_scratch_ppo import OUT, init_eval

def initialize():
    init_eval("600000")
    replay.V=replay.d.m.traci.vehicle
    replay.ego=replay.d.env.ego_id

def main():
    result=OUT/"600000_eval.json"
    while not result.exists():
        time.sleep(10)
    rows=json.loads(result.read_bytes())["rows"]
    jobs=[("ppo_600000",row,False,0) for row in rows if row["outcome"]!="arrival"]
    with ProcessPoolExecutor(8,initializer=initialize) as pool:
        details=list(pool.map(replay.trial,jobs,chunksize=2))
    with (OUT/"600000_failures.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in details))) or ["kind"])
        writer.writeheader();writer.writerows(details)
    print(dict(Counter(r["kind"] for r in details)),flush=True)
if __name__=="__main__": main()
