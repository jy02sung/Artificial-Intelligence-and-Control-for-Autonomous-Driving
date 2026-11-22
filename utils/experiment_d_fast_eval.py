"""Use two freed CPU cores for fixed final policies before checkpoint selection."""
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from utils.experiment_d_run import OUT,STAGES,SEEDS,evaluate,initialize

def publish(path,rows):
    temp=path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp.write_text(json.dumps(rows,indent=2))
    try:
        os.link(temp,path)
    except FileExistsError:
        assert json.loads(path.read_text())==rows or json.loads(path.read_text())==json.loads(temp.read_text())
    finally:
        temp.unlink(missing_ok=True)

def worker(job):
    start=time.time()
    evaluate(job,publish=publish)
    arm,seed,step,stage,part=job
    rows=json.loads((OUT/f"{arm}_{seed}"/f"{part}_{step}_{stage}.json").read_text())
    return dict(job=job,start=start,end=time.time(),episodes=len(rows),steps=sum(r["steps"] for r in rows))

def main():
    arms=json.loads((OUT/"decision.json").read_text())["arms"]
    folders=[OUT/f"{arm}_{seed}" for arm,seed in itertools.product(arms,SEEDS)]
    while not all((d/"policy_1000000.pt").exists() for d in folders):
        time.sleep(10)
    # Baseline and assisted rates use aggregate simulation steps in completed
    # files. Interleaved traffic classes and file completion times cause noise.
    baseline_start=time.time()
    time.sleep(300)
    assist_start=time.time()
    jobs=[(arm,seed,1000000,stage,part) for part,stage,arm,seed in itertools.product(
        ("select","hold"),STAGES,arms,SEEDS) if not (OUT/f"{arm}_{seed}"/f"{part}_1000000_{stage}.json").exists()]
    with ProcessPoolExecutor(2,initializer=initialize) as pool:
        with (OUT/"parallel_evaluation.jsonl").open("a") as log:
            for row in pool.map(worker,jobs):
                print(json.dumps(row),file=log,flush=True)
    def rate(lo,hi):
        files=[p for pattern in ("select_*.json","hold_*.json") for p in OUT.glob("D*_*/"+pattern) if lo<p.stat().st_mtime<=hi]
        steps=sum(r["steps"] for p in files for r in json.loads(p.read_text()))
        return dict(seconds=hi-lo,files=len(files),sim_steps=steps,steps_per_second=steps/(hi-lo))
    stop=time.time()
    (OUT/"parallel_throughput.json").write_text(json.dumps(dict(four_workers=rate(baseline_start,assist_start),six_workers=rate(assist_start,stop)),indent=2))
if __name__=="__main__":main()
