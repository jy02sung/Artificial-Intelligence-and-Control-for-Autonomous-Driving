"""Paired -30 collision penalty experiment from the same scratch DAgger2 checkpoint."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from . import local_scratch_driver as d

BASE = d.OUT
OUT = BASE.parent / "scratch_collision30_20261124"
original_init = d.init

def initialize(model, training):
    d.OUT = OUT
    original_init(model, training)
    assert d.env.rw["collision_penalty"] == 15
    d.env.rw["collision_penalty"] = 30

def evaluate(name, schedule, partition):
    with ProcessPoolExecutor(d.WORKERS, initializer=initialize, initargs=(name,False)) as pool:
        rows = list(pool.map(d.episode, [(r, s, -1, None) for r, s in schedule], chunksize=2))
    assert [(row["route"],row["seed"]) for row in rows] == schedule
    result = dict(model=name, sha256=d.digest(d.weights(name)), rows=rows,
                  counts=dict(Counter(row["outcome"] for row in rows)))
    d.save(f"{name}_{partition}.json",result)
    return result

def run():
    OUT.mkdir(exist_ok=True)
    d.OUT = OUT
    shutil.copyfile(BASE / "dagger2.pt", d.weights("dagger2"))
    parent_sha = d.digest(d.weights("dagger2"))
    reference = json.loads((BASE / "ppo_16_train.json").read_bytes())
    assert parent_sha == reference["parent_sha256"]
    d.save("protocol.json",dict(baseline=str(BASE.name),parent_sha256=parent_sha,
        before=15,after=30,steps=32768,seed=d.SEED,final_update=16,
        endpoint="final update fixed before training; same screen and holdout cases"))
    if not d.weights("ppo_16").exists():
        d.init = initialize
        try:
            d.ppo("dagger2")
        finally:
            d.init = original_init
    for n in [4,8,16]:
        name=f"ppo_{n}"
        evaluate(name,d.cases(199000000,5),"screen")
    after=evaluate("ppo_16",d.cases(200000000,100),"holdout")
    comparisons={}
    for model in ["dagger2","ppo_16"]:
        before=json.loads((BASE/f"{model}_holdout.json").read_bytes())
        assert [(x["route"],x["seed"]) for x in before["rows"]]==[(x["route"],x["seed"]) for x in after["rows"]]
        transitions=Counter((a["outcome"],b["outcome"]) for a,b in zip(before["rows"],after["rows"]))
        comparisons[model]=dict(before=before["counts"],after=after["counts"],
                                diff=d.paired(before,after),
                                transitions={f"{a}->{b}":v for (a,b),v in transitions.items()})
    d.save("comparison.json",comparisons)
    print(json.dumps(comparisons),flush=True)

if __name__=="__main__":
    run()
