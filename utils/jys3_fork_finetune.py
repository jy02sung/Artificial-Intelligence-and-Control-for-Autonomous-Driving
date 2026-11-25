"""Matched fork reward fine-tuning from two previously selected policies."""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from utils import jys3_terminal_ratio as ratio

exp = ratio.exp
ROOT = ratio.ROOT / "fork_finetune"
SOURCES = {"collision5": 550000, "both": 600000}
FORK = {"original": .25, "increased": 1.}
SCREEN = 48110000
HOLD = 49110000
STEPS = (0, 8192, 16384, 32768)


def setup(arm):
    source, variant = arm.split("-")
    ratio.setup(source)
    exp.ROOT = ROOT
    exp.j.PASS_REWARD = FORK[variant]


def train(job):
    arm, seed = job
    setup(arm)
    source = arm.split("-")[0]
    ck = ratio.ROOT/"long_600k_4"/f"{source}_{seed}"/f"policy_{SOURCES[source]}.pt"
    exp.train((arm, seed, STEPS), initial_weights=ck)


def evaluate(job):
    setup(job[0])
    exp.evaluate(job)


def read(path):
    return json.loads(path.read_bytes())


def rows(arm, seed, step, base):
    folder = ROOT / f"{arm}_{seed}"
    return [r for rep in range(4) for r in read(folder/f"eval_{base}_{step}_{rep}.json")]


def main():
    while not (ratio.ROOT/"progress_only_7_10"/"README.md").exists():
        time.sleep(20)
    ROOT.mkdir(exist_ok=False)
    exp.save(ROOT/"protocol.json",dict(sources=SOURCES, fork=FORK, seeds=ratio.SEEDS,
             extra_steps=STEPS,screen=SCREEN,hold=HOLD,optimizer="fresh for both arms"))
    arms = [f"{source}-{variant}" for source in SOURCES for variant in FORK]
    jobs = [(arm,s) for s in ratio.SEEDS for arm in arms]
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(train,jobs))
    for source in SOURCES:
        for seed in ratio.SEEDS:
            hashes = {read(ROOT/f"{source}-{v}_{seed}"/"preflight.json")["initial_hash"] for v in FORK}
            assert len(hashes)==1
    evaluations = [(a,s,n,r,base) for a,s in jobs for base,marks in [(SCREEN,STEPS),(HOLD,(0,32768))]
                   for n in marks for r in range(4)]
    with ProcessPoolExecutor(10) as pool:
        list(pool.map(evaluate,evaluations))
    summary = {}
    for source in SOURCES:
        for base,marks in [(SCREEN,STEPS),(HOLD,(0,32768))]:
            for seed in ratio.SEEDS:
                for n in marks:
                    pair = [rows(f"{source}-{v}",seed,n,base) for v in FORK]
                    a,b = pair
                    assert len(a)==len(b)==64 and all(all(x[k]==y[k] for k in ("route","case","start","traffic_seed")) for x,y in zip(a,b))
                    if n==0:
                        assert all(x["outcome"]==y["outcome"] for x,y in zip(a,b))
                    summary[f"{source}_{base}_{seed}_{n}"] = dict(
                        counts=[Counter(r["outcome"] for r in group) for group in pair],
                        transitions=Counter(x["outcome"]+"->"+y["outcome"] for x,y in zip(a,b)),
                        exit13=[sum(r["route"]==13 and "169728835" in r["forks"] for r in g) for g in pair],
                        main16=[sum(r["route"]==16 and r["start"][1]==0 and "86198725" in r["forks"] for r in g) for g in pair])
    exp.save(ROOT/"summary.json",summary)


if __name__ == "__main__":
    main()
