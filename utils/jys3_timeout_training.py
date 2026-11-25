"""Regenerate original learning trajectories, record timeouts and verify policy equality."""
from concurrent.futures import ProcessPoolExecutor
import json
from utils import jys3_pure_f as exp
BASE=exp.ROOT
OUT=BASE/"timeout_distances"/"training"
def run(job):
    arm,seed=job
    marks=exp.STEPS if seed==11 else [*exp.STEPS,65536]
    exp.ROOT=OUT
    exp.train((arm,seed,marks))
    checks={}
    label=f"{arm}_{seed}"
    for n in marks:
        a,b=[exp.j.torch.load(d/label/f"policy_{n}.pt",weights_only=True)["policy"] for d in (BASE,OUT)]
        checks[n]=all(exp.j.torch.equal(v,b[k]) for k,v in a.items())
    exp.save(OUT/label/"verification.json",checks)
    assert all(checks.values()),(job,checks)
if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(4) as pool:list(pool.map(run,[(a,11) for a in exp.ARMS]+[(a,s) for s in (22,33) for a in ("F0","F2")]))
