"""Select using screening only; compare frozen BC and PPO on the same 400 runs."""
import json,sys,time,os
from collections import Counter
from . import local_bc_ppo as e
from .local_bc_fit import ppo
from .local_bc_ppo import D,x,evaluate
def summary(model,stage):
    d=json.loads((D/(model+"_"+stage+".json")).read_bytes())
    assert d["sha256"]==x.digest(D/(model+".pt"))
    return Counter(r["outcome"] for r in d["rows"])
def score(c):return c["arrival"],-c["collision"]
def choose(models,stage):
    return max(models,key=lambda name:score(summary(name,stage)))
if __name__=="__main__":
    pool=os.environ.get("BC_GROUP","bc8,bc22").split(",")
    while not all((D/(m+"_screen.json")).exists() for m in pool):time.sleep(3)
    parent=choose(pool,"screen");print("BC selected",parent,flush=True)
    if sys.argv[1]=="baseline":
        evaluate(parent,"dev")
    elif sys.argv[1]=="train":
        seed,lr,n,safe=int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),bool(int(sys.argv[5]))
        ppo(parent,n,seed,lr,safe)
    else:
        seed,n=int(sys.argv[2]),int(sys.argv[3])
        steps=sorted({i for i in [4,12,24,n] if i<=n});names=[f"{parent}_p{seed}_{i}" for i in steps]
        for name in names:
            while not (D/(name+"_train.json")).exists():time.sleep(3)
            evaluate(name,"screen")
        best=choose(names,"screen");c=evaluate(best,"dev")
        x.write_json(D/f"selection_{seed}.json",{"parent":parent,"candidate":best,"development":dict(c)})
