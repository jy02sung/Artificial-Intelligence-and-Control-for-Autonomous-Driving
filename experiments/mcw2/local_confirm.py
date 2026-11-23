"""Independent confirmatory split per frozen, dev-qualified candidate."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,random,sys
from . import local_iterate as it
from .local_iterate import D,x,play,init
def main(model):
    index=["V","R240","T","R80","U"].index(model)
    dev=json.loads((D/(model+"_dev.json")).read_bytes())
    c=Counter(r["outcome"] for r in dev["rows"])
    assert len(dev["rows"])==400 and c["arrival"]>=380 and c["collision"]<=19
    path=D/(model+".pt")
    assert x.digest(path)==dev["sha256"]
    rng=random.Random(148200+index)
    lo=980000000+index*1000000
    ids=iter(rng.sample(range(lo,lo+1000000),1600))
    tasks=[(rt,next(ids),0) for i in range(100) for rt in x.m.ROUTES]
    x.write_json(D/(model+"_confirm_plan.json"),dict(sha256=x.digest(path),tasks=tasks))
    if model=="V":
        from .local_lane_features import init as setup
    else:setup=init
    rows=[]
    with ProcessPoolExecutor(5,initializer=setup,initargs=(path,)) as pool:
        for row in pool.map(play,tasks):
            rows.append(row)
            if len(rows)%160==0: print(model,len(rows),flush=True)
    x.write_json(D/(model+"_final.json"),dict(sha256=x.digest(path),rows=rows))
    print(model,dict(Counter(r["outcome"] for r in rows)),flush=True)
if __name__=="__main__": main(sys.argv[1])
