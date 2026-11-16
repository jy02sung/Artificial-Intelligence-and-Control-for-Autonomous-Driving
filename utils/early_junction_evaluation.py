"""Evaluate immutable 50k snapshots alongside ongoing training, in separate outputs."""
import concurrent.futures as cf
import time
from utils import run_junction_conflict_trials as j
s=j.s;t=s.t

def one(spec):
    condition,step=spec
    s.a.b.torch.set_num_threads(1)
    source=s.OUT/'runs'/condition
    checkpoint=source/f'policy-{step}.pt'
    while not checkpoint.exists():time.sleep(2)
    dest=s.OUT/'early-evaluation'/str(step)/condition
    dest.mkdir(parents=True,exist_ok=True)
    t.save(dest/'manifest.json',t.read(source/'manifest.json'))
    rows=s.evaluate(condition,checkpoint,dest)
    count={k:sum(r['outcome']==k for r in rows if r['traffic']=='reference') for k in ('arrival','collision','timeout')}
    t.save(dest/'complete.json',dict(condition=condition,checkpoint=checkpoint.name,counts=count))
    return count

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=100352);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for x in pool.map(one,[(c,args.step) for c in ('reward','observation','control')]):print(x,flush=True)
