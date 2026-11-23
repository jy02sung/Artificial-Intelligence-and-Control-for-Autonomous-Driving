"""Train global lane-head biases against episodic return from the Z240 actor."""
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import torch
from .local_iterate import D,init,x,play
MODEL=D/"Z240.pt"
ARMS=[(0.,0.),(0.,0.65),(0.,1.3),(0.,2.1),(0.55,1.3),(0.8,2.1)]
def setup():
    global bias
    init(MODEL)
    bias=x.agent.policy.lane_head.bias.detach().clone()
def set_arm(i):
    left,right=ARMS[i]
    with torch.no_grad():x.agent.policy.lane_head.bias.copy_(bias+torch.tensor([left,0.,right]))
def trial(task):
    i,r,s=task
    set_arm(i);row=play((r,s,0));row["arm"]=i
    return row
def main():
    rng=random.Random(1354577)
    samples=[(r,rng.randrange(970000000,972000000)) for k in range(3) for r in list(x.m.ROUTES)+x.HARD*2]
    tasks=[(i,r,s) for r,s in samples for i in range(len(ARMS))]
    with ProcessPoolExecutor(5,initializer=setup) as pool:rows=list(pool.map(trial,tasks))
    def score(i):
        counts=Counter();arr=Counter();coll=Counter()
        for r in rows:
            if r["arm"]==i:
                rt=r["route"];counts[rt]+=1;arr[rt]+=r["outcome"]=="arrival";coll[rt]+=r["outcome"]=="collision"
        return sum(arr[r]/n for r,n in counts.items()),-sum(coll[r]/n for r,n in counts.items()),-i
    best=max(range(len(ARMS)),key=score)
    setup();set_arm(best)
    x.agent.save(str(D/"T.pt"));x.env.close()
    x.write_json(D/"T_search.json",dict(arms=ARMS,best=best,rows=rows))
    print("T selected",best,flush=True)
if __name__=="__main__":main()
