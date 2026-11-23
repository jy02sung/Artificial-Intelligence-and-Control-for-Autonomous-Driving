"""Optimize only native output biases by episodic route completion on training seeds."""
import sys,json,random
import torch
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from .local_iterate import D,init,x,base,seeds,play
from . import local_iterate as it

def worker_init():
    global original
    init(base)
    original=[h.bias.detach().clone() for h in (x.agent.policy.lane_head,x.agent.policy.mu_head)]

def search_trial(task):
    name,keep,steer,acc,route,seed=task
    with torch.no_grad():
        x.agent.policy.lane_head.bias.copy_(original[0]+torch.tensor([steer,keep,-steer]))
        x.agent.policy.mu_head.bias.copy_(original[1]+acc)
    row=play((route,seed,0)); row["arm"]=name
    return row
def main():
    rng=random.Random(950123)
    # Different timing of lane commitments and longitudinal acceleration,
    # tuned solely on separate episodic outcomes, with no per-route parameters.
    arms=[(0,0,0),(0.8,0,0),(-0.8,0,0),(0,0,-0.22),(0.8,0,-0.22),(-0.8,0,-0.22),(0,0.55,-0.1),(0,-0.55,-0.1)]
    train=list(x.m.ROUTES)+x.HARD*2
    def block(n):
        return [(route,rng.randrange(951000000,955000000)) for _ in range(n) for route in train]
    def stage(pool,ids,rows):
        tasks=[(j,*arms[j],r,s) for r,s in rows for j in ids]
        out=list(pool.map(search_trial,tasks))
        return out
    def order(rows,ids):
        def key(j):
            rates={}
            for route in x.m.ROUTES:
                subset=[r for r in rows if r["arm"]==j and r["route"]==route]
                rates[route]=sum(r["outcome"]=="arrival" for r in subset)/len(subset)
            return (sum(rates.values()), -sum(r["outcome"]=="collision" for r in rows if r["arm"]==j),-sum(abs(v) for v in arms[j]))
        return sorted(ids,key=key,reverse=True)
    with ProcessPoolExecutor(6,initializer=worker_init) as pool:
        first=stage(pool,range(len(arms)),block(2))
        ids=order(first,range(len(arms)))[:3]
        print("W shortlist",ids,flush=True)
        final=stage(pool,ids,block(6))
    best=order(final,ids)[0]
    rows=first+final
    x.write_json(D/"W_search.json",dict(arms=arms,best=best,shortlist=ids,rows=rows))
    worker_init()
    with torch.no_grad():
        keep,steer,acc=arms[best]
        x.agent.policy.lane_head.bias.add_(torch.tensor([steer,keep,-steer]))
        x.agent.policy.mu_head.bias.add_(acc)
    x.agent.save(str(D/"W.pt"))
    x.env.close()
    print("W best",best,arms[best],flush=True)
if __name__=="__main__":main()
