"""Single actor with trainable lane-head coefficients for gap and lane continuity.
The inputs are derived from the existing normalized 47-D observation. No action filter.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import random
import sys
import torch
from utils.networks import HybridActorCritic
from .local_iterate import D,x,a,base,play,seeds

class LaneFeaturePolicy(HybridActorCritic):
    def __init__(self):
        super().__init__(47,(256,256),activation="tanh",init_log_std=-0.5,lane_keep_bias=1.0)
        self.lane_features=torch.nn.Parameter(torch.zeros(3))
    def forward(self, obs):
        mu,std,logits,value=super().forward(obs)
        # Evaluation and parameter search use single observations on CPU.
        # Index order in observation: two lanes left, left, ego, right, two lanes right.
        additions=[]
        w_gap,w_conn,w_change=self.lane_features.tolist()
        for row in obs.detach().tolist():
            shifts=[]
            for k in (1,3):
                front=120.*row[1+4*k]
                rear=-120.*row[3+4*k]
                front_closure=max(0.,-27.78*row[2+4*k])
                rear_closure=max(0.,27.78*row[4+4*k])
                spacing=min(front/(4.+1.3*front_closure),rear/(4.+1.3*rear_closure))
                proximity=1./(1.+max(0.,spacing)**2)
                conn=row[21+k]-row[23]
                shifts.append(w_gap*proximity+w_conn*conn+w_change)
            additions.append([shifts[0],0.,shifts[1]])
        return mu,std,logits+logits.new_tensor(additions),value

def init(model=None):
    x.OUT=D
    x.initialize(base,False)
    policy=LaneFeaturePolicy()
    missing=policy.load_state_dict(x.agent.policy.state_dict(),strict=False)
    assert missing.missing_keys==["lane_features"] and not missing.unexpected_keys
    if model:policy.load_state_dict(torch.load(model,map_location="cpu"))
    x.agent.policy=policy.eval()

def run_case(task):
    arm,param,route,seed=task
    with torch.no_grad():x.agent.policy.lane_features.copy_(torch.tensor(param))
    row=play((route,seed,0));row["arm"]=arm
    return row

def search():
    rng=random.Random(240600)
    arms=[(0.,0.,0.),(-3.,0.,0.),(-4.,2.,0.),(-7.,3.,0.),(-8.,5.,0.5),
          (-5.,5.,0.5),(-10.,7.,1.),(-3.,3.,1.),(0.,3.,0.),(2.,2.,0.)]
    routes=list(x.m.ROUTES)+x.HARD*2
    def batch(pool,ids,n):
        rows=[]
        for _ in range(n):
            for rt in routes:rows.append((rt,rng.randrange(961000000,966000000)))
        return list(pool.map(run_case,[(i,arms[i],rt,s) for rt,s in rows for i in ids]))
    def rank(results,ids):
        def key(i):
            by=Counter();success=Counter();crash=Counter()
            for r in results:
                if r["arm"]!=i:continue
                rt=r["route"];by[rt]+=1;success[rt]+=r["outcome"]=="arrival";crash[rt]+=r["outcome"]=="collision"
            return (sum(success[k]/v for k,v in by.items()),-sum(crash[k]/v for k,v in by.items()),-abs(arms[i][0]))
        return sorted(ids,key=key,reverse=True)
    with ProcessPoolExecutor(6,initializer=init) as pool:
        s1=batch(pool,list(range(len(arms))),2)
        ids=rank(s1,range(len(arms)))[:3]
        print("V screen",ids,flush=True)
        x.write_json(D/"V_screen.json",dict(arms=arms,shortlist=ids,rows=s1))
        # Evaluate a continuous point between the best two and one extrapolated point.
        p,q=arms[ids[0]],arms[ids[1]]
        for t in (.5,-.5):
            arms.append(tuple(a+t*(b-a) for a,b in zip(p,q)));ids.append(len(arms)-1)
        s2=batch(pool,ids,4)
    chosen=rank(s2,ids)[0]
    init()
    with torch.no_grad():x.agent.policy.lane_features.copy_(torch.tensor(arms[chosen]))
    x.agent.save(str(D/"V.pt"))
    x.env.close()
    x.write_json(D/"V_search.json",dict(arms=arms,chosen=chosen,rows=s2))
    print("V selected",chosen,arms[chosen],flush=True)
def evaluate(stage,model="V"):
    tasks=seeds(stage)
    with ProcessPoolExecutor(6,initializer=init,initargs=(D/(model+".pt"),)) as pool:rows=list(pool.map(play,tasks))
    x.write_json(D/(model+"_"+stage+".json"),dict(sha256=x.digest(D/(model+".pt")),rows=rows))
    print(model,stage,dict(Counter(r["outcome"] for r in rows)),flush=True)
if __name__=="__main__":
    stage=sys.argv[1]
    if stage=="train":search()
    else:evaluate(stage)
