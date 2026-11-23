"""Outcome-based updates of the native network. No human or teacher action labels."""
from collections import deque,Counter
from concurrent.futures import ProcessPoolExecutor
import sys,random,json
import numpy as np
from .local_iterate import D,x,a,base,init,run,seeds
import torch
import torch.nn.functional as F

def collect(task):
    route,seed,offset=task
    rng=random.Random(seed)
    obs,_=x.env.reset(seed=seed,options={"route":route})
    history=[]
    # Explore the two lane-change directions earlier/later across different rollouts.
    with torch.no_grad(): x.agent.policy.lane_head.bias[1]+=offset
    try:
        while True:
            act=x.agent.predict(obs,deterministic=True)
            cmd=int(act[1])
            history.append((obs,1-cmd))
            obs,_,done,cut,info=x.env.step(act)
            if done or cut:break
    finally:
        with torch.no_grad():x.agent.policy.lane_head.bias[1]-=offset
    idx=len(history)-1
    negatives=[]
    if info["collided"]:
        # Only assign lane-action blame when last command moves into an occupied lane.
        # The simulator supplies the terminal collision label.
        row=history[-1]
        if row[1]!=1:negatives.append(row)
    anchors=[row for i,row in enumerate(history[:-6]) if (row[1]!=1 or rng.random()<0.026)]
    arrived=bool(info["arrived"])
    return negatives,anchors,arrived

def dataset():
    path=D/"Z_data.npz"
    if path.exists():return
    rng=random.Random(98931); tasks=[]
    trainseeds=iter(rng.sample(range(943000000,944000000),576))
    for i in range(24):
        routes=list(x.m.ROUTES)+x.HARD*2
        rng.shuffle(routes)
        for r in routes:tasks.append((r,next(trainseeds),rng.choice([0.,0.,-1.,1.])))
    x.write_json(D/"Z_seeds.json",tasks)
    neg=[];anc=[];arr=0
    with ProcessPoolExecutor(6,initializer=init,initargs=(base,)) as pool:
        for i,(n,ac,ok) in enumerate(pool.map(collect,tasks),1):
            neg.extend(n);anc.extend(ac);arr+=ok
            if i%96==0:print("Z collect",i,len(neg),len(anc),flush=True)
    print("Z data",arr,len(neg),len(anc),flush=True)
    np.savez_compressed(path,bad=np.stack([s for s,l in neg]),target=np.array([l for s,l in neg]),anchor=np.stack([s for s,l in anc]))

def train():
    dataset()
    data=np.load(D/"Z_data.npz")
    init(base)
    pi=x.agent.policy
    states=torch.from_numpy(data["anchor"]);bad=torch.from_numpy(data["bad"])
    target=torch.from_numpy(data["target"]).long()
    with torch.no_grad():
        originals=pi(states)
        orig_logp=originals[2].log_softmax(-1)
        orig_mu=originals[0]
    opt=torch.optim.Adam(list(pi.trunk.parameters())+list(pi.lane_head.parameters()),lr=4e-5)
    gen=torch.Generator().manual_seed(5442)
    for i in range(1,241):
        ix=torch.randint(len(states),(384,),generator=gen)
        j=torch.randint(len(bad),(48,),generator=gen)
        mu,_,lg,_=pi(states[ix])
        neglogit=pi(bad[j])[2]
        mask=F.one_hot(target[j],3).bool()
        cost=(neglogit.logsumexp(-1)-neglogit.masked_fill(mask,-torch.inf).logsumexp(-1)).mean()
        kl=F.kl_div(lg.log_softmax(-1),orig_logp[ix],reduction="batchmean",log_target=True)
        loss=cost+3.0*kl+8.0*F.mse_loss(mu,orig_mu[ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),1.);opt.step()
        if i in (60,240):
            x.agent.save(str(D/f"Z{i}.pt"));print("Z",i,cost.item(),kl.item(),flush=True)
    x.env.close()
if __name__=="__main__":train()
