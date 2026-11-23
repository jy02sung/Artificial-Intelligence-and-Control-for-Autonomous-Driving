"""Learn from actual successful lane changes plus actual collisions, native mcw2 actor."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json,random
import numpy as np
import torch
import torch.nn.functional as F
from .local_iterate import x,a,base,D,init

def collect(task):
    rt,seed,keep=task
    env,agent=x.env,x.agent
    with torch.no_grad():
        orig=agent.policy.lane_head.bias[1].item()
        agent.policy.lane_head.bias[1]=orig+keep
    obs,_=env.reset(seed=seed,options={"route":rt})
    changes=[];latest=None
    while True:
        act=agent.predict(obs,deterministic=True)
        cmd=int(act[1]); latest=(obs,1-cmd)
        if cmd:
            lane=a.V.getLaneIndex(env.ego_id);edge=a.V.getRoadID(env.ego_id)
        nxt,_,done,cut,info=env.step(act)
        if done or cut:break
        if cmd and a.V.getRoadID(env.ego_id)==edge and a.V.getLaneIndex(env.ego_id)==lane+cmd:
            changes.append(latest)
        obs=nxt
    with torch.no_grad():agent.policy.lane_head.bias[1]=orig
    return (changes if info["arrived"] else []),([latest] if info["collided"] and cmd else [])
def train():
    rng=random.Random(99213)
    path=D/"R_data.npz"
    if not path.exists():
        tasks=[];ids=iter(rng.sample(range(968000000,969000000),256))
        for i in range(8):
            block=list(x.m.ROUTES)+x.HARD*4
            rng.shuffle(block)
            for rt in block:tasks.append((rt,next(ids),rng.choice([-1.,-.5,0.,.5])))
        x.write_json(D/"R_train_seeds.json",tasks)
        pos=[];neg=[]
        with ProcessPoolExecutor(4,initializer=init,initargs=(base,)) as pool:
            for i,(good,bad) in enumerate(pool.map(collect,tasks),1):
                pos.extend(good);neg.extend(bad)
                if i%64==0:print("R",i,len(pos),len(neg),flush=True)
        np.savez_compressed(path,pos=np.stack([s for s,l in pos]),act=np.array([l for s,l in pos]),
                            neg=np.stack([s for s,l in neg]),negact=np.array([l for s,l in neg]))
    data=np.load(path); zd=np.load(D/"Z_data.npz")
    init(base); pi=x.agent.policy
    anchors=torch.tensor(zd["anchor"])
    good=torch.tensor(data["pos"]);acts=torch.tensor(data["act"]).long()
    bad=torch.tensor(np.concatenate([zd["bad"],data["neg"]]))
    badacts=torch.tensor(np.concatenate([zd["target"],data["negact"]])).long()
    with torch.no_grad():
        mu,_,lg,_=pi(anchors);old=lg.log_softmax(-1)
    pi.load_state_dict(torch.load(D/"Z60.pt",map_location="cpu"))
    opt=torch.optim.Adam(list(pi.trunk.parameters())+list(pi.lane_head.parameters()),lr=3e-5)
    gen=torch.Generator().manual_seed(99312)
    def pick(t,n):return torch.randint(0,len(t),(n,),generator=gen)
    for i in range(1,241):
        ia=pick(anchors,384);ig=pick(good,96);ib=pick(bad,64)
        am,_,lg,_=pi(anchors[ia])
        posloss=F.cross_entropy(pi(good[ig])[2],acts[ig])
        b=pi(bad[ib])[2];mask=F.one_hot(badacts[ib],3).bool()
        avoid=(b.logsumexp(-1)-b.masked_fill(mask,-torch.inf).logsumexp(-1)).mean()
        # Penalize probability assigned to physically absent neighbour lanes during fitting.
        prob=lg.softmax(-1)
        invalid=(prob[:,0]*(anchors[ia,22]<0)+prob[:,2]*(anchors[ia,24]<0)).mean()
        kl=F.kl_div(lg.log_softmax(-1),old[ia],reduction="batchmean",log_target=True)
        loss=avoid+0.22*posloss+1.5*invalid+3.*kl+8.*F.mse_loss(am,mu[ia])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),1.);opt.step()
        if i in (80,240):
            x.agent.save(str(D/f"R{i}.pt"))
    x.env.close()
if __name__=="__main__":train()
