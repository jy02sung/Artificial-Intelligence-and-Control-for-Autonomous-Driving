"""Fit a new BC student from complete demonstrations; split by whole episode."""
import json
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from utils.jys3 import m,HP,OBS_DIM,SCHEMA
from utils.jys3_bc_v2 import OUT

def load(rows):
    obs, labels = [], []
    for row in rows:
        with np.load(OUT/'samples'/row['file']) as data:
            x, y = data['obs'], data['actions']
            assert x.shape==(row['steps'],OBS_DIM) and y.shape==(len(x),2)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            obs.append(x);labels.append(y)
    return torch.from_numpy(np.concatenate(obs)),torch.from_numpy(np.concatenate(labels))

def main():
    rows=json.loads((OUT/'train.json').read_bytes())
    complete=[r for r in rows if r['outcome']=='arrival']
    train=[r for r in complete if r['seed']<24113400]
    valid=[r for r in complete if r['seed']>=24113400]
    x,y=load(train); vx,vy=load(valid)
    fit(x,y,vx,vy,len(train),len(valid),OUT)


def fit(x,y,vx,vy,train_count,valid_count,folder):
    torch.set_num_threads(2)
    model=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP).policy
    optim=torch.optim.Adam(model.parameters(),lr=.0004)
    cls=(1-y[:,1]).long()
    vc=(1-vy[:,1]).long()
    movers=torch.where(cls!=1)[0]
    gen=torch.Generator().manual_seed(11)
    history=[];best=-float('inf')
    for epoch in range(1,31):
        model.train()
        for batch in torch.randperm(len(x),generator=gen).split(512):
            extra=movers[torch.randint(len(movers),(64,),generator=gen)]
            ids=torch.cat([batch,extra])
            mu,_,logits,_=model(x[ids])
            loss=F.cross_entropy(logits,cls[ids])+F.smooth_l1_loss(mu[:len(batch),0],y[batch,0],beta=.15)
            optim.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        model.eval()
        with torch.no_grad():
            mu,_,z,_=model(vx); pred=z.argmax(dim=1)
            errs=(mu[:,0]-vy[:,0]).abs().mean().item()
            conf=torch.bincount(vc*3+pred,minlength=9).view(3,3)
            f1=2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)
            score=float((f1[0]+f1[2])/2)-.1*errs
            history.append(dict(epoch=epoch,change_f1=f1.tolist(),accel_mae=errs,confusion=conf.tolist(),score=score))
        if score>best:
            best=score;selected=epoch
            torch.save(dict(schema=SCHEMA,policy=model.state_dict()),folder/'bc.pt')
    report=dict(fit_episodes=train_count,valid_episodes=valid_count,fit_samples=len(x),valid_samples=len(vx),fit_actions=dict(Counter(map(int,y[:,1]))),selected_epoch=selected,history=history)
    (folder/'fit.json').write_text(json.dumps(report,indent=2));print(selected,len(x),len(vx),flush=True)

if __name__=='__main__':main()
