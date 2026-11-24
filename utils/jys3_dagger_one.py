"""One DAgger round: BC student rollouts, expert labels, aggregate supervised fit, frozen evaluation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import json
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from utils.jys3 import environment, m, HP, SCHEMA, OBS_DIM
from utils.jys3_bc_v2 import OUT as BC, expert
from utils.jys3_fit_bc import load

OUT = BC.parent / "jys3_dagger_one"
INITIAL = m.ROOT / "models/jys3/onehop_bc_v2.pt"

def policy(path):
    agent=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP)
    state=torch.load(path,weights_only=True,map_location="cpu")
    assert state['schema']==SCHEMA
    agent.policy.load_state_dict(state['policy'])
    agent.policy.eval()
    return agent

def drive(job):
    collect,rep=job
    torch.set_num_threads(1)
    agent=policy(INITIAL if collect else OUT/'dagger1.pt')
    rows=[]
    with tempfile.TemporaryDirectory() as folder:
        env=environment('high_real_ratio',11,Path(folder))
        try:
            for n in range(1,17):
                route=f'route_{n:02d}'
                seed=(27110000 if collect else 26010000)+100*rep+n
                lane=min(rep%2,env.counts[env.routes[route][0]]-1)
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed,options={'route':route})
                states,labels=[],[]
                for t in range(1,1001):
                    action=agent.predict(obs,deterministic=True)
                    if collect:
                        states.append(obs)
                        labels.append(expert(obs,env))
                    obs,_,done,cut,info=env.step(action)
                    if done or cut:break
                result='arrival' if info['arrived'] else 'collision' if info['collided'] else 'stall' if info['stalled'] else 'timeout'
                row=dict(route=n,seed=seed,lane=lane,steps=t,outcome=result)
                if collect:
                    row['file']=f'{route}_{seed}.npz'
                    np.savez_compressed(OUT/row['file'],obs=states,actions=labels)
                rows.append(row)
        finally:env.close()
    return rows

def batch(collect):
    with ProcessPoolExecutor(4) as pool:
        rows=[r for block in pool.map(drive,[(collect,i) for i in range(4)]) for r in block]
    (OUT/('collection.json' if collect else 'evaluation.json')).write_text(json.dumps(rows,indent=2))
    return rows

def fit(rows):
    orig=json.loads((BC/'train.json').read_bytes())
    x,y=load([r for r in orig if r['seed']<24113400 and r['outcome']=='arrival'])
    vx,vy=load([r for r in orig if r['seed']>=24113400 and r['outcome']=='arrival'])
    xs,ys=[x],[y]
    for r in rows:
        with np.load(OUT/r['file']) as data:
            assert data['obs'].shape==(r['steps'],OBS_DIM) and data['actions'].shape==(r['steps'],2)
            assert np.isfinite(data['obs']).all() and np.isfinite(data['actions']).all()
            xs.append(torch.from_numpy(data['obs']));ys.append(torch.from_numpy(data['actions']))
    x,y=torch.cat(xs),torch.cat(ys)
    torch.set_num_threads(2)
    model=policy(INITIAL).policy
    opt=torch.optim.Adam(model.parameters(),lr=.0002)
    cls=(1-y[:,1]).long()
    changed=torch.where(cls!=1)[0]
    gen=torch.Generator().manual_seed(12)
    best=-float('inf')
    for epoch in range(1,13):
        model.train()
        for batch_ids in torch.randperm(len(x),generator=gen).split(512):
            extra=changed[torch.randint(len(changed),(64,),generator=gen)]
            ids=torch.cat((batch_ids,extra))
            mu,_,z,_=model(x[ids])
            loss=F.smooth_l1_loss(mu[:len(batch_ids),0],y[batch_ids,0],beta=.15)+F.cross_entropy(z,cls[ids])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            mu,_,z,_=model(vx)
            vc=(1-vy[:,1]).long()
            conf=torch.bincount(vc*3+z.argmax(1),minlength=9).reshape(3,3)
            f1=2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)
            score=float((f1[0]+f1[2])/2-.1*(mu[:,0]-vy[:,0]).abs().mean())
        if score>best:
            best=score
            torch.save(dict(schema=SCHEMA,policy=model.state_dict()),OUT/'dagger1.pt')
            selected=epoch
    (OUT/'fit.json').write_text(json.dumps(dict(total=len(x),added=sum(r['steps'] for r in rows),selected=selected,epochs=12)))
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    rows=batch(True)
    fit(rows)
    evaluation=batch(False)
    baseline=json.loads((BC/'student.json').read_bytes())
    assert [(r['route'],r['seed'],r['lane']) for r in evaluation]==[(r['route'],r['seed'],r['lane']) for r in baseline]
    print(Counter(r['outcome'] for r in evaluation))
