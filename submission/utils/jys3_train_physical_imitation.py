"""Train 50-input imitation policies and compare their paired lane-driven journeys."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

import numpy as np
import torch

from utils import jys3_collect_physical_expert as data
from utils import jys3_fit_bc as bc_fit
from utils import jys3_imitation_physical as replay

dagger = replay.dagger
v3 = replay.v3
ROOT = data.ROOT.parent / 'jys3_physical_imitation_160'
TRAIN = 27110000
VALID = 69280100
PANELS = {'matched':26010000, 'fresh':70310100}


@contextmanager
def simulator():
    with TemporaryDirectory() as tmp:
        meta = data.bc.environment('high_real_ratio',11,Path(tmp)/'map')
        env = replay.physical.Env(v3.ARMS[0],Path(tmp)/'physical',11)
        env.sim.signs,env.sim.road_flags=meta.signs,meta.road_flags
        try:
            yield env
        finally:
            env.close()


def save_samples(folder,row,obs,actions):
    name=f"route_{row['route']:02d}_{row['seed']}.npz"
    with (folder/(name+'.tmp')).open('wb') as f:
        np.savez_compressed(f,obs=obs,actions=actions)
    (folder/(name+'.tmp')).replace(folder/name)
    row['file']=name


def run_shard(kind,model_name,rep,base,checkpoint=None):
    torch.set_num_threads(1)
    folder=ROOT/model_name/kind
    folder.mkdir(parents=True,exist_ok=True)
    model=dagger.policy(checkpoint) if checkpoint else None
    rows=[]
    with simulator() as env:
        for route in range(1,17):
            name=f'route_{route:02d}'
            lane=min(rep%2,env.counts[env.routes[name][0]]-1)
            seed=base+rep*100+route
            if kind=='validation':
                for attempt in range(10):
                    row,x,y=data.episode(env,route,rep,seed+10000*attempt)
                    if row['outcome']=='arrival':
                        break
                else:
                    raise RuntimeError(f'no expert completion on {route}/{rep}')
                save_samples(folder,row,x,y)
            elif kind=='collection':
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed,options={'route':name})
                x,y=[],[]
                for t in range(1,1001):
                    assert obs.shape==(52,) and tuple(obs[50:])==(0.,0.)
                    state=obs[:50].copy()
                    x.append(state)
                    y.append(data.bc.expert(state,env))
                    obs,_,done,cut,info=env.step(model.predict(state,deterministic=True))
                    if done or cut:break
                assert done or cut
                row=dict(route=route,lane=lane,seed=seed,steps=t,outcome=info['outcome'])
                save_samples(folder,row,np.stack(x),np.stack(y))
            else:
                _,row=replay.drive(env,model,route,seed,lane,True)
                row.update(route=route,seed=seed,depart_lane=lane,case=rep)
            rows.append(row)
    v3.save(folder/f'{rep}.json',rows)
    return rows


def submit(pool,kind,name,base,checkpoint=None,reps=4):
    return [pool.submit(run_shard,kind,name,rep,base,checkpoint) for rep in range(reps)]


def gather(jobs):
    return [r for job in jobs for r in job.result()]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    rows=v3.read(data.ROOT/'train.json')
    ledger=v3.read(data.ROOT/'manifest.json')
    assert Counter(r['route'] for r in rows)=={n:10 for n in range(1,17)}
    for r in rows:
        assert r['outcome']=='arrival'
        assert hashlib.sha256((data.ROOT/'samples'/r['file']).read_bytes()).hexdigest()==r['sha256']
    x,y=dagger.load_rollouts(rows,data.ROOT/'samples')
    assert len(x)==ledger['samples']==57458
    models={n:ROOT/n for n in ('BC','DAgger','BC_DAgger')}
    for folder in models.values():folder.mkdir(exist_ok=True)
    random_init=models['DAgger']/'random_init.pt'
    agent=dagger.m.PPO(obs_dim=50,act_dim=2,buffer=dagger.m.RolloutBuffer(),seed=11,**dagger.HP)
    torch.save(dict(schema=dagger.SCHEMA,policy=agent.policy.state_dict()),random_init)
    checkpoints={'BC':models['BC']/'bc.pt','DAgger':models['DAgger']/'dagger1.pt',
                 'BC_DAgger':models['BC_DAgger']/'dagger1.pt'}
    with ProcessPoolExecutor(6) as pool:
        val_jobs=submit(pool,'validation','expert',VALID,reps=2)
        cold_jobs=submit(pool,'collection','DAgger',TRAIN,random_init)
        old_cold=dagger.BC.parent/'jys3_dagger_cold'/'dagger1.pt'
        old_jobs=submit(pool,'matched','old_DAgger',PANELS['matched'],old_cold)
        valid=gather(val_jobs)
        vx,vy=dagger.load_rollouts(valid,ROOT/'expert'/'validation')
        bc_fit.fit(x,y,vx,vy,len(rows),len(valid),models['BC'])
        warm_jobs=submit(pool,'collection','BC_DAgger',TRAIN,checkpoints['BC'])
        support={'BC':len(x)}
        evaluations={}
        for model in models:
            if model!='BC':
                cold=model=='DAgger'
                samples=gather(cold_jobs if cold else warm_jobs)
                dx,dy=dagger.load_rollouts(samples,models[model]/'collection')
                sx,sy=(dx,dy) if cold else (torch.cat([x,dx]),torch.cat([y,dy]))
                dagger.supervised_fit(sx,sy,vx,vy,random_init if cold else checkpoints['BC'],
                                      models[model],len(dx))
                support[model]=len(sx)
            for panel,base in PANELS.items():
                evaluations[model,panel]=submit(pool,panel,model,base,checkpoints[model])
        new={key:gather(jobs) for key,jobs in evaluations.items()}
        old=gather(old_jobs)
    used={r['seed'] for r in rows+valid}
    assert len(used)==192
    train_seeds={TRAIN+100*rep+n for rep in range(4) for n in range(1,17)}
    assert not used & train_seeds
    panel_seeds={tag:{r['seed'] for r in new['BC',tag]} for tag in PANELS}
    assert not panel_seeds['matched'] & panel_seeds['fresh']
    for tag,seeds in panel_seeds.items():
        assert len(seeds)==64 and not seeds & (used | train_seeds)
        starts=[(r['seed'],r['start']) for r in new['BC',tag]]
        for name in models:
            assert starts==[(r['seed'],r['start']) for r in new[name,tag]]
    baseline={'DAgger':old}
    orig=v3.read(dagger.BC.parent/'jys3_dagger_cold'/'evaluation.json')
    assert [(r['seed'],r['lane']) for r in orig]==[(r['seed'],r['depart_lane']) for r in old]
    for name in ('BC','BC_DAgger'):
        baseline[name]=[dict(r['new'],seed=r['seed']) for rep in range(4)
                        for r in v3.read(replay.ROOT/f'{name}_{rep}.json')['rows']]
    result={}
    for name in models:
        a=new[name,'matched']; b=baseline[name]
        assert [(r['seed'],r['start']) for r in a]==[(r['seed'],r['start']) for r in b]
        result[name]=dict(update_samples=support[name],
            checkpoint_sha256=hashlib.sha256(checkpoints[name].read_bytes()).hexdigest(),
            old_matched=Counter(r['outcome'] for r in b),
            matched=Counter(r['outcome'] for r in a),
            fresh=Counter(r['outcome'] for r in new[name,'fresh']),
            transitions=Counter(o['outcome']+' -> '+n['outcome'] for o,n in zip(b,a)))
    v3.save(ROOT/'summary.json',dict(validation_samples=len(vx),models=result))


if __name__=='__main__':
    main()
