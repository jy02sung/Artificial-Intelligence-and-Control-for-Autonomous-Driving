"""Iterative dataset aggregation on corrected forks; predeclared selection and holdouts."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import jys3_train_physical_imitation as old

ROOT = old.ROOT.parent / 'jys3_physical_dagger_multi'
FIRST = old.ROOT/'BC_DAgger'/'dagger1.pt'
SHA = '6690df50d70c82c1b61230e2e90d112c6ddb03af61516385e1d8c56f66a83eb3'
OUTCOMES = ('arrival','collision','stall','timeout','off_route')
ROUTES = tuple(range(1,17))
read, save = old.v3.read, old.v3.save


def checkpoint(k):
    return FIRST if k == 1 else ROOT/f'R{k}'/f'dagger{k}.pt'


def shard(k, kind, rep, base, routes):
    folder=ROOT/f'R{k}'/kind
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/f'{rep}.json'
    collect=kind in ('uniform','weak')
    expected=[base+rep*100+r for r in routes]
    if manifest.exists():
        records=read(manifest)
        assert [r['seed'] for r in records]==expected
        assert not collect or all((folder/r['file']).is_file() for r in records)
        return records
    torch.set_num_threads(1)
    policy=old.dagger.policy(checkpoint(k-1 if collect else k))
    records=[]
    with old.simulator() as env:
        assert env.sim.physical_forks and not env.sim.lc_safety_enabled
        for route,seed in zip(routes,expected):
            name=f'route_{route:02d}'
            lane=min(rep%2,env.counts[env.routes[name][0]]-1)
            if collect:
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed,options={'route':name})
                x,y=[],[]
                for step in range(1,1001):
                    assert obs.shape==(52,) and not obs[50:].any()
                    state=obs[:50].copy()
                    x.append(state)
                    y.append(old.data.bc.expert(state,env))
                    obs,_,done,cut,info=env.step(policy.predict(state,deterministic=True))
                    if done or cut:break
                assert done or cut
                row=dict(steps=step,outcome=info['outcome'],route=route,seed=seed,lane=lane)
                old.save_samples(folder,row,np.stack(x),np.stack(y))
            else:
                _,row=old.replay.drive(env,policy,route,seed,lane,True)
                row.update(route=route,seed=seed,case=rep,depart_lane=lane)
            assert row['outcome'] in OUTCOMES
            records.append(row)
    save(manifest,records)
    return records


def jobs(pool,k,kind,reps,base,routes=ROUTES):
    return [pool.submit(shard,k,kind,rep,base,routes) for rep in range(reps)]


def existing(folder,reps):
    return [r for i in range(reps) for r in read(folder/f'{i}.json')]


def fit(k,x,y,vx,vy,critical):
    folder=ROOT/f'R{k}'
    model=old.dagger.policy(checkpoint(k-1)).policy
    opt=torch.optim.Adam(model.parameters(),lr=.0002)
    cls=(1-y[:,1]).long()
    changed=torch.where(cls!=1)[0]
    target=(1-vy[:,1]).long()
    gen=torch.Generator().manual_seed(12)
    ledger=[]
    best=-float('inf')
    for epoch in range(1,13):
        model.train()
        for batch in torch.randperm(len(x),generator=gen).split(512):
            extra=changed[torch.randint(len(changed),(64,),generator=gen)]
            danger=critical[torch.randint(len(critical),(64,),generator=gen)] if len(critical) else critical
            ids=torch.cat((batch,danger,extra))
            mu,_,logits,_=model(x[ids])
            n=len(batch)+len(danger)
            loss=F.smooth_l1_loss(mu[:n,0],y[ids[:n],0],beta=.15)+F.cross_entropy(logits,cls[ids])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu,_,z,_=model(vx)
            conf=torch.bincount(target*3+z.argmax(1),minlength=9).reshape(3,3)
            f1=2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)
            mae=float((mu[:,0]-vy[:,0]).abs().mean())
            score=float((f1[0]+f1[2])/2)-.1*mae
        ledger.append(dict(epoch=epoch,score=score,mae=mae,f1=f1.tolist()))
        if score>best:
            best,selected=score,epoch
            old.v3.j.atomic_torch(dict(schema=old.dagger.SCHEMA,policy=model.state_dict()),checkpoint(k))
    save(folder/'fit.json',dict(selected=selected,epochs=ledger))


def measures(rows):
    c=Counter(r['outcome'] for r in rows)
    return {k:c[k] for k in OUTCOMES}


def pairs(before,after):
    assert len(before)==len(after)
    t=Counter()
    for a,b in zip(before,after):
        assert all(a[key]==b[key] for key in ('route','seed','start'))
        t[a['outcome']+' -> '+b['outcome']]+=1
    return dict(t)


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True)
    assert hashlib.sha256(FIRST.read_bytes()).hexdigest()==SHA
    seeds=set()
    def reserve(values):
        values=list(values)
        new=set(values)
        assert len(new)==len(values) and not seeds & new
        seeds.update(new)
    def planned(base,reps,routes=ROUTES):
        return [base+100*rep+r for rep in range(reps) for r in routes]
    expert=read(old.data.ROOT/'train.json')
    valid=existing(old.ROOT/'expert'/'validation',2)
    r1folder=old.ROOT/'BC_DAgger'/'collection'
    r1=existing(r1folder,4)
    for rows in [expert,valid,r1]:
        reserve(r['seed'] for r in rows)
    for base,reps in [*[(b,4) for b in old.PANELS.values()],(81260100,4),(91000000,16)]:
        reserve(planned(base,reps))
    assert len(expert)==160 and len(valid)==32 and len(r1)==64
    x,y=old.dagger.load_rollouts(expert,old.data.ROOT/'samples')
    vx,vy=old.dagger.load_rollouts(valid,old.ROOT/'expert'/'validation')
    assert len(x)==57458 and len(vx)==11548
    critical=[]
    tail_total=0
    def add(rows,folder):
        nonlocal x,y,tail_total
        xx,yy=old.dagger.load_rollouts(rows,folder)
        offset=len(x)
        for r in rows:
            end=offset+r['steps']
            first=max(offset,end-15)
            if r['outcome']!='arrival':tail_total+=end-first
            if r['outcome']=='collision':critical.extend(range(first,end))
            offset=end
        assert offset==len(x)+len(xx)
        x,y=torch.cat((x,xx)),torch.cat((y,yy))
    add(r1,r1folder)
    assert len(x)==74961
    analyses={}
    screens={}
    initial=[r for p,b in old.PANELS.items() for r in existing(old.ROOT/'BC_DAgger'/p,4)]
    assert [r['seed'] for r in initial]==[v for b in old.PANELS.values() for v in planned(b,4)]
    def record(k,rows,weak):
        assert len(rows)==128
        pairs(initial,rows)
        screens[k]=rows
        analyses[k]=dict(states=len(x),critical=len(critical),all_failure_tail=tail_total,
                         weak=weak,counts=measures(rows),sha256=hashlib.sha256(checkpoint(k).read_bytes()).hexdigest())
        save(ROOT/'rounds.json',analyses)
    record(1,initial,[])
    def rank(k):
        c=analyses[k]['counts']
        return c['arrival'],-c['collision'],-c['off_route'],-k
    with ProcessPoolExecutor(8) as pool:
        for k in range(2,9):
            fails=Counter(r['route'] for r in screens[k-1] if r['outcome']!='arrival')
            weak=[5,7,13,14] if k==2 else sorted(ROUTES,key=lambda r:(-fails[r],r))[:4]
            packs=[]
            for kind,base,reps,routes in [('uniform',72000000+k*100000,8,ROUTES),
                                          ('weak',73000000+k*100000,4,weak)]:
                reserve(planned(base,reps,routes))
                packs.append((kind,jobs(pool,k,kind,reps,base,routes)))
            for kind,futures in packs:
                add(old.gather(futures),ROOT/f'R{k}'/kind)
            if not (ROOT/f'R{k}'/'fit.json').exists():
                fit(k,x,y,vx,vy,torch.tensor(critical,dtype=torch.long))
            evals=[f for panel,b in old.PANELS.items() for f in jobs(pool,k,panel,4,b)]
            record(k,old.gather(evals),weak)
            if k==5:
                best=max(screens,key=rank)
                c=analyses[best]['counts']
                if c['arrival']>=122 and c['collision']<=6:
                    break
        best=max(screens,key=rank)
        save(ROOT/'selection.json',dict(best=best,last=k,sha256=analyses[best]['sha256']))
        hold_jobs={m:jobs(pool,m,'hold64',4,81260100) for m in sorted({best,1})}
        new_jobs=jobs(pool,best,'final256',16,91000000)
        hold={m:old.gather(f) for m,f in hold_jobs.items()}
        final=old.gather(new_jobs)
    assert [r['seed'] for r in final]==planned(91000000,16)
    assert all([r['seed'] for r in rows]==planned(81260100,4) for rows in hold.values())
    assert hashlib.sha256(FIRST.read_bytes()).hexdigest()==SHA
    cf=measures(final)
    byroute=lambda rows:{i:measures([r for r in rows if r['route']==i]) for i in ROUTES}
    summary=dict(rounds=analyses,best=best,last=k,unique_seeds=len(seeds),screen_routes=byroute(screens[best]),
                 final_routes=byroute(final),screen_transitions=pairs(initial,screens[best]),
                 hold={m:measures(rs) for m,rs in hold.items()},hold_transitions=pairs(hold[1],hold[best]),
                 final=cf,passed=cf['arrival']>=244 and cf['collision']<=12 and not(cf['timeout'] or cf['stall']))
    save(ROOT/'summary.json',summary)


if __name__=='__main__':
    main()
