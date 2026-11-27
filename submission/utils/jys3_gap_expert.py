"""Observation-only expert for blocked merges, plus matched evaluation and dataset collection."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import os
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

import numpy as np

from utils import jys3_train_physical_imitation as prior

ROOT=prior.ROOT.parent/'jys3_physical_expert_gap'
def audit_rows():
    return prior.v3.read(prior.data.ROOT/'teacher_audit.json')['teacher_panel']
VARIANTS=('align','yield_early','creep')
BASES={'matched':91000000,'fresh':94500000,'candidates':94600000,'validation':94700000,'holdout':94800000}


def command(x,env,variant):
    original=prior.data.bc.expert(x,env)
    lane,count=np.rint(x[32:34]*5).astype(int)
    links=x[40:].reshape(5,2)[:count]
    eligible=links[:,int(x[34]>0)] if x[35]>=0 else links.max(axis=1)
    choices=np.flatnonzero(eligible)
    if not len(choices) or eligible[lane] or x[38] or original[1]:
        return original,'drive'
    direction=int(np.sign(choices[np.argmin(np.abs(choices-lane))]-lane))
    i=1+4*(2-direction)
    v=float(x[0])*env.vmax
    limit=float(x[31])*env.vmax
    fg=float(x[i])*env.visibility
    rg=-float(x[i+2])*env.visibility
    rear_delta=float(x[i+3])*env.vmax
    remaining=float(x[39])*400
    if variant=='recover' and (remaining>24. or v>2. or fg<=3.):
        return original,'blocked'
    if variant=='original':
        return original,'blocked'
    # Preserve stopping space before a real lane end; lateral acceptance stays with the original expert.
    stop_at=2.0 if variant!='yield_early' else 5.
    approach=min(limit,np.sqrt(2*1.7*max(0.,remaining-stop_at)))
    if variant=='recover':
        # Use remaining space at walking speed when a blocked vehicle has almost stopped.
        target=min(1.6,max(0.,(remaining-3.5)/2.))
        mode='reposition'
    elif variant=='creep':
        target=approach
        mode='approach'
    elif fg>max(8.,v) and rg<max(12.,v) and rear_delta<1.5:
        target=min(approach,max(3.,v+2.))
        mode='pull_ahead'
    else:
        # Let the nearby vehicle pass while there is still road to align longitudinally.
        rear_speed=max(0.,v+rear_delta)
        target=min(approach,max(1.2,rear_speed-3.0)) if rg<max(15.,1.3*v) else approach
        if variant=='yield_early' and fg<max(12.,1.1*v):
            target=min(target,max(1.2,v+float(x[i+1])*env.vmax-2.))
        mode='yield' if target<approach else 'approach'
    # Same-lane car following remains active even while the neighboring lane looks clear.
    gap=float(x[9])*env.visibility
    closing=-float(x[10])*env.vmax
    spacing=2.5+max(0.,1.05*v+v*closing/(2*np.sqrt(2.2*3.2)))
    interaction=(spacing/max(.2,gap))**2 if gap<env.visibility else 0.
    idm=2.2*(1.-(v/max(1.,limit))**4-interaction)
    accel=min(idm,(target-v)/1.1)
    norm=accel/(env.max_accel if accel>=0 else env.max_decel)
    return np.array([np.clip(norm,-1.,1.),0.],np.float32),mode


@lru_cache(maxsize=1)
def selected_variant():
    return prior.v3.read(ROOT/'selected.json')['variant']


def candidate_expert(x,env):
    """Expert for the separate speed-adjustment dataset."""
    return command(x,env,selected_variant())[0]


def raw_samples(phase,variant):
    base=ROOT if variant=='yield_early' else ROOT/variant
    return base/phase/'samples'


def block(job):
    phase,variant,cases=job
    rows=[]
    with prior.simulator() as env:
        assert env.sim.physical_forks and not env.sim.lc_safety_enabled
        for route,rep,seed in cases:
            name=f'route_{route:02d}'
            lane=min(rep%2,env.counts[env.routes[name][0]]-1)
            env.ego_depart_lane=str(lane)
            obs,_=env.reset(seed=seed,options={'route':name})
            collect=phase in ('candidates','validation')
            start=[env.snapshot[i] for i in (0,1,7,-1)]
            states,actions=[],[]
            modes=Counter()
            blocked=following=moves=merged_after_wait=0
            waited=False
            for step in range(1,1001):
                assert obs.shape==(52,) and not obs[50:].any()
                x=obs[:50].copy()
                a,mode=command(x,env,variant)
                modes[mode]+=1
                if collect:
                    states.append(x);actions.append(a)
                gap=x[9]*env.visibility
                following+=int(gap<15.)
                if mode!='drive':
                    blocked+=1
                    waited=True
                edge,index=env.snapshot[:2]
                obs,_,done,cut,info=env.step(a)
                post=env.snapshot
                if post and post[0]==edge and post[1]!=index:
                    moves+=1
                    merged_after_wait+=int(waited and a[1]!=0)
                    waited=False
                elif post and post[0]!=edge:
                    waited=False
                if done or cut:break
            assert done or cut
            assert info['outcome'] != 'arrival' or info['route_intact']
            row=dict(route=route,rep=rep,lane=lane,seed=seed,start=start,steps=step,
                     outcome=info['outcome'],blocked=blocked,following=following,
                     moves=moves,merge_after_wait=merged_after_wait,modes=dict(modes))
            if collect and row['outcome']=='arrival':
                folder=raw_samples(phase,variant);folder.mkdir(parents=True,exist_ok=True)
                prior.save_samples(folder,row,np.stack(states),np.stack(actions))
                row['sha256']=hashlib.sha256((folder/row['file']).read_bytes()).hexdigest()
            rows.append(row)
    return rows


def run(phase,variant,cases,pool):
    chunks=[cases[i:i+4] for i in range(0,len(cases),4)]
    rows=sum(pool.map(block,[(phase,variant,c) for c in chunks]),[])
    prior.v3.save(ROOT/f'{phase}_{variant}.json',rows)
    return rows


def counts(rows):
    return Counter(r['outcome'] for r in rows)


def cases(phase,reps):
    return [(r,i,BASES[phase]+i*100+r) for i in range(reps) for r in range(1,17)]


def inspect_split(rows,folder,env):
    """Use the expert's actual neighboring-lane gap test on every saved state."""
    gaps=close=episodes=close_episodes=0
    total=Counter()
    for row in rows:
        path=folder/row['file']
        assert row['outcome']=='arrival'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with np.load(path,allow_pickle=False) as sample:
            x,a=sample['obs'],sample['actions']
        assert x.shape==(row['steps'],50) and a.shape==(len(x),2)
        assert all(np.isfinite(item).all() and np.abs(item).max()<=1 for item in (x,a))
        assert np.isin(a[:,1],[-1,0,1]).all()
        t=np.arange(len(x))
        lane,count=np.rint(x[:,32:34]*5).astype(int).T
        links=x[:,40:].reshape(-1,5,2)
        side=(x[:,34]>0).astype(int)
        eligible=np.where(x[:,35,None]>=0,links[t,:,side],links.max(axis=2)).astype(bool)
        eligible &= np.arange(5)<count[:,None]
        desired=np.argmin(np.where(eligible,abs(np.arange(5)-lane[:,None]),10),axis=1)
        direction=np.sign(desired-lane)
        index=1+4*(2-direction)
        speed=x[:,0]*env.vmax
        front=x[t,index]*env.visibility
        rear=-x[t,index+2]*env.visibility
        front_closing=-x[t,index+1]*env.vmax
        rear_closing=x[t,index+3]*env.vmax
        blocked=(eligible.any(axis=1)&(direction!=0)&(x[:,38]==0)&
                 (x[t,np.where(direction==1,37,36)]==0)&
                 ((front<=3+.45*speed+np.maximum(0,front_closing))|
                  (rear<=3+.50*np.maximum(0,speed+rear_closing)+np.maximum(0,rear_closing))))
        following=x[:,9]*env.visibility<15
        gaps+=int(blocked.sum())
        close+=int(following.sum())
        episodes+=int(blocked.any())
        close_episodes+=int(following.any())
        total.update(map(int,a[:,1]))
    return dict(episodes=len(rows),steps=sum(total.values()),lane_commands=dict(total),
                traffic_blocked_episodes=episodes,traffic_blocked_steps=gaps,
                close_following_episodes=close_episodes,close_following_steps=close)


def verify(variant,output):
    train=prior.v3.read(output/'train.json')
    valid=prior.v3.read(output/'valid.json')
    original=prior.v3.read(prior.data.ROOT/'train.json')
    panels=[cases(k,n) for k,n in [('matched',16),('fresh',8),('candidates',20),('validation',4),('holdout',8)]]
    all_seeds=[item[2] for panel in panels for item in panel]+[r['seed'] for r in original]
    assert len(set(all_seeds))==len(all_seeds)
    for phase,rows,per_route in [('candidates',train,10),('validation',valid,2)]:
        assert Counter(r['route'] for r in rows)=={r:per_route for r in range(1,17)}
        assert len(set(r['seed'] for r in rows))==len(rows)
        target=output/('samples' if phase=='candidates' else 'valid_samples')
        target.mkdir(exist_ok=True)
        for row in rows:
            src=raw_samples(phase,variant)/row['file']
            dst=target/row['file']
            if not dst.exists():
                try:
                    os.link(src,dst)
                except OSError:
                    shutil.copy2(src,dst)
    candidates=prior.v3.read(ROOT/f'candidates_{variant}.json')
    ordered=[]
    for route in range(1,17):
        for parity in range(2):
            group=[r for r in candidates if r['outcome']=='arrival' and r['route']==route and r['rep']%2==parity]
            ordered.extend(group[:5])
    assert len(ordered)==len(train)
    with prior.simulator() as env:
        stats={k:inspect_split(rows,folder,env) for k,rows,folder in [
            ('original_train',original,prior.data.ROOT/'samples'),('train',train,output/'samples'),
            ('valid',valid,output/'valid_samples'),
            ('same_cohort_first_successes',ordered,raw_samples('candidates',variant))]}
    report=dict(expert=variant,traffic='high_real_ratio',demand_vph=15000,physical_forks=True,
                safety_gate=False,observations=50,statistics=stats,
                train_merge_after_wait=sum(r['merge_after_wait']>0 for r in train),
                train_merge_after_wait_events=sum(r['merge_after_wait'] for r in train),
                strata=dict(Counter(r['stratum'] for r in train)),
                candidates=dict(counts(prior.v3.read(ROOT/f'candidates_{variant}.json'))),
                validation_candidates=dict(counts(prior.v3.read(ROOT/f'validation_{variant}.json'))))
    prior.v3.save(output/'manifest.json',report)
    archive=output/'demonstrations.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as bundle:
        for name in ('train.json','valid.json','manifest.json','selected.json'):
            bundle.write(output/name,name)
        for folder,rows in [('samples',train),('valid_samples',valid)]:
            for row in rows:
                file=Path(folder)/row['file']
                bundle.write(output/file,str(file))
    with ZipFile(archive) as bundle:
        assert len(bundle.namelist())==4+len(train)+len(valid)
        assert bundle.testzip() is None
    print(report,flush=True)


def recovered_examples():
    baseline=prior.v3.read(ROOT/'candidates_original.json')
    candidate=selected_variant()
    lookup={r['seed']:r for r in prior.v3.read(ROOT/f'candidates_{candidate}.json')}
    recovered=[]
    for old in baseline:
        new=lookup[old['seed']]
        assert old['start']==new['start']
        if old['outcome']!='arrival' and new['outcome']=='arrival':
            recovered.append(dict(**new,original_outcome=old['outcome'],original_steps=old['steps']))
    prior.v3.save(ROOT/'recovered_examples.json',recovered)
    with ZipFile(ROOT/'recovered_examples.zip','w',compression=ZIP_DEFLATED) as bundle:
        bundle.write(ROOT/'recovered_examples.json','recovered_examples.json')
        for row in recovered:
            file=raw_samples('candidates',candidate)/row['file']
            assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            bundle.write(file,'samples/'+row['file'])
    with ZipFile(ROOT/'recovered_examples.zip') as bundle:
        assert bundle.testzip() is None
    print(Counter(r['route'] for r in recovered),flush=True)


def select():
    # A tie favors recovering one of the known deadlocks, then the narrower intervention.
    order=['original','recover','yield_early']
    metrics={}
    for candidate in order:
        matched=audit_rows() if candidate=='original' else prior.v3.read(ROOT/f'matched_{candidate}.json')
        fresh=prior.v3.read(ROOT/f'fresh_{candidate}.json')
        probe=prior.v3.read(ROOT/f'probe_{candidate}.json')
        outcomes=counts(matched+fresh)
        metrics[candidate]=[outcomes['collision']+outcomes['off_route'],-outcomes['arrival'],
                            -counts(probe)['arrival'],order.index(candidate)]
    winner=min(order,key=metrics.get)
    prior.v3.save(ROOT/'selected.json',dict(variant=winner,status='screened',ranking=metrics))
    print(winner,metrics,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['probe','evaluate','collect','verify','probe_recover','holdout','select','baseline','diagnose','collect_reference','verify_reference','recovered'])
    parser.add_argument('--candidate', choices=[*VARIANTS,'recover'])
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.phase=='select':
        select()
        return
    if args.phase=='recovered':
        recovered_examples()
        return
    reference=args.phase.endswith('_reference')
    output=ROOT/'reference_dense' if reference else ROOT
    output.mkdir(exist_ok=True)
    if args.phase in ('verify','verify_reference'):
        verify('original' if reference else selected_variant(),output)
        return
    with ProcessPoolExecutor(8) as pool:
        if args.phase=='baseline':
            print(counts(run('matched','original',cases('matched',16),pool)),flush=True)
            return
        if args.phase in ('probe','probe_recover'):
            failed=[(r['route'],(r['seed']-91000000-r['route'])//100,r['seed']) for r in audit_rows() if r['outcome']=='timeout']
            if args.phase=='probe_recover':
                print(counts(run('probe','recover',failed,pool)),flush=True)
                return
            for variant in ('original',*VARIANTS):
                rows=run('probe',variant,failed,pool)
                print(variant,counts(rows),flush=True)
            def rank(candidate):
                rows=prior.v3.read(ROOT/f'probe_{candidate}.json')
                outcomes=counts(rows)
                return (outcomes['collision']+outcomes['off_route'], -outcomes['arrival'],
                        sum(r['steps'] for r in rows))
            prior.v3.save(ROOT/'selected.json',dict(variant=min(VARIANTS,key=rank),status='probe_winner'))
            return
        variant=args.candidate or selected_variant()
        if args.phase=='diagnose':
            failed=[(r['route'],r['rep'],r['seed']) for phase in ('candidates','validation')
                    for r in prior.v3.read(ROOT/f'{phase}_{variant}.json') if r['outcome']!='arrival']
            print(counts(run('failed_collection','original',failed,pool)),flush=True)
            return
        if args.phase=='holdout':
            for option in ('original',variant):
                print(option,counts(run('holdout',option,cases('holdout',8),pool)),flush=True)
            return
        if args.phase=='evaluate':
            for phase,reps in [('matched',16),('fresh',8)]:
                if phase=='fresh' and not (ROOT/'fresh_original.json').exists():
                    run(phase,'original',cases(phase,reps),pool)
                rows=run(phase,variant,cases(phase,reps),pool)
                print(phase,counts(rows),flush=True)
            return
        if reference:
            variant='original'
            prior.v3.save(output/'selected.json',dict(variant=variant,status='reference_expert'))
        else:
            assert variant==selected_variant()
        for phase,reps in [('candidates',20),('validation',4)]:
            rows=run(phase,variant,cases(phase,reps),pool)
            selected=[]
            for r in range(1,17):
                for parity in range(2):
                    group=[v for v in rows if v['route']==r and v['rep']%2==parity and v['outcome']=='arrival']
                    if phase=='candidates':
                        assert len(group)>=5
                        regular=group[:3]
                        difficult=sorted(group[3:],key=lambda v:(-int(v['merge_after_wait']>0),-v['blocked'],-v['following'],v['seed']))[:2]
                        for v in regular:v['stratum']='ordered'
                        for v in difficult:v['stratum']='traffic_enriched'
                        selected+=regular+difficult
                    else:
                        selected+=group[:1]
            assert len(selected)==(160 if phase=='candidates' else 32)
            name='train' if phase=='candidates' else 'valid'
            prior.v3.save(output/f'{name}.json',selected)
            print(phase,counts(rows),len(selected),flush=True)


if __name__=='__main__':
    main()
