"""Observation-only expert for blocked merges, plus matched evaluation and dataset collection."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path

import numpy as np

from utils import jys3_train_physical_imitation as prior

ROOT=prior.ROOT.parent/'jys3_physical_expert_gap'
AUDIT=prior.v3.read(prior.data.ROOT/'teacher_audit.json')['teacher_panel']
VARIANTS=('align','yield_early','creep')
BASES={'matched':91000000,'fresh':94500000,'candidates':94600000,'validation':94700000}


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
    if variant=='original':
        return original,'blocked'
    # Preserve stopping space before a real lane end; lateral acceptance stays with the original expert.
    stop_at=2.0 if variant!='yield_early' else 5.
    approach=min(limit,np.sqrt(2*1.7*max(0.,remaining-stop_at)))
    if variant=='creep':
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


def expert(x,env):
    return command(x,env,prior.v3.read(ROOT/'selected.json')['variant'])[0]


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
            row=dict(route=route,rep=rep,lane=lane,seed=seed,start=start,steps=step,
                     outcome=info['outcome'],blocked=blocked,following=following,
                     moves=moves,merge_after_wait=merged_after_wait,modes=dict(modes))
            if collect and row['outcome']=='arrival':
                folder=ROOT/phase/'samples';folder.mkdir(parents=True,exist_ok=True)
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['probe','evaluate','collect'])
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as pool:
        if args.phase=='probe':
            failed=[(r['route'],(r['seed']-91000000-r['route'])//100,r['seed']) for r in AUDIT if r['outcome']=='timeout']
            for variant in ('original',*VARIANTS):
                rows=run('probe',variant,failed,pool)
                print(variant,counts(rows),flush=True)
            return
        variant=prior.v3.read(ROOT/'selected.json')['variant']
        if args.phase=='evaluate':
            for phase,reps in [('matched',16),('fresh',8)]:
                if phase=='fresh':
                    run(phase,'original',cases(phase,reps),pool)
                rows=run(phase,variant,cases(phase,reps),pool)
                print(phase,counts(rows),flush=True)
            return
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
            prior.v3.save(ROOT/('train.json' if phase=='candidates' else 'valid.json'),selected)
            print(phase,counts(rows),len(selected),flush=True)


if __name__=='__main__':
    main()
