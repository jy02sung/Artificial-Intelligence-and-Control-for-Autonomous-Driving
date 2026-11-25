"""Diagnose stopped side-by-side vehicles; test holding more approach space in a slow queue."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from utils import jys3_rolling_expert as roll

ROOT=roll.ROOT.parent/'jys3_rolling_recovery'
CASES=[(5,4),(14,6)]
MODES=('reserve20','reserve30','queue_tail')


def control(x,env,mode):
    action,waiting,d,reason=roll._decision(x,env,'strict')
    if mode=='strict' or not waiting:
        return action,waiting,d,reason
    i=1+4*(2-d)
    v=float(x[0])*env.vmax
    front=float(x[i])*env.visibility
    rear=-float(x[i+2])*env.visibility
    remain=float(x[39])*400.
    # A slow vehicle close to the intended lane can stop while the ego is still approaching.
    vf=max(0.,v+float(x[i+1])*env.vmax)
    vr=max(0.,v+float(x[i+3])*env.vmax)
    if remain>70. or not ((front<20. and vf<13.) or (rear<18. and vr<13.)):
        return action,waiting,d,reason
    reserve={'reserve20':20.,'reserve30':30.,'queue_tail':24.}[mode]
    target=np.sqrt(2*2.5*max(0.,remain-reserve))
    if mode=='queue_tail':
        target=min(target,max(0.,vf-4.+.35*(front-10.)))
    accel=min(float(action[0])* (env.max_decel if action[0]<0 else env.max_accel),(target-v)/1.3)
    a=np.array([np.clip(accel/(env.max_decel if accel<0 else env.max_accel),-1.,1.),0.],np.float32)
    return a,waiting,d,'reserve'


def inspect_case(job):
    route,rep=job
    with roll.baseline.simulator() as env:
        path=f'route_{route:02d}'
        env.ego_depart_lane=str(min(rep%2,env.counts[env.routes[path][0]]-1))
        seed=roll.BASE+100*rep+route
        x,_=env.reset(seed=seed,options={'route':path})
        edge='170059994'
        dest_lane=3 if route==5 else 2
        traffic=roll.traci
        track={}
        encounters=[]
        for k in range(1000):
            a,w,d,_=control(x[:50],env,'strict')
            eid,li=env.snapshot[:2]; v=env.snapshot[7]
            vehicles=traffic.lane.getLastStepVehicleIDs(f'{edge}_{dest_lane}')
            ep=traffic.vehicle.getLanePosition(env.ego_id) if eid==edge else None
            poslist=[]
            for vehicle in vehicles:
                pos=traffic.vehicle.getLanePosition(vehicle)
                vs=traffic.vehicle.getSpeed(vehicle)
                track.setdefault(vehicle,[]).append((k,vs,pos))
                if ep is not None:
                    dist=pos-ep
                    length=traffic.vehicle.getLength(vehicle)
                    if -traffic.vehicle.getLength(env.ego_id)<dist<length:
                        poslist.append(vehicle)
            if w and eid==edge:
                encounters.append(dict(k=k,v=v,rem=env.snapshot[-1],position=ep,beside=poslist,
                                       overlap=bool(x[37 if d==1 else 36])))
            if k==999:
                final=[]
                for vehicle in poslist:
                    veh=traffic.vehicle
                    ix=veh.getRouteIndex(vehicle)
                    path_hv=veh.getRoute(vehicle)
                    final.append(dict(id=vehicle,type=veh.getTypeID(vehicle),vehicle_class=veh.getVehicleClass(vehicle),
                        stop_state=veh.getStopState(vehicle),next_edge=path_hv[ix+1] if ix+1<len(path_hv) else None,
                        exits=env.links.get(edge,{}),leader=veh.getLeader(vehicle,120.),length=veh.getLength(vehicle),
                        position=veh.getLanePosition(vehicle),timeline=track[vehicle]))
            x,_,done,cut,info=env.step(a)
            if done or cut:break
        assert info['outcome']=='timeout' and k==999
        return dict(route=route,seed=seed,encounters=encounters,side=final,steps=k+1)


def shard(job):
    mode,cases=job
    with roll.baseline.simulator() as env:
        return [roll.episode(env,r,i,mode,controller=control) for r,i in cases]


def run():
    ROOT.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(8) as pool:
        probes={mode:[] for mode in MODES}
        workers=[(mode,[case]) for mode in MODES for case in CASES]
        for (mode,_),rows in zip(workers,pool.map(shard,workers)):
            probes[mode]+=rows
        roll.baseline.v3.save(ROOT/'probes.json',probes)
        diag=list(pool.map(inspect_case,CASES))
        roll.baseline.v3.save(ROOT/'neighbors.json',diag)
        for m,rows in probes.items():
            print(m,[(r['route'],r['outcome'],r['longest_slow']*.2) for r in rows],flush=True)
        def rank(mode):
            rows=probes[mode]
            return (sum(r['outcome'] in ('collision','off_route') for r in rows),
                    -sum(r['outcome']=='arrival' for r in rows),sum(r['longest_slow'] for r in rows),MODES.index(mode))
        chosen=min(MODES,key=rank)
        if not any(r['outcome']=='arrival' for r in probes[chosen]): return
        cases=[(r,i) for i in range(8) for r in range(1,17)]
        jobs=[(chosen,cases[j:j+8]) for j in range(0,len(cases),8)]
        rows=[v for group in pool.map(shard,jobs) for v in group]
        strict=roll.baseline.v3.read(roll.ROOT/'strict.json')
        assert [(r['seed'],r['start'],r['obs_sha256']) for r in rows]==[(r['seed'],r['start'],r['obs_sha256']) for r in strict]
        roll.baseline.v3.save(ROOT/'panel.json',rows)
        roll.baseline.v3.save(ROOT/'summary.json',dict(candidate=chosen,metrics=roll.summary(rows),
             transitions=dict(Counter(a['outcome']+' -> '+b['outcome'] for a,b in zip(strict,rows)))))
        print(chosen,roll.summary(rows),flush=True)


if __name__=='__main__':
    run()
