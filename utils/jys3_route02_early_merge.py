"""Replay one recorded congested merge with one earlier left command per branch."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from utils.jys3_classroom_gap_bc import DENSE, prior
from env.sumo_env import traci

ROOT = DENSE.parent / 'route02_early_merge_probe'
TICKS = (356,371,427,441,558)
SEED = 94601802


def run(change):
    row=next(r for r in prior.v3.read(DENSE/'train.json') if r['seed']==SEED)
    with np.load(DENSE/'samples'/row['file']) as packed:
        states,actions=packed['obs'],packed['actions']
    log={}
    with prior.simulator() as env:
        env.ego_depart_lane=str(row['lane'])
        obs,_=env.reset(seed=SEED,options={'route':'route_02'})
        for k in range(1000):
            a=prior.data.bc.expert(obs[:50],env)
            if change is None or k<=change:
                assert np.array_equal(states[k],obs[:50]) and np.array_equal(actions[k],a)
            if k in TICKS and (change is None or k==change):
                edge,lane=env.snapshot[:2]
                assert edge=='844484877' and lane==0 and a[1]==0
                v=float(obs[0])*env.vmax
                dv=float(obs[8])*env.vmax
                pos=traci.vehicle.getLanePosition(env.ego_id)
                others=[(traci.vehicle.getLanePosition(i),i) for i in traci.lane.getLastStepVehicleIDs(edge+'_1')]
                rear=max((q for q in others if q[0]<pos),default=(None,None))[1]
                log[k]=dict(time=k*.2,speed=v,remaining=env.snapshot[-1],
                            front=float(obs[5])*env.visibility,rear=float(-obs[7])*env.visibility,
                            front_min=3+.45*v+max(0,-float(obs[6])*env.vmax),
                            rear_min=3+.5*max(0,v+dv)+max(0,dv), vehicle=rear, rear_accel=[])
            if k==change:
                a=a.copy()
                a[1]=1
            obs,_,done,cut,info=env.step(a)
            for tick,entry in log.items():
                if 0<=k-tick<15 and entry['vehicle'] in traci.vehicle.getIDList():
                    entry['rear_accel'].append(traci.vehicle.getAcceleration(entry['vehicle']))
                if k==tick:
                    entry['next_lane']=env.snapshot[1] if env.snapshot else None
            if done or cut:
                result=dict(tick=change,steps=k+1,outcome=info['outcome'],log=log)
                if change is None:
                    assert k+1==row['steps'] and info['outcome']==row['outcome']
                return result
    raise RuntimeError('No terminal outcome')


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(6) as pool:
        runs=list(pool.map(run,[None,*TICKS]))
    prior.v3.save(ROOT/'results.json',runs)
    for r in runs[1:]:
        k=r['tick']
        v=r['log'][k]
        print(k,r['outcome'],r['steps'],v['next_lane'],v['vehicle'],
              min(v['rear_accel'],default=None), min(runs[0]['log'][k]['rear_accel'],default=None))
