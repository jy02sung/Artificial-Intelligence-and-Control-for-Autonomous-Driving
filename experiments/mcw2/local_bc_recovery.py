"""Offline intervention/recovery demonstrations, using only native SUMO action labels."""
import random,sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from . import local_bc_ppo as b
from .local_bc_ppo import D,m,v,sim,x
OUT=D/"recovery_demos";OUT.mkdir(exist_ok=True)
def episode(task):
    r,sd=task[:2]; lateral=len(task)>2
    folder=D/"lateral_demos" if lateral else OUT
    dest=folder/f"{r}_{sd}.npz"
    if dest.exists():
        with np.load(dest) as d:return str(d["outcome"])
    rng=random.Random(sd);env=b.env;ego=env.ego_id
    obs,_=env.reset(seed=sd,options={"route":r})
    v.setSpeedMode(ego,31);v.setLaneChangeMode(ego,1621);v.setSpeed(ego,-1)
    rows=[];labels=[];nxt=0;reason="timeout";pert=0
    for step in range(3000):
        edge=v.getRoadID(ego);lane=v.getLaneIndex(ego);speed=v.getSpeed(ego)
        injected=step==nxt
        if injected:
            nxt=step+(rng.randrange(70,161) if lateral else rng.randrange(60,201));pert+=1
            if pert<=7:
                # Displace to a low speed or a randomly selected neighbouring lane.
                # The intervention step is excluded from the demonstration labels.
                if (not lateral and rng.random()<.76) or edge.startswith(":"):
                    v.setSpeedMode(ego,0)
                    v.setSpeed(ego,rng.choice([0.,1.,3.,6.,10.,min(25.,speed+4.)]))
                else:
                    count=m.traci.edge.getLaneNumber(edge)
                    choices=[j for j in (lane-1,lane+1) if 0<=j<count]
                    if lateral and choices:target=rng.choice(choices)
                    else:target=lane+rng.choice([-1,1])
                    if 0<=target<count:v.changeLane(ego,target,.2)
            else:injected=False
        m.traci.simulationStep();env.step_count+=1
        if ego in sim.getCollidingVehiclesIDList():reason="collision";break
        if ego in sim.getStartingTeleportIDList():reason="teleport";break
        if ego in sim.getArrivedIDList():reason="arrival";break
        if ego not in v.getIDList():reason="other";break
        if not injected:
            delta=(v.getSpeed(ego)-speed)/env.step_length
            cmd=v.getLaneIndex(ego)-lane if v.getRoadID(ego)==edge else 99
            rows.append(obs);labels.append((np.clip(delta/(env.max_accel if delta>=0 else env.max_decel),-1,1),cmd))
        else:
            v.setSpeedMode(ego,31);v.setSpeed(ego,-1)
        obs=env._get_obs()
    with dest.with_suffix(".tmp").open("wb") as f:np.savez_compressed(f,obs=np.asarray(rows,np.float32),actions=np.asarray(labels,np.float32),outcome=reason)
    dest.with_suffix(".tmp").replace(dest)
    return reason
def main():
    lateral=len(sys.argv)>1 and sys.argv[1]=="lateral"
    if lateral:
        (D/"lateral_demos").mkdir(exist_ok=True)
        ids=iter(random.Random(80223).sample(range(1220000000,1230000000),640))
        seq=[(f"route_{rt:02}",next(ids),True) for i in range(128) for rt in (2,5,11,13,14)]
    else:
        ids=iter(random.Random(45110).sample(range(1200000000,1210000000),768))
        seq=[(rt,next(ids)) for i in range(48) for rt in m.ROUTES]
    c=Counter()
    with ProcessPoolExecutor(5,initializer=b.setup) as pool:
        for i,reason in enumerate(pool.map(episode,seq),1):
            c[reason]+=1
            if i%96==0:print("recovery",i,c,flush=True)
    x.write_json(D/("lateral_summary.json" if lateral else "recovery_summary.json"),dict(outcomes=c,tasks=seq))
if __name__=="__main__":main()
