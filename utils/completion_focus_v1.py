"""Short full-route PPO fine-tune from a preserved checkpoint; paired fresh evaluation."""
import csv,itertools,json,os,random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from utils.experiment_b import agent_for,digest,m,save,torch,traci
from utils.experiment_b_eval import evaluate_episode
from utils.experiment_c_env import OUT as C,make_c
from utils.experiment_c_train import atomic_torch

OUT=C.parent/'completion_focus_v1'
BASE=C/'C0_33'/'policy_900000.pt'
FOCUS=(13,14,5,7,16)
VALID=(992100,992200,992300,992400)
# Keep all 16 routes in rotation; add one complete exposure to each hard route in every short block.
class Schedule:
    def __init__(self,env,kind):
        self.env,self.kind,self.i=env,kind,0
    def reset(self):
        b,i=divmod(self.i,10)
        rng=random.Random(410000+b)
        anchors=list(range(1,17))
        random.Random(850000+b//16).shuffle(anchors)
        z=[anchors[(5*b+j)%16] for j in range(5)]
        if self.kind=='focus':z+=list(FOCUS)
        else:z += [anchors[(5*b+j+5)%16] for j in range(5)]
        rng.shuffle(z)
        route=f'route_{z[i]:02d}'
        # Match starting-lane counts per route over repeated appearances.
        n=self.counts.get(route,0)
        self.counts[route]=n+1
        self.env.ego_depart_lane=str(min(n%2,self.env.counts[self.env.routes[route][0]]-1))
        self.i+=1
        return self.env.reset(seed=73000000+self.i,options={'route':route})
    def step(self,a):return self.env.step(a)
def train(kind):
    torch.set_num_threads(1)
    d=OUT/kind;d.mkdir(exist_ok=True,parents=True)
    if (d/'policy_20000.pt').is_file():return
    a=agent_for(33);a.load(str(BASE))
    env=make_c('high_real_ratio','C0',33,d/'map')
    try:
        sch=Schedule(env,kind);sch.counts={}
        obs,_=sch.reset();stats=[0.,0]
        for target in (5000,10000,15000,20000):
            while a.num_timesteps<target:
                obs,last,_=a.collect_rollout(sch,obs,ep_stats=stats,steps=min(2048,target-a.num_timesteps),timeout_is_failure=False)
                adv,ret=a.buffer.compute_gae(last,a.gamma,a.gae_lambda);a.update(adv,ret)
            atomic_torch(a.policy.state_dict(),d/f'policy_{target}.pt')
            save(d/'progress.json',dict(steps=target,episodes=sch.counts))
    finally:env.close()
def shard(job):
    kind,base=job
    if (OUT/f'{kind}_{base}.json').is_file():return
    torch.set_num_threads(1)
    path=BASE if kind=='baseline' else OUT/kind/'policy_20000.pt'
    a=agent_for(33)
    if kind!='sumo':a.load(str(path));a.policy.eval()
    env=make_c('high_real_ratio','C0',33,OUT/'maps'/str(os.getpid()))
    rows=[]
    try:
        for lane,route in itertools.product((0,1),m.ROUTES):
            sim=base+lane
            if kind=='sumo':
                env.ego_depart_lane=str(min(lane,env.counts[env.routes[route][0]]-1));env.reset(seed=sim,options={'route':route})
                ego=env.ego_id
                start=[*env.snapshot[:2],traci.vehicle.getLanePosition(ego),env.snapshot[7],traci.simulation.getTime()]
                traci.vehicle.setSpeedMode(ego,31);traci.vehicle.setLaneChangeMode(ego,1621);traci.vehicle.setSpeed(ego,-1)
                outcome='timeout'
                for t in range(3000):
                    traci.simulationStep()
                    if ego in traci.simulation.getCollidingVehiclesIDList():outcome='collision';break
                    if ego in traci.simulation.getStartingTeleportIDList():outcome='teleport';break
                    if ego in traci.simulation.getArrivedIDList():outcome='arrival';break
                r=dict(route=route,sim_seed=sim,lane_case=lane,outcome=outcome,start=start)
            else:
                r=evaluate_episode(env,a,route,sim,lane)
            rows.append(r)
        save(OUT/f'{kind}_{base}.json',rows)
    finally:env.close()
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(6) as pool:
        jobs=[pool.submit(train,k) for k in ('focus','control')]
        early=[pool.submit(shard,j) for j in itertools.product(('sumo','baseline'),VALID)]
        for f in jobs:f.result()
        late=[pool.submit(shard,j) for j in itertools.product(('focus','control'),VALID)]
        for f in early+late:f.result()
    from collections import Counter
    ref=None;results={}
    for k in ('baseline','control','focus','sumo'):
        rs=[r for b in VALID for r in json.loads((OUT/f'{k}_{b}.json').read_bytes())]
        starts=[r['start'] for r in rs]
        if ref is None:ref=starts
        assert len(rs)==128 and starts==ref
        results[k]=dict(total=Counter(r['outcome'] for r in rs),routes={rt:Counter(r['outcome'] for r in rs if r['route']==rt) for rt in m.ROUTES})
    save(OUT/'summary.json',results)

