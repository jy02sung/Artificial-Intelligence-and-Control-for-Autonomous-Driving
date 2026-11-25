"""Matched short pure-PPO test, replacing only observation entries 21:26."""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from utils import completion_focus_v1 as old
from utils.experiment_b import agent_for, save, torch, traci
from utils.experiment_b_eval import evaluate_episode

OUT=old.OUT.parent/'route120_short'
SEEDS=(993100,993200,993300,993400)
EDGES={13:'1063527748',14:'1063527748',16:'86198728'}

def route120(env, offset):
    edge=traci.vehicle.getRoadID(env.ego_id)
    if edge.startswith(':'):
        return 1.
    lane=traci.vehicle.getLaneIndex(env.ego_id)+offset
    if not 0<=lane<traci.edge.getLaneNumber(edge):
        return -1.
    lid=f'{edge}_{lane}'
    remaining=traci.lane.getLength(lid)-traci.vehicle.getLanePosition(env.ego_id)
    path=env.routes[env.episode_route]
    queue=[(traci.vehicle.getRouteIndex(env.ego_id),lid,remaining)]
    far=remaining
    seen=set()
    for i,lid,distance in queue:
        if (i,lid) in seen:
            continue
        seen.add((i,lid))
        if distance>=120 or i+1>=len(path):
            return 1.
        far=max(far,distance)
        for link in traci.lane.getLinks(lid):
            if link[0].startswith(path[i+1]+'_'):
                queue.append((i+1,link[0],distance+link[7]+traci.lane.getLength(link[0])))
    return float(np.clip(far/120,0,1))

def make(stage,arm,seed,folder):
    env=original(stage,arm,seed,folder)
    env._lane_connectivity=lambda offset: route120(env,offset)
    return env

original=old.make_c

def train():
    old.OUT=OUT
    old.make_c=make
    old.train('focus')

def probe():
    env=original('high_real_ratio','C0',33,OUT/'probe')
    rows=[]
    try:
        for n,edge,lane,remaining in [(13,'170059994',2,25),(14,'170059994',2,25),(13,'1063527748',2,60),(14,'1063527748',2,60),(16,'86198728',0,60)]:
            env.reset(seed=993100,options={'route':f'route_{n:02d}'})
            lid=f'{edge}_{lane}'
            traci.vehicle.moveTo(env.ego_id,lid,traci.lane.getLength(lid)-remaining)
            before=env._get_obs()
            func=env._lane_connectivity
            env._lane_connectivity=lambda offset:route120(env,offset)
            after=env._get_obs()
            env._lane_connectivity=func
            assert np.array_equal(before[:21],after[:21]) and np.array_equal(before[26:],after[26:])
            rows.append(dict(route=n,edge=edge,lane=lane,remaining=remaining,original=before[21:26].tolist(),new=after[21:26].tolist()))
    finally:
        env.close()
    save(OUT/'probe.json',rows)

def evaluate(job):
    arm,step,seed=job
    torch.set_num_threads(1)
    path=old.BASE if step==0 else (old.OUT if arm=='original' else OUT)/'focus'/f'policy_{step}.pt'
    agent=agent_for(33)
    agent.load(str(path))
    agent.policy.eval()
    env=(original if arm=='original' else make)('high_real_ratio','C0',33,OUT/'eval_maps'/str(os.getpid()))
    original_step=env.step
    rows=[]
    try:
        for n,lane in [(13,0),(14,0),(13,1),(14,1),(16,0)]:
            changes=[]
            left=[]
            def traced(action):
                edge,l,*_=env.snapshot
                dist=env.snapshot[-1]
                if n==16 and edge==EDGES[n] and l==0 and env._quantize_lane_change(action[env.lane_change_index])==1:
                    left.append(dist)
                result=original_step(action)
                post=env.snapshot
                if post and edge==EDGES[n] and post[0]==edge and post[1]!=l:
                    changes.append([l,post[1],dist])
                return result
            env.step=traced
            row=evaluate_episode(env,agent,f'route_{n:02d}',seed+lane,lane)
            fork=next(x for x in row['forks'].values() if x['edge']==EDGES[n])
            rows.append(dict(route=n,lane=lane,start=row['start'],passed=fork['actual']==fork['expected'],reached='approach_lane' in fork,outcome=row['outcome'],first_left=left[0] if left else None,changes=changes,last=row['last']))
    finally:
        env.close()
    save(OUT/f'{arm}_{step}_{seed}.json',rows)

if __name__=='__main__':
    import sys
    OUT.mkdir(exist_ok=True)
    if sys.argv[1]=='train':
        probe()
        train()
    else:
        with ProcessPoolExecutor(6) as pool:
            list(pool.map(evaluate,itertools.product(('original','route120'),map(int,sys.argv[1:]),SEEDS)))
