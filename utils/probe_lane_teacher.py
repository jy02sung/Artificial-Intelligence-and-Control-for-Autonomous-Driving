"""Training-only lane teacher; intervention outcomes are never policy scores."""
import argparse,json
import numpy as np
import torch
from utils.curriculum_data import OUT,ROOT,BASE,s,source_agent,save,digest,context
from utils.merge_adapter_env import make_merge_env

CASES=[(f'route_{(2,11,13,14)[i%4]:02d}',930021+i,(4000,4500,5000)[i%3]) for i in range(12)]

def candidate_action(raw,ctx,merge,risks,valid):
    action=raw.copy()
    if merge[0]>.25 and merge[2]<merge[1]:action[0]=min(action[0],-.8)
    requested=int(np.sign(raw[1]))
    if requested and (requested not in valid or risks[requested]>.1):action[1]=0
    needed=int(np.sign(ctx['needed']))
    if needed and needed in valid:
        action[1]=needed if risks[needed]<.1 else 0
    return action

def run(shard,shards):
    torch.set_num_threads(1);agent,_=source_agent();env=make_merge_env();folder=OUT/'lane-teacher-candidates'
    try:
        for route,seed,warm in CASES[shard::shards]:
            for mode in ['baseline','candidate']:
                path=folder/f'{route}-{seed}-{mode}.json'
                sig=dict(code_sha=digest(ROOT/'utils/probe_lane_teacher.py'),parent_sha=digest(BASE),route=route,seed=seed,warmup_steps=warm,mode=mode)
                if path.exists():
                    assert json.loads(path.read_text(encoding='utf8'))['signature']==sig
                    continue
                env.warmup_steps=warm;obs,_=env.reset(seed=seed,options={'route':route});rows=[]
                while True:
                    # Original parent ignores merge inputs, as in the paired C teacher collection.
                    baseobs=obs.copy();baseobs[60:63]=0;raw=agent.predict(baseobs,deterministic=True)
                    ctx=context(env,raw);edge=ctx['edge'];lane=ctx['lane']
                    valid=[] if edge.startswith(':') else [d for d in (-1,1) if 0<=lane+d<s.a.b.traci.edge.getLaneNumber(edge)]
                    risks={d:env.safety_context(np.array([raw[0],d]))['lane'] for d in valid}
                    action=candidate_action(raw,ctx,obs[60:63],risks,valid) if mode=='candidate' else raw.copy()
                    row=dict(t=env.step_count*env.step_length,obs=obs.tolist(),baseline_action=raw.tolist(),action=action.tolist(),context=context(env,action),target_risks=risks,overridden=not np.array_equal(raw,action))
                    obs,_,done,cut,info=env.step(action);v=s.a.b.traci.vehicle;alive=env.ego_id in v.getIDList()
                    row.update(after_speed=v.getSpeed(env.ego_id) if alive else None,after_lane=v.getLaneIndex(env.ego_id) if alive else None,after_edge=v.getRoadID(env.ego_id) if alive else None)
                    rows.append(row)
                    if done or cut:break
                outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
                save(path,dict(signature=sig,rows=rows,outcome=outcome,official_evaluation=False,accepted_for_training=False,route_overrides=env.route_override_count))
                print(dict(route=route,seed=seed,mode=mode,outcome=outcome,steps=len(rows)),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();assert 0<=a.shard<a.shards;run(a.shard,a.shards)
