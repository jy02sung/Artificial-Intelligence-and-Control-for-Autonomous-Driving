"""Training-seed teacher candidates; interventions never count as policy evaluation."""
import argparse,json
import torch
from utils.curriculum_data import OUT,ROOT,s,source_agent,make_env,save,digest,context
from env.junction_lookahead_trials import JunctionLookaheadEnv
CASES=[('route_01',910001,4000),('route_02',910002,4500),('route_01',930001,4500),('route_02',930002,4500)]

def run(shard=0,shards=1):
    torch.set_num_threads(1);agent,meta=source_agent();folder=OUT/'teacher-candidates';folder.mkdir(exist_ok=True)
    env=make_env();env.planning_horizon=150
    try:
        for route,seed,warmup in CASES[shard::shards]:
            for mode in ['baseline','candidate']:
                path=folder/f'{route}-{seed}-{mode}.json'
                signature=dict(code_sha=digest(ROOT/'utils/probe_curriculum_teacher.py'),parent_sha=digest(OUT/'baseline/policy.pt'),route=route,seed=seed,warmup_steps=warmup,mode=mode)
                if path.exists():
                    assert json.loads(path.read_text(encoding='utf8'))['signature']==signature
                    continue
                env.warmup_steps=warmup;obs,_=env.reset(seed=seed,options={'route':route});rows=[];count=0
                while True:
                    raw=agent.predict(obs,deterministic=True);action=raw.copy();feature=JunctionLookaheadEnv.conflict_features(env)
                    # Uses only the candidate observation: visible foe ETA before ego ETA.
                    active=mode=='candidate' and feature[0]>.25 and feature[2]<feature[1]
                    if active:action[0]=min(action[0],-.8)
                    changed=bool(action[0]!=raw[0]);count+=int(changed)
                    ctx=context(env,action);t=env.step_count*env.step_length
                    row=dict(t=t,obs=obs.tolist(),merge=list(feature),baseline_action=raw.tolist(),action=action.tolist(),overridden=changed,context=ctx)
                    obs,reward,done,cut,info=env.step(action)
                    v=s.a.b.traci.vehicle
                    row['after_speed']=v.getSpeed(env.ego_id) if env.ego_id in v.getIDList() else None
                    rows.append(row)
                    if done or cut:break
                outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
                save(path,dict(signature=signature,outcome=outcome,seconds=env.step_count*env.step_length,overridden_steps=count,official_evaluation=False,accepted_for_training=False,rule='risk>.25 and visible foe ETA before ego ETA: cap acceleration command at -0.8; original lane action',rows=rows))
                print(dict(route=route,seed=seed,mode=mode,outcome=outcome,seconds=env.step_count*env.step_length,overrides=count),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();assert 0<=a.shard<a.shards;run(a.shard,a.shards)
