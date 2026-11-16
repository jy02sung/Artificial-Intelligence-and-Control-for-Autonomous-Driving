"""Audited baseline demonstrations and deterministic full-scenario initialization."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
import torch
from utils import run_junction_conflict_trials as legacy
s=legacy.s
ROOT=s.ROOT
OUT=ROOT/'results/curriculum_keep_20261116'
BASE=s.SOURCE

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    tmp.replace(path)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_agent():
    meta=s.t.read(ROOT/'results/junction_conflict_trials_20261116/runs/control/manifest.json')
    agent=s.a.b.PPO(obs_dim=63,act_dim=2,**meta['hp'])
    agent.load(str(BASE))
    return agent,meta

def make_env():
    env=s.make_env('control',evaluation=False,none=False)
    assert not env.lc_safety_enabled
    return env

def context(env,action):
    v=s.a.b.traci.vehicle
    nav=env.navigation_context();lane=v.getLaneIndex(env.ego_id)
    needed=env.navigation.offset(lane,nav['current_targets'])
    risk=env.safety_context(action)
    follower=env._neighbor(0,False);speed=v.getSpeed(env.ego_id)
    rear_ttc=(follower[0]+2.5)/(follower[1]-speed) if follower and follower[1]>speed else None
    return dict(edge=v.getRoadID(env.ego_id),lane=lane,speed=speed,
        gap=risk['gap'],lane_risk=risk['lane'],ttc=risk['ttc'],rear_ttc=rear_ttc,
        merge_risk=env.conflict_features()[0],needed=needed,
        constrained=bool(nav['current_targets']) and len(nav['current_targets'])<env.navigation.counts.get(v.getRoadID(env.ego_id),0),
        prepare=nav['distance'],front=env._neighbor(0,True),rear=follower)

def safe_context(x):
    return x['gap']<.15 and x['lane_risk']<.1 and x['merge_risk']<.05 and (x['ttc'] is None or x['ttc']>5) and (x['rear_ttc'] is None or x['rear_ttc']>3)

def collect(episodes):
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    agent,meta=source_agent()
    summaries=[];keep=[];safe=[]
    env=make_env()
    try:
        for i in range(episodes):
            route=f'route_{i%16+1:02d}';seed=910001+i
            dest=OUT/'data-episodes'/f'{route}-{seed}.json'
            if dest.exists():
                episode=json.loads(dest.read_text(encoding='utf8'))
                assert episode['parent_sha']==digest(BASE)
            else:
                env.warmup_steps=(4000,4500,5000)[i%3]
                obs,_=env.reset(seed=seed,options={'route':route});rows=[]
                while True:
                    act=agent.predict(obs,deterministic=True);ctx=context(env,act)
                    nxt,reward,done,cut,info=env.step(act)
                    alive=env.ego_id in s.a.b.traci.vehicle.getIDList()
                    after_speed=s.a.b.traci.vehicle.getSpeed(env.ego_id) if alive else None
                    after_lane=s.a.b.traci.vehicle.getLaneIndex(env.ego_id) if alive else None
                    after_edge=s.a.b.traci.vehicle.getRoadID(env.ego_id) if alive else None
                    rows.append(dict(t=(env.step_count-1)*env.step_length,obs=obs.tolist(),action=act.tolist(),reward=reward,context=ctx,after_speed=after_speed,after_lane=after_lane,after_edge=after_edge))
                    obs=nxt
                    if done or cut:
                        outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
                        break
                assert env.route_override_count==0
                episode=dict(route=route,seed=seed,warmup_steps=env.warmup_steps,parent_sha=digest(BASE),outcome=outcome,rows=rows,route_overrides=0)
                save(dest,episode)
            rows=episode['rows'];accepted=collections.Counter();safe_n=0
            if episode['outcome']=='arrival':
                for j,row in enumerate(rows):
                    ctx=row['context'];window=rows[max(0,j-25):min(len(rows),j+26)]
                    action=row['action'];toward=ctx['needed']*action[1]>0
                    same=row['after_edge']==ctx['edge'];changed=same and row['after_lane']!=ctx['lane']
                    correct=not ctx['constrained'] or ctx['needed']==0 or toward
                    smooth=safe_context(ctx) and all(safe_context(z['context']) for z in window)
                    if correct and smooth and row['t']>=2 and row['after_speed'] is not None:
                        category='lane_change' if changed and correct else 'preparation' if ctx['prepare'] is not None and ctx['prepare']<=300 else 'cruise'
                        if accepted[category]<160 and (category=='lane_change' or j%5==0):
                            keep.append(dict(obs=row['obs'],action=action,category=category,route=episode['route'],seed=episode['seed'],t=row['t']))
                            accepted[category]+=1
                    # Only retained safe successful actions; no invented teacher.
                    decel=row['after_speed'] is not None and row['after_speed']<ctx['speed']-.05
                    hazard=ctx['merge_risk']>0 or ctx['gap']>.05 or ctx['ttc'] is not None and ctx['ttc']<8
                    margin=all((z['context']['ttc'] is None or z['context']['ttc']>1.5) and (z['context']['rear_ttc'] is None or z['context']['rear_ttc']>2) and z['context']['lane_risk']<.2 for z in rows[j:min(len(rows),j+26)])
                    if margin and ((hazard and decel and action[0]<0) or (changed and toward and smooth)):
                        safe.append(dict(obs=row['obs'],action=action,category='yield' if hazard else 'safe_change',route=episode['route'],seed=episode['seed'],t=row['t']))
                        safe_n+=1
            summary=dict(route=episode['route'],seed=episode['seed'],warmup_steps=episode['warmup_steps'],outcome=episode['outcome'],steps=len(rows),keep=dict(accepted),safe=safe_n,
                merge_risk_steps=sum(x['context']['merge_risk']>0 for x in rows),front_risk_steps=sum(x['context']['ttc'] is not None and x['context']['ttc']<5 for x in rows),lane_decision_steps=sum(x['context']['constrained'] for x in rows))
            summaries.append(summary);save(OUT/'collection-progress.json',dict(completed=len(summaries),target=episodes,last=summary));print(summary,flush=True)
    finally:env.close()
    for name,data in [('keep',keep),('safe',safe)]:
        np.savez_compressed(OUT/f'{name}.npz',obs=np.asarray([x['obs'] for x in data],dtype=np.float32).reshape(-1,63),actions=np.asarray([x['action'] for x in data],dtype=np.float32).reshape(-1,2),category=np.asarray([x['category'] for x in data]))
        save(OUT/f'{name}-provenance.json',dict(parent_sha=digest(BASE),records=data,qualification='Automated temporal margin and route/outcome filters. Baseline is not a universal expert; source actions use student observation only.'))
    save(OUT/'data-manifest.json',dict(parent_sha=digest(BASE),episodes=summaries,keep=len(keep),safe=len(safe),keep_sha=digest(OUT/'keep.npz'),safe_sha=digest(OUT/'safe.npz'),training_seed_range=[910001,910000+episodes],development_seeds=[79101,79102,79103],final_reserved_seeds=list(range(991001,991021)),rules='Arrival episode + per-frame safety and +/-5s filter for keep; safe demonstrations require downstream margins and actual deceleration or safe route-directed change.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=32);a=p.parse_args();collect(a.episodes)
