"""Frozen-policy empty-road and reward activation diagnostics; no action override."""
import collections
import concurrent.futures as cf
import copy
import numpy as np
from utils import run_route_following_trials as runner
from utils import run_observation_reward_trials as t

OUT=t.ROOT/'results/route_following_audit_20261114'

def one(spec):
    condition,step,route,traffic=spec
    name=f'{condition}-{step}-{route}-{traffic}'
    dest=OUT/(name+'.json')
    if dest.exists():return t.read(dest)
    t.a.b.torch.set_num_threads(1);torch=t.a.b.torch;tr=t.a.b.traci;v=tr.vehicle
    directory=runner.OUT/'runs'/condition;m=t.read(directory/'manifest.json')
    for f,h in m['source_hashes'].items():assert t.a.b.digest(t.ROOT/f)==h,f
    model=directory/f'policy-{step}.pt'
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(model))
    env=t.make_env(condition,evaluation=True,none=traffic=='none')
    counters=collections.Counter();history=collections.deque(maxlen=100);probes=[];stable=0;last=None
    try:
        obs,_=env.reset(seed=19101,options={'route':route})
        for _ in range(4500):
            edge=v.getRoadID(env.ego_id);lane=v.getLaneIndex(env.ego_id);pos=v.getLanePosition(env.ego_id)
            speed=v.getSpeed(env.ego_id);remaining=tr.lane.getLength(v.getLaneID(env.ego_id))-pos
            nav=env.navigation_context();need=env.navigation.offset(lane,nav['current_targets'])
            action=agent.predict(obs,deterministic=True);command=env._quantize_lane_change(action[1])
            with torch.no_grad():
                prob=torch.softmax(agent.policy(torch.tensor(obs).unsqueeze(0))[2],-1).squeeze(0).tolist()
            if need and speed>.5 and env.step_count%5==0:
                hidden=obs.copy();hidden[31:34]=[0,-1,0];hidden[47:]=0
                with torch.no_grad():hp=torch.softmax(agent.policy(torch.tensor(hidden).unsqueeze(0))[2],-1).squeeze(0).tolist()
                idx=0 if need>0 else 2
                probes.append(dict(edge=edge,t=env.step_count*.2,needed=int(np.sign(need)),prob=prob,hidden_prob=hp,
                    correct_probability=prob[idx],hidden_correct_probability=hp[idx],choice_changed=int(np.argmax(prob)!=np.argmax(hp))))
            pre=copy.deepcopy(env.route_rewards)
            record=dict(t=env.step_count*.2,edge=edge,lane=lane,position=pos,speed=speed,remaining=remaining,nav=nav,
                        obs_nav=obs[31:34].tolist(),extra=obs[47:].tolist(),action=action.tolist(),prob=prob,leader=v.getLeader(env.ego_id,100))
            obs,reward,done,cut,info=env.step(action)
            bonus=info.get('route_selection_reward',0.)
            selection=maintenance=0.
            for k,s in env.route_rewards.items():
                prev=pre.get(k)
                maintenance+=s['paid']-(prev['paid'] if prev else 0)
                if prev:selection+=.1*max(prev['best']-s['best'],0)
                # First invocation can also improve; infer its positive selection remainder below.
            selection=max(selection,bonus-maintenance,0)
            penalty=min(0,bonus-selection-maintenance)
            counters.update(dict(steps=1,base_reward=reward-bonus,bonus_total=bonus,selection_total=selection,maintenance_total=maintenance,
                departure_penalty=penalty,selection_events=int(selection>1e-8),maintenance_steps=int(maintenance>1e-8),departure_events=int(penalty< -1e-8),nonzero_bonus_steps=int(abs(bonus)>1e-8)))
            if need:
                counters['wrong_lane_steps']+=1;counters['toward_requests']+=int(command*need>0)
            record.update(reward=reward,bonus=bonus,command_sent=info['lane_change_applied'])
            if env.ego_id in v.getIDList():
                record.update(after_lane=v.getLaneIndex(env.ego_id),after_edge=v.getRoadID(env.ego_id))
                if need and record['after_edge']==edge:counters['toward_changes']+=int((record['after_lane']-lane)*need>0)
            history.append(record)
            state=(edge,lane,round(pos,3));stable=stable+1 if speed<.05 and state==last else 0;last=state
            if done or cut:
                outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout';break
            if stable>=50:outcome='stationary_10s';break
        else:outcome='diagnostic_900s_cap'
        result=dict(condition=condition,checkpoint=step,route=route,traffic=traffic,seed=19101,outcome=outcome,seconds=env.step_count*.2,
            metrics=dict(counters),tail=list(history),sensitivity=probes,policy_sha=t.a.b.digest(model),source_commit=m['source_commit'],route_overrides=env.route_override_count)
        assert env.route_override_count==0
        t.save(dest,result);return result
    finally:env.close()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[(c,200000,f'route_{i:02}','none') for c in ('baseline','guidance','route_reward') for i in range(1,17)]
    specs += [('route_reward',100352,f'route_{i:02}','none') for i in range(1,17)]
    specs += [('route_reward',s,f'route_{i:02}','reference') for s in (100352,200000) for i in range(1,17)]
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(one,specs),1):
            t.save(OUT/'status.json',dict(completed=i,total=len(specs),stage='running'))
    t.save(OUT/'status.json',dict(completed=len(specs),total=len(specs),stage='complete'))

if __name__=='__main__':main()
