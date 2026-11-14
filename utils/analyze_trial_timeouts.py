"""Read-only replay of every timeout; stop after reproducing its stationary state."""
import collections
import concurrent.futures as cf
import json
import os
from utils import run_observation_reward_trials as t

OUT=t.ROOT/'results/observation_reward_timeout_audit_20261114'

def cases():
    return [r for p in sorted((t.OUT/'runs').glob('*/policy-*-evaluation.json'))
            for r in t.read(p) if r['outcome'] not in ('arrival','collision')]

def key(r):
    return '-'.join(str(r[k]).replace('.pt','') for k in ('condition','checkpoint','traffic','route','seed'))

def replay(r):
    dest=OUT/('extended' if r.get('_full') else 'cases')/(key(r)+'.json')
    if dest.exists():return t.read(dest)
    t.a.b.torch.set_num_threads(1)
    tr=t.a.b.traci;v=tr.vehicle
    directory=t.OUT/'runs'/r['condition']
    m=t.read(directory/'manifest.json')
    for f,h in m['source_hashes'].items():
        assert t.a.b.digest(t.ROOT/f)==h,('changed evaluation source',f)
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(directory/r['checkpoint']))
    env=t.make_env(r['condition'],evaluation=True,none=r['traffic']=='none')
    target=r['diagnostic']['samples'][-1]
    history=collections.deque(maxlen=300);stats=collections.Counter();matched=0
    events=[];previous_edge=None;reproduced=False;end='diagnostic_cap'
    try:
        obs,_=env.reset(seed=r['seed'],options={'route':r['route']})
        while env.step_count<(54000 if r.get('_full') else 4500):
            edge=v.getRoadID(env.ego_id);lane=v.getLaneIndex(env.ego_id);lid=v.getLaneID(env.ego_id)
            pos=v.getLanePosition(env.ego_id);speed=v.getSpeed(env.ego_id)
            nav=env.navigation_context();action=agent.predict(obs,deterministic=True)
            direction=env._quantize_lane_change(float(action[1]));desired=env.navigation.offset(lane,nav['current_targets'])
            record=dict(t=env.step_count*.2,edge=edge,lane=lane,position=pos,speed=speed,
                        remaining=tr.lane.getLength(lid)-pos,nav=nav,obs_nav=obs[31:34].tolist(),
                        action=action.tolist(),direction=direction,leader=v.getLeader(env.ego_id,100),
                        neighbors={str(d):[env._neighbor(d,True),env._neighbor(d,False)] for d in (-1,1)})
            if edge!=previous_edge:
                events.append(dict(record));previous_edge=edge
            if edge==target['edge']:
                stats['target_edge_steps']+=1
                if desired:
                    stats['wrong_lane_steps']+=1
                    stats['toward_request_steps' if direction*desired>0 else 'hold_steps' if direction==0 else 'away_request_steps']+=1
                    if direction*desired>0:
                        record['lane_change_state']=list(v.getLaneChangeState(env.ego_id,direction))
                if nav['ambiguous']:stats['ambiguous_steps']+=1
            before=obs
            obs,reward,done,cut,info=env.step(action)
            record.update(reward=reward,command_sent=bool(info['lane_change_applied']))
            if env.ego_id in v.getIDList():
                record.update(after_edge=v.getRoadID(env.ego_id),after_lane=v.getLaneIndex(env.ego_id),after_speed=v.getSpeed(env.ego_id))
                if edge==target['edge'] and desired and direction*desired>0:
                    stats['toward_sent_steps']+=int(info['lane_change_applied'])
                    stats['toward_actual_same_edge_steps']+=int(record['after_edge']==edge and record['after_lane']!=lane)
            history.append(record)
            same=(edge==target['edge'] and lane==target['lane_index'] and abs(pos-target['position'])<.15 and speed<.05)
            matched=matched+1 if same else 0
            if matched>=50:reproduced=True;end='matched_original_stop_10s';break
            if done or cut:end='collision' if info['collided'] else 'arrival' if info['arrived'] else 'other';break
        result=dict(case={k:r[k] for k in ('condition','checkpoint','traffic','route','seed')},
                    original_category=r['diagnostic']['category'],original_final=target,reproduced=reproduced,
                    diagnostic_end=end,seconds=env.step_count*.2,stats=dict(stats),entries=events,tail=list(history),
                    route_overrides=env.route_override_count,source_commit=m['source_commit'],
                    policy_sha=t.a.b.digest(directory/r['checkpoint']))
        assert env.route_override_count==0
        t.save(dest,result)
        return result
    finally:env.close()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        rows=cases();t.save(OUT/'manifest.json',dict(cases=len(rows),workers=6,training=False,action_override=False,
                   cap_seconds=900,match_seconds=10,note='Original 10800s timeout remains authoritative; replay ends at matching stop, not a new performance outcome.'))
        with cf.ProcessPoolExecutor(max_workers=6) as pool:
            for i,result in enumerate(pool.map(replay,rows),1):
                t.save(OUT/'status.json',dict(stage='replay',completed=i,total=len(rows)))
                print(i,key(result['case']),result['diagnostic_end'],flush=True)
        t.save(OUT/'status.json',dict(stage='complete',completed=len(rows),total=len(rows)))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
