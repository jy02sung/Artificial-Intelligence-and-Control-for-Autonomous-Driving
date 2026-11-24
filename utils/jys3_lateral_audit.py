"""Audit short lateral reversals against the immediate road choice, with paired replays."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile
import json
from utils.jys3_ppo_long import DIR,ARM,paired
from utils.jys3 import environment,torch
from utils.jys3_dagger_one import policy

def run(job):
    step,rep=job
    torch.set_num_threads(1)
    model=policy(paired.ck(ARM,step))
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        env=environment('high_real_ratio',11,Path(tmp))
        try:
            for route in range(1,17):
                name=f'route_{route:02d}'
                seed=paired.BASE['screen']+route
                env.ego_depart_lane=str(min(rep,env.counts[env.routes[name][0]]-1))
                def drive(skip=None):
                    obs,_=env.reset(seed=seed,options={'route':name})
                    last=None;cycles=[];moves=0;commands=Counter(); rewarded=0; invalid=0; reasons=Counter(); motion=[]
                    for t in range(1000):
                        edge,lane,path,idx,_,_,_,speed,remaining=env.snapshot
                        act=model.predict(obs,deterministic=True)
                        if skip and skip[0]<=t<=skip[1]:act[env.lane_change_index]=0
                        cmd=int(act[env.lane_change_index])
                        commands[cmd]+=1
                        if last and last['edge']!=edge:last=None
                        obs2,reward,done,cut,info=env.step(act)
                        post=env.snapshot
                        if cmd and not info['lane_change_applied']:
                            invalid+=1
                            reason='internal' if edge.startswith(':') else 'solid' if edge in env.solid else 'lane_absent' if not 0<=lane+cmd<env.counts.get(edge,0) else 'other'
                            reasons[f'{reason}|{edge}|{lane}|{cmd}']+=1
                        if post and post[0]==edge and post[1]!=lane and not edge.startswith(':'):
                            moves+=1
                            targets=env.links[edge][path[idx+1]] if idx+1<len(path) else []
                            event=dict(t=t,edge=edge,a=lane,b=post[1],m=remaining,v=speed,
                                       ready_before=lane in targets,ready_after=post[1] in targets,
                                       lead_gap_obs=float(obs[9])*env.visibility,
                                       lead_closing_obs=-float(obs[10])*env.vmax,reward=reward)
                            if last and last['a']==event['b'] and last['b']==event['a'] and t-last['t']<=25:
                                cycles.append([last,event])
                            motion.append(event)
                            last=event
                            rewarded+=int(reward>0)
                        obs=obs2
                        if done or cut:break
                    outcome='arrival' if info['arrived'] else 'collision' if info['collided'] else 'stall' if info['stalled'] else 'timeout'
                    return dict(route=route,rep=rep,outcome=outcome,steps=t+1,moves=moves,commands=dict(commands),invalid=invalid,reasons=dict(reasons),motion=motion,positive_move_rewards=rewarded,cycles=cycles)
                r=drive()
                if r['cycles']:
                    a,b=r['cycles'][0]
                    alt=drive((a['t'],b['t']))
                    r['without_first_cycle']=[alt['outcome'],alt['steps']]
                if r['moves']:
                    held=drive((0,999))
                    r['keep_lane']=[held['outcome'],held['steps']]
                rows.append(r)
        finally:env.close()
    return rows
if __name__=='__main__':
    import sys
    steps=[int(n) for n in sys.argv[1:]]
    with ProcessPoolExecutor(min(6,len(steps)*2)) as pool:
        groups=pool.map(run,[(n,r) for n in steps for r in range(2)])
        for n in steps:
            rows=next(groups)+next(groups)
            expected=json.loads((DIR/f'screen_{n}.json').read_bytes())
            assert [(r['outcome'],r['steps']) for r in rows]==[(r['outcome'],r['steps']) for r in expected]
            paired.save(DIR/f'lateral_{n}.json',rows)
