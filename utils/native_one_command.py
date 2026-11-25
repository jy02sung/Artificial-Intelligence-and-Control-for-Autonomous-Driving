"""Intervene once on a physically existing adjacent lane; retain PPO acceleration."""
import json, os
from concurrent.futures import ProcessPoolExecutor
from utils.native_motion_audit import BASE, EDGES, OUT, agent_for, make_c, torch
from utils.completion_focus_v1 import OUT as PRIOR, VALID

def run(job):
    base, rows, second=job
    torch.set_num_threads(1)
    a=agent_for(33)
    a.load(str(BASE))
    a.policy.eval()
    env=make_c('high_real_ratio','C0',33,OUT/'intervention_maps'/str(os.getpid()))
    out=[]
    try:
        for ref in rows:
            route=ref['route']
            n=int(route[-2:])
            env.ego_depart_lane=str(min(ref['lane_case'],env.counts[env.routes[route][0]]-1))
            obs,_=env.reset(seed=ref['sim_seed'],options={'route':route})
            report=dict(route=n,seed=ref['sim_seed'],issued=False,crossed=False)
            for _ in range(3000):
                edge,l,path,i,j,d,sides,v,rem=env.snapshot
                action=a.predict(obs,deterministic=True)
                intervene=False
                if edge==EDGES[n] and not report['issued']:
                    targets=env.links[edge][path[i+1]]
                    if l not in targets:
                        direction=1 if min(targets)>l else -1
                        report.update(issued=True,m=rem,v=v,original=env._quantize_lane_change(action[env.lane_change_index]),gaps=sides[direction],expected=path[i+1])
                        action=action.copy()
                        action[env.lane_change_index]=direction
                        intervene=True
                if second and report['issued'] and not intervene and not report.get('second') and edge==EDGES[n] and l==2:
                    action=action.copy()
                    action[env.lane_change_index]=-1
                    report['second']=True
                obs,_,done,cut,info=env.step(action)
                post=env.snapshot
                if intervene:
                    report['changed']=bool(post and post[0]==edge and post[1]==l+direction)
                if post and post[0]==report.get('expected'):
                    report['crossed']=True
                if done or cut:
                    report.update(last_edge=edge,last_lane=l,last_m=rem,last_v=v,last_targets=sorted(env.links.get(edge,{}).get(path[i+1],())) if i+1<len(path) else [])
                    report['outcome']='arrival' if info['arrived'] else 'collision' if info['collided'] else 'timeout'
                    break
            out.append(report)
        return out
    finally:
        env.close()

if __name__=='__main__':
    import sys
    second='--second' in sys.argv
    jobs=[]
    for base in VALID:
        rows=json.loads((PRIOR/f'baseline_{base}.json').read_bytes())
        jobs.append((base,[r for r in rows if r['outcome']=='timeout' and (not second or r['route']=='route_13')], second))
    with ProcessPoolExecutor(4) as pool:
        result=[r for group in pool.map(run,jobs) for r in group]
    (OUT/('two_commands_13.json' if second else 'one_command.json')).write_text(json.dumps(result,indent=2))
