"""Match one successful run by route and departure lane to each failure, compare on the same road."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from . import local_audit as a

def read(folder):
    return [json.loads(p.read_bytes()) for p in sorted((a.OUT/folder).glob('*.json'))]

def control(task):
    fail, ref, edge, pos=task
    v=a.V; env=a.env
    obs,_=env.reset(seed=ref['seed'],options={'route':ref['route']})
    best=None
    while True:
        action=a.agent.predict(obs,deterministic=True)
        if v.getRoadID(env.ego_id)==edge:
            delta=abs(v.getLanePosition(env.ego_id)-pos)
            if best is None or delta<best['pos_difference']:
                best=dict(pos_difference=delta,**a.snapshot(obs,action,None))
        obs,_,done,cut,info=env.step(action)
        if done or cut:break
    assert info['arrived'] and not info['collided'] and env.step_count==ref['steps']
    return dict(route=ref['route'],incident_seed=fail['seed'],control_seed=ref['seed'],frame=best)

def main():
    by=defaultdict(list)
    for r in read('fresh'):
        if r['outcome']=='arrival':by[r['route'],r['start_lane']].append(r)
    tasks=[]; used=set()
    for r in read('replay'):
        h=r['history'][-1]
        candidates=by[r['route'],r['start_lane']]
        if not candidates:continue
        pick=min(candidates,key=lambda x:(x['seed'] in used,x['seed']))
        used.add(pick['seed'])
        tasks.append((r,pick,h['edge'],h['pos']))
    with ProcessPoolExecutor(6,initializer=a.setup) as pool: rows=list(pool.map(control,tasks))
    (a.OUT/'matched_controls.json').write_text(json.dumps(rows),encoding='utf8')
if __name__=='__main__':main()
