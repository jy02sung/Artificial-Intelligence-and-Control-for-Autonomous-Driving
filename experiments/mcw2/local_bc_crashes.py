"""Deterministically replay terminal incidents and record the preceding policy inputs."""
from collections import Counter,deque
import json,sys
from . import local_bc_ppo as e
def replay(name, stage):
    e.setup(e.D/(name+'.pt'))
    cases=[r for r in json.loads((e.D/(name+'_'+stage+'.json')).read_bytes())['rows'] if r['outcome']=='collision']
    records=[]
    for row in cases:
        obs,_=e.env.reset(seed=row['seed'],options={'route':row['route']})
        frames=deque(maxlen=6)
        while True:
            action=e.agent.predict(obs,True)
            if e.env.step_count>=row['steps']-6:
                cmd=int(action[1]);k=2-cmd
                frames.append(dict(step=e.env.step_count,cmd=cmd,acc=float(action[0]),lane=e.v.getLaneID('ego'),speed=e.v.getSpeed('ego'),
                     front=float(obs[9]*120),target_front=float(obs[1+4*k]*120),target_rear=float(-obs[3+4*k]*120)))
            obs,_,dn,cut,info=e.env.step(action)
            if dn or cut:
                assert info['collided'] and e.env.step_count==row['steps']
                contacts=e.a.collisions();records.append(dict(row=row,frames=list(frames),contacts=contacts));break
    print(name,len(cases),Counter(f['contacts'][0]['type'] if f['contacts'] else '?' for f in records),Counter(bool(f['frames'][-1]['cmd']) for f in records),flush=True)
    e.x.write_json(e.D/(name+'_'+stage+'_crashes.json'),records);e.env.close()
if __name__=='__main__':replay(*sys.argv[1:])
