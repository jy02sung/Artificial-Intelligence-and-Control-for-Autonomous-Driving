"""BC v2. Give blocked merges room before the lane end and label on expert trajectories."""
import json
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from utils import jys3_collect_bc as source
from utils.jys3 import environment, m

OUT = m.ROOT / "results" / "jys3_bc_v2"
SEED = 11

def expert(obs, env):
    action = source.expert(obs, env)
    lane, count = np.rint(obs[32:34]*5).astype(int)
    links = obs[40:].reshape(5,2)[:count]
    eligible = links[:,int(obs[34]>0)] if obs[35]>=0 else links.max(axis=1)
    if not eligible.any() or eligible[lane] or action[1] or obs[38]:
        return action
    # Reserve longitudinal space while waiting for a safe lateral gap.
    remaining = 400.*float(obs[39])
    speed = float(obs[0])*env.vmax
    target = np.sqrt(2.*2.5*max(0.,remaining-14.))
    braking = np.clip((target-speed)/1.3,-env.max_decel,env.max_accel)
    action[0] = min(action[0], braking/(env.max_decel if braking<0 else env.max_accel))
    return action

def run(job):
    mode, rep, ns, base = job
    records=[]
    with tempfile.TemporaryDirectory() as d:
        env=environment('high_real_ratio',11,Path(d))
        if mode=='student':
            from utils.jys3 import HP, torch, OBS_DIM
            torch.set_num_threads(1)
            agent=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP)
            agent.policy.load_state_dict(torch.load(OUT/'bc.pt',weights_only=True)['policy'])
            agent.policy.eval()
        try:
            for n in ns:
                route=f'route_{n:02d}'
                seed=base+100*rep+n
                lane=min(rep%2,env.counts[env.routes[route][0]]-1)
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed,options={'route':route})
                states,labels=[],[]
                for t in range(1,1001):
                    action=(agent.predict(obs,deterministic=True) if mode=='student' else
                            source.expert(obs,env) if mode=='original' else expert(obs,env))
                    if mode=='collect':
                        states.append(obs)
                        labels.append(action)
                    obs,_,done,cut,info=env.step(action)
                    if done or cut:break
                outcome='arrival' if info['arrived'] else 'collision' if info['collided'] else 'stall' if info['stalled'] else 'timeout'
                row=dict(route=n,lane=lane,seed=seed,steps=t,outcome=outcome)
                if mode=='collect':
                    fname=f'{route}_{seed}.npz'
                    np.savez_compressed(OUT/'samples'/fname,obs=states,actions=labels)
                    row['file']=fname
                records.append(row)
        finally:env.close()
    return records

def sweep(mode,reps,base):
    jobs=[(mode,i,range(1,17),base) for i in range(reps)]
    return parallel(jobs)

def parallel(jobs):
    with ProcessPoolExecutor(6) as pool:
        return [r for block in pool.map(run,jobs) for r in block]

def save(name,records):
    (OUT/(name+'.json')).write_text(json.dumps(records,indent=2))
    print(name,dict(Counter(r['outcome'] for r in records)),flush=True)

if __name__=='__main__':
    import sys
    OUT.mkdir(exist_ok=True)
    task=sys.argv[1]
    if task=='probe':
        save('probe',parallel([('new',rep,[n],24112400) for rep,n in [(0,13),(2,14),(4,5),(4,13),(5,5)]]))
    elif task=='collect':
        (OUT/'samples').mkdir(exist_ok=True)
        save('train',sweep('collect',12,24112400))
    elif task in ('student','new','original'):
        save(task,sweep(task,4,26010000))
