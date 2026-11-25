"""Isolate the second fork with matched synthetic starts on lane 2."""
from concurrent.futures import ProcessPoolExecutor
import itertools,os,json
from utils.route120_short import OUT,SEEDS,original,make,old,agent_for,torch,traci,save

def run(job):
    arm,seed=job
    torch.set_num_threads(1)
    a=agent_for(33)
    a.load(str((old.OUT if arm=='original' else OUT)/'focus'/'policy_20000.pt'))
    a.policy.eval()
    env=(original if arm=='original' else make)('high_real_ratio','C0',33,OUT/'fork_maps'/str(os.getpid()))
    rows=[]
    try:
        for n in (13,14):
            rt=f'route_{n}'
            env.ego_depart_lane='0'
            env.reset(seed=seed,options={'route':rt})
            edge='1063527748'
            lid=edge+'_2'
            traci.vehicle.moveTo(env.ego_id,lid,traci.lane.getLength(lid)-100)
            obs=env._get_obs()
            start=env.snapshot
            target=env.routes[rt][start[3]+1]
            assert start[0]==edge and start[1]==2 and abs(start[-1]-100)<.01
            out='timeout'
            for _ in range(3000):
                obs,_,done,cut,info=env.step(a.predict(obs,deterministic=True))
                if info['collided']:
                    out='collision'
                    break
                if env.snapshot and env.snapshot[0]==target:
                    out='passed'
                    break
                if done or cut:
                    break
            rows.append(dict(route=n,seed=seed,arm=arm,outcome=out,start=[start[0],start[1],start[-1],start[7],start[6]],expected=target))
    finally:
        env.close()
    return rows
if __name__=='__main__':
    with ProcessPoolExecutor(4) as pool:
        rows=[r for block in pool.map(run,itertools.product(('original','route120'),SEEDS)) for r in block]
    for s in SEEDS:
        group=[r for r in rows if r['seed']==s]
        assert all(r['start']==group[0]['start'] for r in group)
    save(OUT/'isolated_fork.json',rows)
