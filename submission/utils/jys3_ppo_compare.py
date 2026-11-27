"""Paired jys3 PPO comparison; deterministic screening and separate independent confirmation."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from utils.jys3 import environment, HP, m, SCHEMA, ScheduledEpisodes, atomic_torch, torch
from utils.jys3_dagger_one import policy

OUT=m.ROOT/'results'/'jys3_ppo_comparison'
MODELS={'bc':'onehop_bc_v2.pt','bc_dagger':'onehop_dagger1.pt','dagger':'onehop_dagger_cold.pt'}
HARD=(5,13,14,16)
STEPS=(0,8192,16384,32768)
BASE={'screen':29110000,'hold':30110000}

def save(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)

def ck(arm,n):
    return OUT/arm/f'policy_{n}.pt'

def train(arm):
    torch.set_num_threads(1)
    d=OUT/arm
    agent=policy(m.ROOT/'models'/'jys3'/MODELS[arm])
    agent.policy.train()
    env=environment('high_real_ratio',11,d/'map')
    try:
        schedule=ScheduledEpisodes(env,11,d/'episodes.jsonl')
        obs,_=schedule.reset()
        stats=[0.,0]
        for target in (*STEPS,49152,65536):
            while agent.num_timesteps<target:
                obs,last,_=agent.collect_rollout(schedule,obs,ep_stats=stats,steps=2048,timeout_is_failure=True)
                adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
                agent.update(adv,ret)
                save(d/'progress.json',dict(step=agent.num_timesteps,time=time.time()))
            atomic_torch({'schema':SCHEMA,'policy':agent.policy.state_dict()},ck(arm,target))
            if target==32768:
                decision=d/'extend.json'
                while not decision.exists():time.sleep(1)
                if not json.loads(decision.read_bytes())['extend']:break
    finally:env.close()

def evaluate(job):
    arm,n,part,rep=job
    torch.set_num_threads(1)
    agent=policy(ck(arm,n))
    rows=[]
    with tempfile.TemporaryDirectory() as folder:
        env=environment('high_real_ratio',11,Path(folder))
        try:
            for route in range(1,17):
                name=f'route_{route:02d}'
                # Pair the traffic seed across requested starting lanes.
                seed=BASE[part]+100*(rep//2)+route
                lane=min(rep%2,env.counts[env.routes[name][0]]-1)
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed,options={'route':name})
                start=[env.snapshot[0],env.snapshot[1],env.snapshot[7],env.snapshot[-1]]
                for t in range(1,1001):
                    obs,_,done,cut,info=env.step(agent.predict(obs,deterministic=True))
                    if done or cut:break
                outcome='arrival' if info['arrived'] else 'collision' if info['collided'] else 'stall' if info['stalled'] else 'timeout' if cut else 'other'
                rows.append(dict(route=route,case=rep,seed=seed,lane=lane,start=start,outcome=outcome,steps=t))
        finally:env.close()
    return rows

def score(rows):
    c=Counter(r['outcome'] for r in rows)
    return (c['arrival'],-c['collision'],sum(r['route'] in HARD and r['outcome']=='arrival' for r in rows))

def run():
    OUT.mkdir(parents=True,exist_ok=False)
    for arm in MODELS:(OUT/arm).mkdir()
    save(OUT/'protocol.json',dict(seed=11,HP=HP,inputs={a:hashlib.sha256((m.ROOT/'models'/'jys3'/f).read_bytes()).hexdigest() for a,f in MODELS.items()},steps=STEPS,hard=HARD,bases=BASE,extension='arrivals AND hard-route arrivals increase from 16384 to 32768',selection='arrivals, fewer collisions, hard-route arrivals, earliest checkpoint'))
    jobs={}; submitted=set(); shards={}; merged={}; selected={}; extended={}
    with ProcessPoolExecutor(3) as training, ProcessPoolExecutor(8) as evaluation:
        training_jobs={arm:training.submit(train,arm) for arm in MODELS}
        while True:
            for f in training_jobs.values():
                if f.done(): f.result()
            for arm in MODELS:
                for n in (*STEPS,49152,65536):
                    if not ck(arm,n).exists():continue
                    parts=['screen']
                    if arm in selected and n in {0,selected[arm],best_ppo[arm]}:parts.append('hold')
                    for part in parts:
                        for rep in range(2 if part=='screen' else 4):
                            key=(arm,n,part,rep)
                            if key not in submitted:
                                submitted.add(key)
                                jobs[evaluation.submit(evaluate,key)]=key
            if jobs:
                ready,_=wait(jobs,timeout=1,return_when=FIRST_COMPLETED)
                for f in ready:
                    a,n,p,rep=jobs.pop(f)
                    key=(a,n,p)
                    shards.setdefault(key,{})[rep]=f.result()
                    count=2 if p=='screen' else 4
                    if len(shards[key])==count:
                        rows=[r for i in range(count) for r in shards[key][i]]
                        merged[key]=rows
                        save(OUT/a/f'{p}_{n}.json',rows)
            else:time.sleep(.5)
            for arm in MODELS:
                if arm not in extended and all((arm,n,'screen') in merged for n in STEPS):
                    before,after=[score(merged[arm,n,'screen']) for n in (16384,32768)]
                    extended[arm]=after[0]>before[0] and after[2]>before[2]
                    save(OUT/arm/'extend.json',dict(extend=extended[arm],before=before,after=after))
                if arm in extended and arm not in selected:
                    ns=(*STEPS,49152,65536) if extended[arm] else STEPS
                    if all((arm,n,'screen') in merged for n in ns):
                        rank=lambda n:(*score(merged[arm,n,'screen']),-n)
                        selected[arm]=max(ns,key=rank)
                        if len(selected)==1:best_ppo={}
                        best_ppo[arm]=max(ns[1:],key=rank)
                        save(OUT/arm/'selection.json',dict(selected=selected[arm],best_positive=best_ppo[arm]))
            if len(selected)==3 and all((a,n,'hold') in merged for a in MODELS for n in {0,selected[a],best_ppo[a]}):break
    for part in ('screen','hold'):
        keys=[[(r['case'],r['seed'],r['route'],r['start']) for r in rs] for (_,_,p),rs in merged.items() if p==part]
        assert all(k==keys[0] for k in keys)
    save(OUT/'summary.json',dict(selected=selected,best_positive=best_ppo,extended=extended,scores={f'{a}_{n}_{p}':dict(Counter(r['outcome'] for r in rows)) for (a,n,p),rows in merged.items()}))

if __name__=='__main__':run()
