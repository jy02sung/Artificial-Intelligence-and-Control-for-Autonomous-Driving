"""Extend the seeded pure PPO trajectory, screen by fixed road cases and confirm on fresh seeds."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import os
os.environ["IC_LIBSUMO"]="1"
from utils import jys3_ppo_compare as paired
from utils.jys3 import OBS_DIM, SCHEMA, HP, torch, m, environment, ScheduledEpisodes, atomic_torch

ARM='ppo_300k'
DIR=paired.OUT/ARM
OLD=paired.OUT/'ppo_only'
END=300000
MARKS=[0,8192,16384,*range(32768,END,16384),END]

def counts(rows):
    return dict(Counter(r['outcome'] for r in rows))

def evaluation(job):
    step,part,rep=job
    if part=='fresh':
        paired.BASE['fresh']=32110000
    return paired.evaluate((ARM,step,part,rep))

def submit(pool,n,part,size):
    return [pool.submit(evaluation,(n,part,i)) for i in range(size)]

def finish(fs,n,part):
    rows=[r for f in fs for r in f.result()]
    paired.save(DIR/f'{part}_{n}.json',rows)
    return rows

def main():
    DIR.mkdir(exist_ok=False)
    paired.save(DIR/'protocol.json',dict(seed=11,steps=MARKS,hp=HP,screen_base=paired.BASE['screen'],fresh_base=32110000,selection='most arrivals, fewest crashes, most arrivals on routes 5/13/14/16, earliest'))
    torch.set_num_threads(1)
    agent=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP)
    env=environment('high_real_ratio',11,DIR/'map')
    schedule=ScheduledEpisodes(env,11,DIR/'episodes.jsonl')
    pending={}; screens={}; checks={}
    try:
        obs,_=schedule.reset()
        stats=[0.,0]
        with ProcessPoolExecutor(6) as pool:
            for target in MARKS:
                while agent.num_timesteps<target:
                    obs,last,_=agent.collect_rollout(schedule,obs,ep_stats=stats,steps=min(2048,target-agent.num_timesteps),timeout_is_failure=True)
                    adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
                    agent.update(adv,ret)
                    paired.save(DIR/'progress.json',{'step':agent.num_timesteps})
                state=agent.policy.state_dict()
                atomic_torch(dict(schema=SCHEMA,policy=state),paired.ck(ARM,target))
                if target<=32768:
                    past=torch.load(OLD/f'policy_{target}.pt',weights_only=True)['policy']
                    checks[target]=all(torch.equal(v,past[k]) for k,v in state.items())
                    paired.save(DIR/'reproduction.json',checks)
                if checks.get(target):
                    screens[target]=json.loads((OLD/f'screen_{target}.json').read_bytes())
                    paired.save(DIR/f'screen_{target}.json',screens[target])
                else:
                    pending[target]=submit(pool,target,'screen',2)
                for n in list(pending):
                    if all(f.done() for f in pending[n]):
                        screens[n]=finish(pending.pop(n),n,'screen')
            for n,fs in pending.items():screens[n]=finish(fs,n,'screen')
            selected=max(MARKS,key=lambda n:(*paired.score(screens[n]),-n))
            fresh={n:submit(pool,n,'fresh',4) for n in sorted({0,32768,selected,END})}
            validation={n:finish(fs,n,'fresh') for n,fs in fresh.items()}
        for results in (screens,validation):
            keys=[[(r['case'],r['route'],r['seed'],r['start']) for r in rows] for rows in results.values()]
            assert all(k==keys[0] for k in keys)
        paired.save(DIR/'summary.json',dict(selected=selected,screen={n:counts(screens[n]) for n in MARKS},fresh={n:counts(r) for n,r in validation.items()}))
    finally:env.close()
if __name__=='__main__':main()
