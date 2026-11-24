"""Add a random-initialization control and validate all four endpoints at 32768."""
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from utils import jys3_ppo_compare as paired
from utils.jys3 import OBS_DIM, SCHEMA, HP, torch, m

ARM="ppo_only"
DIRECTORY=paired.OUT/ARM

def train():
    from utils.jys3 import environment,ScheduledEpisodes,atomic_torch
    torch.set_num_threads(1)
    agent=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP)
    env=environment("high_real_ratio",11,DIRECTORY/"map")
    try:
        schedule=ScheduledEpisodes(env,11,DIRECTORY/"episodes.jsonl")
        obs,_=schedule.reset()
        stats=[0.,0]
        for target in paired.STEPS:
            while agent.num_timesteps<target:
                obs,last,_=agent.collect_rollout(schedule,obs,ep_stats=stats,steps=2048,timeout_is_failure=True)
                adv,returns=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
                agent.update(adv,returns)
                paired.save(DIRECTORY/"progress.json",{"step":agent.num_timesteps})
            atomic_torch(dict(schema=SCHEMA,policy=agent.policy.state_dict()),paired.ck(ARM,target))
    finally:env.close()

def launch(pool,arm,step,part):
    return [pool.submit(paired.evaluate,(arm,step,part,i)) for i in range(2 if part=="screen" else 4)]

def finish(futures,arm,step,part):
    rows=[r for f in futures for r in f.result()]
    ref=json.loads((paired.OUT/"bc"/f"{part}_0.json").read_bytes())
    def keys(rs):
        return [(r['case'],r['route'],r['seed'],r['lane'],r['start']) for r in rs]
    assert keys(rows)==keys(ref)
    paired.save(paired.OUT/arm/f"{part}_{step}.json",rows)
    return rows

def main():
    DIRECTORY.mkdir(exist_ok=False)
    with ProcessPoolExecutor(1) as trainer,ProcessPoolExecutor(8) as pool:
        task=trainer.submit(train)
        old={a:launch(pool,a,32768,"hold") for a in paired.MODELS}
        task.result()
        screens={s:launch(pool,ARM,s,"screen") for s in paired.STEPS}
        rows={s:finish(fs,ARM,s,"screen") for s,fs in screens.items()}
        selected=max(paired.STEPS,key=lambda s:(*paired.score(rows[s]),-s))
        holds={s:launch(pool,ARM,s,"hold") for s in sorted({0,selected,32768})}
        table={a:dict(Counter(r['outcome'] for r in finish(fs,a,32768,'hold'))) for a,fs in old.items()}
        cold={s:dict(Counter(r['outcome'] for r in finish(fs,ARM,s,'hold'))) for s,fs in holds.items()}
    table[ARM]=cold[32768]
    paired.save(DIRECTORY/'summary.json',dict(initial='random',seed=11,initial_sha256=hashlib.sha256(paired.ck(ARM,0).read_bytes()).hexdigest(),selected=selected,screen={s:dict(Counter(r['outcome'] for r in v)) for s,v in rows.items()},hold=cold,equal_budget_hold=table))

if __name__=="__main__":main()
