"""Transfer the jys2 C0 optimizer/exploration and timeout handling to the 47-input student."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv,json,math,random,shutil,sys
import numpy as np
from . import local_scratch_driver as d

BASE=d.OUT
OUT=BASE.parent/'scratch_c0_update_20261124'
def init_eval(name):
    d.OUT=OUT
    d.init(name,False)

def train():
    OUT.mkdir(exist_ok=True)
    d.OUT=OUT
    shutil.copyfile(BASE/'dagger2.pt',d.weights('dagger2'))
    d.init('dagger2',True)
    a=d.agent
    for group in a.optimizer.param_groups:group['lr']=1e-4
    a.lane_entropy_coef=.002
    assert d.env.rw['collision_penalty']==15 and a.n_epochs==10 and a.minibatch_size==64
    d.save('protocol.json',dict(source='jys2 C0',source_branch_commit='fb56565',parent_sha256=d.digest(d.weights('dagger2')),lr=.0001,lane_entropy_coef=.002,timeout='original PPO collect_rollout bootstrap',collision=-15,updates=16,total_steps=32768))
    rng=random.Random(d.SEED+3);d.torch.manual_seed(d.SEED+3);np.random.seed(d.SEED+3)
    sched=d.cases(198000000,40);blocks=[sched[i:i+16] for i in range(0,len(sched),16)]
    for b in blocks:rng.shuffle(b)
    episodes=iter(sum(blocks,[]))
    class Scheduled:
        def reset(self):
            route,seed=next(episodes)
            return d.env.reset(seed=seed,options={'route':route})
        def step(self,action):return d.env.step(action)
    env=Scheduled();obs,_=env.reset();stats=[0.,0];rows=[]
    try:
        for n in range(1,17):
            obs,last,_=a.collect_rollout(env,obs,ep_stats=stats,steps=2048)
            adv,ret=a.buffer.compute_gae(last,a.gamma,a.gae_lambda)
            metrics=a.update(adv,ret);assert all(map(math.isfinite,metrics.values()))
            rows.append(dict(update=n,steps=n*2048,**metrics))
            if n in (4,8,16):a.save(str(d.weights(f'ppo_{n}')))
            d.save('updates.json',rows)
    finally:d.env.close()

def evaluate():
    d.OUT=OUT
    name='ppo_16'
    cases=d.cases(200000000,100)
    with ProcessPoolExecutor(d.WORKERS,initializer=init_eval,initargs=(name,)) as pool:
        rows=list(pool.map(d.episode,[(r,s,-1,None) for r,s in cases],chunksize=2))
    before=json.loads((BASE/'ppo_16_holdout.json').read_bytes())
    assert [(r['route'],r['seed']) for r in rows]==cases==[(r['route'],r['seed']) for r in before['rows']]
    out=dict(rows=rows,counts=dict(Counter(x['outcome'] for x in rows)),sha256=d.digest(d.weights(name)))
    d.save('ppo_16_holdout.json',out);d.save('paired.json',dict(before=before['counts'],after=out['counts'],diff=d.paired(before,out)))
if __name__=='__main__':
    train() if sys.argv[1]=='train' else evaluate()
