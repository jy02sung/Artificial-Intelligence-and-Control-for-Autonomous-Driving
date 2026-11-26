"""Paired S2 replacement trial with a 10-second clear-front stopped terminal."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import msvcrt

from utils import jys3_physical_scratch as prior

v3 = prior.v3
ARM = v3.ARMS[1]
ROOT = prior.ROOT.parent / "clear_stop_10s"
SEEDS = (11,22)


class Env(prior.physical.Env):
    progress_cap = 3.
    off_route_penalty = -4.

    def reset(self, **kwargs):
        self.clear_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, cut, info = super().step(action)
        if not (done or cut):
            gap = info['gap']
            clear = info['speed'] < .5 and (gap is None or gap > 25.)
            self.clear_steps = self.clear_steps + 1 if clear else 0
            if self.clear_steps == 50:
                reward -= 5.
                self.sums['clear_stop'] -= 5.
                info['outcome'] = 'clear_stop'
                done = True
                if self.outcome_log is not None:
                    record = dict(route=self.episode_route, traffic_seed=self.current_seed,
                                  outcome='clear_stop', steps=self.sim.step_count, off_route=None,
                                  edge=self.snapshot[0], lane=self.snapshot[1], gap=gap)
                    with self.outcome_log.open('a',encoding='utf8') as fp:
                        print(json.dumps(record),file=fp)
        return obs,reward,done,cut,info


def check():
    import numpy as np
    with TemporaryDirectory() as tmp:
        env = Env(ARM,Path(tmp),11)
        try:
            assert env.step_length==.2 and v3.j.COLLISION_REWARD==-7 and v3.j.TIMEOUT_REWARD==-5
            assert env.progress_cap==3 and v3.j.PASS_REWARD==1 and v3.j.ARRIVAL_REWARD==5 and env.off_route_penalty==-4
            env.reset(seed=41110001,options={'route':'route_01'})
            for _ in range(1000):
                _,r,done,cut,i=env.step(np.array([-1.,0.],dtype=np.float32))
                if done or cut:
                    assert done and not cut and i['outcome']=='clear_stop' and env.clear_steps==50
                    assert abs(r+5)<1e-6 and env.sums['clear_stop']==-5
                    break
            else:
                raise AssertionError('clear stop missing')
        finally: env.close()


def initialize(seeds):
    for seed in seeds:
        d=ROOT/f'{ARM}_{seed}'
        d.mkdir(parents=True,exist_ok=True)
        model=v3.core.agent(seed)
        initial=v3.v1.digest(model)
        assert initial==v3.read(prior.ROOT/f'{ARM}_{seed}'/'initial.json')['sha256']
        if not (d/'policy_0.pt').exists():
            v3.j.atomic_torch({'policy':model.policy.state_dict()},d/'policy_0.pt')


def rows(root,seeds,step,panel):
    return [r for seed in seeds for r in v3.v2.snapshot_rows(root,ARM,seed,step,panel)]


def counts(root,seeds,step,panel):
    a=rows(root,seeds,step,panel)
    assert len(a)==64*len(seeds)
    return Counter(r['outcome'] for r in a)


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'manager.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        check()
        initialize(SEEDS)
        with ProcessPoolExecutor(6) as pool:
            control=ROOT/'frozen_original'
            jobs=[]
            for seed in SEEDS:
                d=control/f'{ARM}_{seed}'
                d.mkdir(parents=True,exist_ok=True)
                for rep in range(4):
                    if not (d/f'eval_{v3.SCREEN}_300000_{rep}.json').exists():
                        jobs.append(pool.submit(v3.eval_job,control,ARM,seed,300000,rep,v3.SCREEN,
                                    env_type=Env,checkpoint=prior.ROOT/f'{ARM}_{seed}'/'policy_300000.pt'))
            v3.screen_stage((ARM,),SEEDS,300000,ROOT,pool,from_stage1=False,env_type=Env)
            for fut in jobs: fut.result()
            snapshots={s:counts(ROOT,SEEDS,s,v3.SCREEN) for s in (200000,250000,300000)}
            c=snapshots[300000]
            passed=(c['timeout']/128<.05 and c['arrival']/128>=.45 and
                    c['arrival']>(snapshots[200000]['arrival']+snapshots[250000]['arrival'])/2)
            v3.save(ROOT/'stage1_decision.json',dict(promoted=passed,screen=snapshots,
                    frozen=counts(control,SEEDS,300000,v3.SCREEN),original=counts(prior.ROOT,SEEDS,300000,v3.SCREEN)))
            if not passed:return
            seeds=(11,22,33)
            initialize(seeds)
            v3.screen_stage((ARM,),seeds,600000,ROOT,pool,from_stage1=False,env_type=Env)
            def score(n):
                c=counts(ROOT,seeds,n,v3.SCREEN)
                return c['arrival'],-c['collision'],-n
            best=max(v3.MARKS,key=score)
            jobs=[]
            for n in sorted({best,600000}):
                for panel in (v3.PAIRED,v3.FRESH,v3.NOVEL):
                    for seed in seeds:
                        for rep in range(4):
                            if not (ROOT/f'{ARM}_{seed}'/f'eval_{panel}_{n}_{rep}.json').exists():
                                jobs.append(pool.submit(v3.eval_job,ROOT,ARM,seed,n,rep,panel,env_type=Env))
            for fut in jobs:fut.result()
            v3.save(ROOT/'stage2_final.json',dict(best=best,panels={
                f'{n}_{panel}':counts(ROOT,seeds,n,panel) for n in sorted({best,600000})
                for panel in (v3.PAIRED,v3.FRESH,v3.NOVEL)}))


if __name__=='__main__':
    run()
