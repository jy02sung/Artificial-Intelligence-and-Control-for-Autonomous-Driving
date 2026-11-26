"""Frozen-actor critic warm-up followed by reference-regularized PPO on corrected forks."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from utils import jys3_physical_imitation_ppo as base

j = base.v3.j
ROOT = base.ROOT.with_name('jys3_physical_imitation_ppo_anchor')
WARM = 20000
BETA = 1.
HOLDOUT = 81260100


class Env(base.trial.prior.physical.Env):
    progress_cap = 6.
    off_route_penalty = -8.

    def reset(self, **kwargs):
        self.clear_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, cut, info = super().step(action)
        if not (done or cut):
            clear = info['speed'] < 2. and (info['gap'] is None or info['gap'] > 25.)
            self.clear_steps = self.clear_steps + 1 if clear else 0
            if clear:
                self.sums['idle_clear'] -= .02
                reward -= .02
            if self.clear_steps == 50:
                self.sums['clear_stop'] -= 7.
                reward -= 7.
                info['outcome'] = 'clear_stop'
                done = True
        return obs,reward,done,cut,info


@contextmanager
def environment():
    with TemporaryDirectory() as tmp:
        meta = base.imitation.data.bc.environment('high_real_ratio',11,Path(tmp)/'meta')
        env = Env(base.trial.ARM,Path(tmp)/'env',11)
        env.sim.signs,env.sim.road_flags = meta.signs,meta.road_flags
        j.ARRIVAL_REWARD,j.PASS_REWARD,j.COLLISION_REWARD,j.TIMEOUT_REWARD = 10.,2.5,-10.,-8.
        try:
            yield env
        finally:
            env.close()


def model_at(step):
    return base.imitation.dagger.policy(base.checkpoint(step, ROOT))


def evaluate(step, panel, rep):
    torch.set_num_threads(1)
    model = model_at(step)
    rows = []
    panel_seed = HOLDOUT if panel=='holdout' else base.imitation.PANELS[panel]
    with environment() as env:
        for route in range(1,17):
            seed = panel_seed + rep*100 + route
            lane = min(rep%2,env.counts[env.routes[f'route_{route:02d}'][0]]-1)
            _,row=base.imitation.replay.drive(env,model,route,seed,lane,True)
            rows.append(dict(**row,route=route,seed=seed,case=rep))
    return rows


def save_model(name, model):
    j.atomic_torch(dict(schema=base.imitation.dagger.SCHEMA,policy=model.policy.state_dict()),ROOT/name)


def update_log(phase,step,metrics,env):
    record=dict(phase=phase,step=step,metrics=metrics,rewards=dict(env.sums))
    with (ROOT/'updates.jsonl').open('a') as fp:
        print(json.dumps(record,allow_nan=False),file=fp)


def warm_up(model, reference):
    for name, par in model.policy.named_parameters():
        par.requires_grad_(name.startswith('v_head.'))
    optimizer=torch.optim.Adam(model.policy.v_head.parameters(),lr=2e-5)
    with environment() as env:
        episodes=base.Episodes(env,111,ROOT/'warm_episodes.jsonl')
        obs,_=episodes.reset()
        stats=[0.,0]
        while model.num_timesteps<WARM:
            env.sums.clear()
            count=min(2048,WARM-model.num_timesteps)
            obs,last,_=model.collect_rollout(episodes,obs,ep_stats=stats,steps=count,timeout_is_failure=True)
            _,targets=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
            states,_,_=model.buffer.to_tensors(model.device)
            target=torch.from_numpy(targets)
            with torch.no_grad():
                features=model.policy.trunk(states)
                assert torch.equal(model.policy.det_action(states),reference.det_action(states))
            def loss(ids):
                return (model.policy.v_head(features[ids]).squeeze(-1)-target[ids]).square().mean()
            before=float(loss(slice(None)).detach())
            for _ in range(10):
                for ids in np.array_split(np.random.permutation(count),int(np.ceil(count/64))):
                    optimizer.zero_grad()
                    loss(ids).backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.v_head.parameters(),.5)
                    optimizer.step()
            update_log('critic',model.num_timesteps,dict(mse_before=before,mse_after=float(loss(slice(None)).detach())),env)
    for key, val in model.policy.state_dict().items():
        if not key.startswith('v_head.'):
            assert torch.equal(val,reference.state_dict()[key]),key
    assert not torch.equal(model.policy.v_head.weight,reference.v_head.weight)
    save_model('critic_20000.pt',model)
    model.policy.requires_grad_(True)


def main():
    torch.set_num_threads(1)
    assert hashlib.sha256(base.INITIAL.read_bytes()).hexdigest()==base.HASH
    ROOT.mkdir(exist_ok=False)
    model=model_at(0)
    reference=model_at(0).policy.requires_grad_(False).eval()
    model.policy.train()
    model.gamma=.995
    model.lane_entropy_coef=model.entropy_coef=0.
    model.optimizer.param_groups[0]['lr']=2e-5
    jobs={}
    def queue(pool,step,panels):
        for panel in panels:
            jobs[step,panel]=[pool.submit(evaluate,step,panel,r) for r in range(4)]
    def finish(step, panels):
        rows={p:[r for f in jobs[step,p] for r in f.result()] for p in panels}
        assert all(len(v)==64 for v in rows.values())
        return rows
    panels=base.imitation.PANELS
    with ProcessPoolExecutor(6) as pool:
        queue(pool,0,panels)
        warm_up(model,reference)
        baseline=finish(0,panels)
        base.v3.save(ROOT/'eval_0.json',baseline)
        selections={0:baseline}
        with environment() as env:
            episodes=base.Episodes(env,11,ROOT/'ppo_episodes.jsonl')
            obs,_=episodes.reset()
            stats=[0.,0]
            for step in range(2048,65537,2048):
                env.sums.clear()
                obs,last,_=model.collect_rollout(episodes,obs,ep_stats=stats,steps=2048,timeout_is_failure=True)
                adv,ret=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
                metrics=model.update(adv,ret,reference_policy=reference,reference_coef=BETA)
                assert model.num_timesteps==WARM+step
                update_log('ppo',step,metrics,env)
                if step%base.INTERVAL:
                    continue
                save_model(f'policy_{step}.pt',model)
                queue(pool,step,panels)
                for mark in range(base.INTERVAL,step+1,base.INTERVAL):
                    if mark not in selections and (mark==step==32768 or all(f.done() for p in panels for f in jobs[mark,p])):
                        selections[mark]=finish(mark,panels)
                        base.v3.save(ROOT/f'eval_{mark}.json',selections[mark])
                if step==32768:
                    decision=base.decision(baseline,selections[step])
                    base.v3.save(ROOT/'decision.json',dict(step=step,decision=decision))
                    if decision!='extend':
                        break
        for mark in range(base.INTERVAL,step+1,base.INTERVAL):
            if mark not in selections:
                selections[mark]=finish(mark,panels)
                base.v3.save(ROOT/f'eval_{mark}.json',selections[mark])
        def rank(mark):
            counts=base.panel_counts(sum(selections[mark].values(),[]))
            return counts['arrival'],-counts['collision'],-mark
        best=max((m for m in selections if m),key=rank)
        for mark in sorted({0,best,step}):
            queue(pool,mark,['holdout'])
        for mark in sorted({0,best,step}):
            base.v3.save(ROOT/f'holdout_{mark}.json',finish(mark,['holdout'])['holdout'])
        base.v3.save(ROOT/'selection.json',dict(best_ppo=best,last=step,holdout_seed=HOLDOUT))


if __name__=='__main__':
    main()
