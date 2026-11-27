"""Pure PPO (random init, no imitation) with a traffic curriculum and the two-term reward (pmh, no safety gate).

Question: can PPO alone reach the project target (arrival >= 95 %, collision < 5 %
on high traffic)? Earlier pure-PPO runs from random weights went straight into
high_real_ratio and reached 29-58 % (600k steps at most). Most early episodes
there end in a collision, so the policy learns to stop before it learns routes.

Here the traffic grows only after the policy can drive the lighter one:
    stage 0  none            no other vehicles (lane choice / route following only)
    stage 1  low             20 % of the base flows, car types only
    stage 2  real_ratio      50 % of the base flows, observed vehicle mix
    stage 3  high_real_ratio 100 % (15,000 veh/h), the evaluation traffic
A stage is left when the stochastic training policy completed at least GATE[stage]
of its last WINDOW finished training episodes (and spent MIN_STAGE_STEPS there).
The last stage is kept until the step budget ends.

Reward: +1 arrival, -1 collision, nothing else (utils.pmh_jys3_ppo_minreward.TwoTermReward).
Timeouts and lane-end stalls end the episode with 0 and count as failures.
Network and PPO: jys3 defaults (256x256 tanh shared actor-critic, lr 1e-4
unchanged, n_steps 2048, 10 epochs, minibatch 64, gamma .99, lambda .95,
clip .2, lane entropy .002, lane-keep bias 1). Initial weights from the
training seed; their SHA-256 is written to protocol.json.
Episodes: blocks of routes 1-16 shuffled, traffic seed = seed*1e6 + n,
start lane alternates per block (utils.pmh_jys3_ppo_target.Schedule, no extra hard routes).

Checkpoints every EVERY steps. After training each checkpoint taken in stage 3
is evaluated on high traffic, 128 selection episodes (26010000 + 29010000);
the untouched test (48010000, 256) is run once for the chosen checkpoints.

Usage: python -m utils.pmh_ppo_curriculum --seed 11 [--steps 3000000]
       python -m utils.pmh_ppo_curriculum --smoke
"""
import argparse
import hashlib
import json
import tempfile
import time
from collections import deque
from pathlib import Path

import torch

from utils.jys3 import environment, HP, m, SCHEMA, OBS_DIM
from utils.experiment_c_train import atomic_torch
from utils import pmh_jys3_dagger_multi as dm
from utils.pmh_jys3_ppo_minreward import TwoTermReward
from utils.pmh_jys3_ppo_target import Schedule

OUT = m.ROOT / 'results/pmh_ppo_curriculum'
STAGES = ('none', 'low', 'real_ratio', 'high_real_ratio')
GATE = (0.90, 0.85, 0.80)
WINDOW = 100
MIN_STAGE_STEPS = 50000
EVERY = 102400


def digest(policy):
    h = hashlib.sha256()
    for k, v in sorted(policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke():
    with tempfile.TemporaryDirectory() as d:
        for stage in STAGES:
            env = environment(stage, 11, Path(d) / stage)
            try:
                obs, _ = env.reset(seed=1, options={'route': 'route_13'})
                for _ in range(50):
                    obs, r, done, cut, info = env.step(env.action_space.sample())
                    if done or cut:
                        break
                print(stage, obs.shape, 'ok', flush=True)
            finally:
                env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--steps', type=int, default=3000000)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.smoke:
        return smoke()
    folder = OUT / f'seed{a.seed}'
    folder.mkdir(parents=True, exist_ok=False)
    agent = m.PPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=a.seed, **HP)
    (folder / 'protocol.json').write_text(json.dumps(dict(
        start='random init', initial_sha256=digest(agent.policy), seed=a.seed, steps=a.steps,
        reward={'arrival': 1, 'collision': -1}, stages=STAGES, gate=GATE, window=WINDOW,
        min_stage_steps=MIN_STAGE_STEPS, hp=HP, every=EVERY, imitation=False, safety_gate=False,
        selection_bases=[dm.EVAL_BASE, dm.HOLDOUT_BASE], test_base=48010000), indent=1))
    atomic_torch(dict(schema=SCHEMA, policy=agent.policy.state_dict()), folder / 'policy_0.pt')
    log = (folder / 'train.jsonl').open('a')
    stage, stage_start, recent = 0, 0, deque(maxlen=WINDOW)
    stage_of = {0: STAGES[0]}
    env = environment(STAGES[stage], a.seed, folder / 'map')
    try:
        schedule = Schedule(env, a.seed, folder / 'episodes.jsonl', hard_reps=0)
        episodes = TwoTermReward(schedule)
        obs, _ = episodes.reset()
        stats, t0 = [0., 0], time.time()
        while agent.num_timesteps < a.steps:
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats, timeout_is_failure=True)
            adv, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            info = agent.update(adv, ret)
            out = episodes.outcomes
            recent.extend(o == 'arrival' for o in out)
            rate = sum(recent) / len(recent) if recent else 0.
            print(json.dumps(dict(step=agent.num_timesteps, sec=round(time.time() - t0), stage=STAGES[stage],
                                  std=agent.policy.current_std, episodes=len(out),
                                  **{k: out.count(k) for k in ('arrival', 'collision', 'stall', 'timeout')},
                                  rolling_arrival=round(rate, 3),
                                  **{k: round(v, 5) for k, v in info.items()})), file=log, flush=True)
            episodes.outcomes = []
            if agent.num_timesteps % EVERY == 0:
                atomic_torch(dict(schema=SCHEMA, policy=agent.policy.state_dict()),
                             folder / f'policy_{agent.num_timesteps}.pt')
                stage_of[agent.num_timesteps] = STAGES[stage]
            if (stage < len(GATE) and len(recent) == WINDOW and rate >= GATE[stage]
                    and agent.num_timesteps - stage_start >= MIN_STAGE_STEPS):
                stage += 1
                stage_start = agent.num_timesteps
                recent.clear()
                env.close()
                env = environment(STAGES[stage], a.seed, folder / 'map')
                schedule.env = env
                obs, _ = episodes.reset()       # the unfinished episode is dropped (not counted)
                stats = [0., 0]
                print(json.dumps(dict(advance=STAGES[stage], step=agent.num_timesteps,
                                      sec=round(time.time() - t0))), file=log, flush=True)
            (folder / 'stages.json').write_text(json.dumps(stage_of, indent=1))
    finally:
        env.close()
        log.close()
    rows = {}
    for ck in sorted(folder.glob('policy_*.pt'), key=lambda f: int(f.stem.split('_')[1])):
        if stage_of.get(int(ck.stem.split('_')[1])) != STAGES[-1]:
            continue
        ev = dm.run_batch(ck, False, 4, dm.EVAL_BASE, folder, a.workers)
        ho = dm.run_batch(ck, False, 4, dm.HOLDOUT_BASE, folder, a.workers)
        rows[ck.stem] = dict(eval=dm.counts(ev), holdout=dm.counts(ho))
        (folder / 'selection.json').write_text(json.dumps(rows, indent=1))
        print(a.seed, ck.stem, rows[ck.stem], flush=True)


if __name__ == '__main__':
    main()
