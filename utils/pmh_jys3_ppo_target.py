"""PPO towards the project target (arrival >= 95 %, collision < 5 %), no safety gate (pmh).

Start weights: models/pmh/jys3_dagger10_fixed.pt (BC v2 + DAgger, 245/256
arrivals on 30010000). Pure PPO from random weights reached 111/192 after
600k steps (docs/experiments/pmh-fork-reward-team-600k), so PPO here starts
from the imitation policy and has to keep and improve it.

What the earlier PPO fine-tune (utils/pmh_jys3_ppo_minreward) showed:
  plain PPO collapsed within 20k steps; a KL anchor (beta 1) held 243-245/256
  but did not improve. Changes in this script:
  1. separate critic: the value function is its own 256x256 MLP. In the
     shared-trunk network every value-loss gradient also moves the actor's
     features; the critic starts untrained, so its large early loss rewrites
     the imitation features (suspected cause of the collapse).
  2. weaker anchor: beta in {0, 0.1, 0.3} (KL to the start policy).
  3. hard-route emphasis: every block of episodes is routes 1-16 plus routes
     5, 13 and 14 twice more (22 episodes), since nearly all remaining
     failures are lane-end stalls on those routes; the other routes still
     appear in every block so they are not forgotten.
Unchanged from pmh_jys3_ppo_minreward: reward +1 arrival / -1 collision only,
critic warm-up (10 x 2048 steps, critic only), start sigma 0.22, jys3 PPO
hyperparameters (lr 1e-4 unchanged, n_steps 2048, 10 epochs, minibatch 64,
gamma .99, lambda .95, clip .2, lane entropy .002).

Checkpoints are saved as actor weights in the jys3 schema (the critic is kept
in a separate file), so every evaluation tool for the jys3 policy loads them.
Each checkpoint (every 20,480 steps) is evaluated on the 64-episode sets
26010000 and 29010000 (selection only). The untouched test set is 256
episodes on base TEST_BASE, used once for the chosen checkpoints.

Usage: python -m utils.pmh_jys3_ppo_target --chain sep01 --seed 11
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from utils.jys3 import environment, HP, m, SCHEMA, OBS_DIM
from utils.experiment_c_train import atomic_torch
from utils.networks import build_mlp
from utils import pmh_jys3_dagger_multi as dm
from utils.pmh_jys3_ppo_minreward import TwoTermReward, AnchoredPPO, START as _S

START = m.ROOT / 'models/pmh/jys3_dagger10_fixed.pt'
assert START == _S
OUT = m.ROOT / 'results/pmh_jys3_ppo_target'
BETA = {'sep0': 0.0, 'sep01': 0.1, 'sep03': 0.3}
HARD = (5, 13, 14)
WARMUP_ROLLOUTS = 10
INIT_LOG_STD = -1.5
EVERY = 20480
TEST_BASE = 48010000          # untouched 256-episode test set (16 routes x 16 reps)


class Schedule:
    """Blocks of routes 1-16 + HARD twice, shuffled; traffic seed = seed*1e6 + n; start lane alternates per block."""

    def __init__(self, env, seed, log):
        self.env, self.seed, self.log, self.n, self.queue, self.block = env, seed, log, 0, [], -1

    def reset(self):
        if not self.queue:
            self.block += 1
            self.queue = list(range(1, 17)) + list(HARD) * 2
            random.Random(self.seed * 100000 + self.block).shuffle(self.queue)
        route = f'route_{self.queue.pop():02d}'
        self.env.ego_depart_lane = str(min(self.block % 2, self.env.counts[self.env.routes[route][0]] - 1))
        obs, info = self.env.reset(seed=self.seed * 1000000 + self.n, options={'route': route})
        with self.log.open('a') as f:
            print(json.dumps([self.n, route, self.seed * 1000000 + self.n]), file=f)
        self.n += 1
        return obs, info

    def step(self, action):
        return self.env.step(action)


def add_critic(policy):
    policy.critic = build_mlp(OBS_DIM, (256, 256), 1, 'tanh')
    trunk_forward = policy.forward

    def forward(obs):
        mu, std, logits, _ = trunk_forward(obs)
        return mu, std, logits, policy.critic(obs).squeeze(-1)

    policy.forward = forward
    return policy


class TargetPPO(AnchoredPPO):
    def fit_value(self, returns):
        obs_t, _, _ = self.buffer.to_tensors(self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        total = 0.
        for _ in range(self.n_epochs):
            for mb in torch.randperm(len(obs_t)).split(self.minibatch_size):
                loss = ((self.policy.critic(obs_t[mb]).squeeze(-1) - ret_t[mb]) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += loss.item()
        return total


def actor_state(policy):
    return {k: v for k, v in policy.state_dict().items() if not k.startswith('critic.')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chain', choices=sorted(BETA), required=True)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--steps', type=int, default=10 * EVERY)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(1)
    folder = OUT / f'{a.chain}_seed{a.seed}'
    folder.mkdir(parents=True, exist_ok=False)
    agent = TargetPPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=a.seed, **HP)
    state = torch.load(START, weights_only=True, map_location='cpu')
    assert state['schema'] == SCHEMA
    agent.policy.load_state_dict(state['policy'])
    add_critic(agent.policy)
    with torch.no_grad():
        agent.policy.log_std.fill_(INIT_LOG_STD)
    agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=HP['lr'])   # include the critic; lr unchanged
    agent.setup(dm.policy(START).policy, BETA[a.chain])
    (folder / 'protocol.json').write_text(json.dumps(dict(
        start=str(START.relative_to(m.ROOT)), chain=a.chain, beta=BETA[a.chain], seed=a.seed, steps=a.steps,
        reward={'arrival': 1, 'collision': -1}, hard_routes=HARD, critic='separate 256x256 tanh MLP',
        warmup_rollouts=WARMUP_ROLLOUTS, init_log_std=INIT_LOG_STD, hp=HP, every=EVERY,
        selection_bases=[dm.EVAL_BASE, dm.HOLDOUT_BASE], test_base=TEST_BASE), indent=1))
    log = (folder / 'train.jsonl').open('a')
    env = environment('high_real_ratio', a.seed, folder / 'map')
    try:
        episodes = TwoTermReward(Schedule(env, a.seed, folder / 'episodes.jsonl'))
        obs, _ = episodes.reset()
        stats = [0., 0]
        t0 = time.time()
        for w in range(WARMUP_ROLLOUTS):
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats, timeout_is_failure=True)
            _, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            print(json.dumps(dict(warmup=w, value_loss=agent.fit_value(ret), sec=round(time.time() - t0))), file=log, flush=True)
        agent.num_timesteps = 0
        episodes.outcomes = []
        atomic_torch(dict(schema=SCHEMA, policy=actor_state(agent.policy)), folder / 'policy_0.pt')
        while agent.num_timesteps < a.steps:
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats, timeout_is_failure=True)
            adv, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            info = agent.update(adv, ret)
            out = episodes.outcomes
            print(json.dumps(dict(step=agent.num_timesteps, sec=round(time.time() - t0), std=agent.policy.current_std,
                                  episodes=len(out), arrival=out.count('arrival'), collision=out.count('collision'),
                                  **{k: round(v, 5) for k, v in info.items()})), file=log, flush=True)
            episodes.outcomes = []
            if agent.num_timesteps % EVERY == 0:
                atomic_torch(dict(schema=SCHEMA, policy=actor_state(agent.policy)), folder / f'policy_{agent.num_timesteps}.pt')
        atomic_torch(dict(critic=agent.policy.critic.state_dict()), folder / 'critic_last.pt')
    finally:
        env.close()
        log.close()
    rows = {}
    for ck in sorted(folder.glob('policy_*.pt'), key=lambda f: int(f.stem.split('_')[1])):
        ev = dm.run_batch(ck, False, 4, dm.EVAL_BASE, folder, a.workers)
        ho = dm.run_batch(ck, False, 4, dm.HOLDOUT_BASE, folder, a.workers)
        rows[ck.stem] = dict(eval=dm.counts(ev), holdout=dm.counts(ho))
        (folder / 'selection.json').write_text(json.dumps(rows, indent=1))
        print(a.chain, a.seed, ck.stem, rows[ck.stem], flush=True)


if __name__ == '__main__':
    main()
