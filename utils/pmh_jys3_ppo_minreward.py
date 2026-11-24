"""PPO fine-tuning of the DAgger policy with a two-term reward (pmh, no safety gate).

Start weights: models/pmh/jys3_dagger10_fixed.pt (BC v2 -> DAgger rounds 1-10,
fixed chain, seed 11; final 256: 245 arrivals, 2 collisions, 9 stalls).

Reward (the env's own reward is discarded):
    +1  arrival (route completed in order)
    -1  collision
Nothing else. A timeout or a lane-end stall ends the episode as a failure with
0, so it already loses the +1; no separate term is needed. No fork-pass
shaping: the start policy already follows the routes.

Why the jys3 PPO runs got worse, and what is different here
  1. The value head was never trained (BC/DAgger fit only the actor), so the
     first PPO updates used advantages from a random critic.
     -> critic warm-up: WARMUP_ROLLOUTS rollouts of 2048 steps in which only
        the value head is fitted (actor frozen); these steps are not counted.
  2. The accel exploration noise was the untrained init sigma = exp(-0.5) = 0.61
     (accel is in [-1, 1]), so rollouts were much noisier than the policy.
     -> log_std set to INIT_LOG_STD = -1.5 (sigma 0.22) at the start; still learnt.
  3. Nothing kept the policy near the imitation solution.
     -> chain 'kl' adds BETA * KL(start policy || policy) on the rollout states
        (lane categorical KL + accel Gaussian KL at the current sigma).
        chain 'plain' is the same with BETA = 0.

PPO hyperparameters are the jys3 ones (utils.experiment_b.HP: lr 1e-4,
n_steps 2048, 10 epochs, minibatch 64, gamma .99, lambda .95, clip .2,
lane entropy .002). The learning rate is not changed.
Episodes: utils.experiment_c_train.ScheduledEpisodes (16 routes in shuffled
blocks, traffic seed = seed*1e6 + n, start lane alternating), high_real_ratio.
Checkpoints every 20,480 steps; each is evaluated on the 64-episode holdout
(base 29010000). The policy drives alone in every episode.

Usage: python -m utils.pmh_jys3_ppo_minreward --chain kl --seed 11 --steps 204800
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.jys3 import environment, HP, m, SCHEMA, OBS_DIM
from utils.experiment_c_train import ScheduledEpisodes, atomic_torch
from utils import pmh_jys3_dagger_multi as dm

START = m.ROOT / 'models/pmh/jys3_dagger10_fixed.pt'
OUT = m.ROOT / 'results/pmh_jys3_ppo_minreward'
WARMUP_ROLLOUTS = 10
INIT_LOG_STD = -1.5
BETA = {'plain': 0.0, 'kl': 1.0}
EVERY = 20480


class TwoTermReward:
    """+1 arrival, -1 collision, 0 otherwise."""

    def __init__(self, inner):
        self.inner = inner
        self.outcomes = []

    def reset(self):
        return self.inner.reset()

    def step(self, action):
        obs, _, done, cut, info = self.inner.step(action)
        reward = 1.0 if info['arrived'] else -1.0 if info['collided'] else 0.0
        if done or cut:
            self.outcomes.append('arrival' if info['arrived'] else 'collision' if info['collided']
                                 else 'stall' if info['stalled'] else 'timeout')
        return obs, reward, done, cut, info


class AnchoredPPO(m.PPO):
    def setup(self, ref, beta):
        self.ref, self.beta = ref, beta
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def anchor(self, obs):
        mu, std, logits, _ = self.policy(obs)
        with torch.no_grad():
            rmu, _, rlogits, _ = self.ref(obs)
        lane = (torch.softmax(rlogits, -1) * (torch.log_softmax(rlogits, -1) - torch.log_softmax(logits, -1))).sum(-1)
        accel = ((mu - rmu) ** 2).sum(-1) / (2 * std.detach() ** 2)
        return (lane + accel).mean()

    def fit_value(self, returns):
        obs_t, _, _ = self.buffer.to_tensors(self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        params = list(self.policy.v_head.parameters()) + list(self.policy.trunk.parameters())
        loss_sum = 0.
        for _ in range(self.n_epochs):
            for mb in torch.randperm(len(obs_t)).split(self.minibatch_size):
                trunk = self.policy.trunk(obs_t[mb]).detach()        # actor frozen: trunk not updated
                loss = ((self.policy.v_head(trunk).squeeze(-1) - ret_t[mb]) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                loss_sum += loss.item()
        return loss_sum

    def update(self, advantages, returns):
        if not self.beta:
            return super().update(advantages, returns)
        # PPO update of utils/algorithms.ppo with the anchor term added to each minibatch loss
        obs_t, act_t, old_t = self.buffer.to_tensors(self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        stats = dict(policy_loss=0., value_loss=0., anchor=0., approx_kl=0.)
        n = 0
        for _ in range(self.n_epochs):
            for mb in torch.randperm(len(obs_t)).split(self.minibatch_size):
                log_prob, entropy, value = self.policy.evaluate_actions(obs_t[mb], act_t[mb])
                ratio = torch.exp(log_prob - old_t[mb])
                pl = -torch.min(ratio * adv_t[mb], ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]).mean()
                vl = ((value - ret_t[mb]) ** 2).mean()
                an = self.anchor(obs_t[mb])
                loss = pl + self.value_coef * vl + self.beta * an \
                    - self.lane_entropy_coef * self.policy.lane_entropy(obs_t[mb]).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats['approx_kl'] += (old_t[mb] - log_prob).mean().item()
                stats['policy_loss'] += pl.item(); stats['value_loss'] += vl.item(); stats['anchor'] += an.item()
                n += 1
        return {k: v / n for k, v in stats.items()}


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
    agent = AnchoredPPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=a.seed, **HP)
    state = torch.load(START, weights_only=True, map_location='cpu')
    assert state['schema'] == SCHEMA
    agent.policy.load_state_dict(state['policy'])
    with torch.no_grad():
        agent.policy.log_std.fill_(INIT_LOG_STD)
    ref = dm.policy(START).policy
    agent.setup(ref, BETA[a.chain])
    (folder / 'protocol.json').write_text(json.dumps(dict(
        start=str(START.relative_to(m.ROOT)), chain=a.chain, beta=BETA[a.chain], seed=a.seed, steps=a.steps,
        reward={'arrival': 1, 'collision': -1}, warmup_rollouts=WARMUP_ROLLOUTS, init_log_std=INIT_LOG_STD,
        hp=HP, every=EVERY, holdout_base=dm.HOLDOUT_BASE), indent=1))
    log = (folder / 'train.jsonl').open('a')
    env = environment('high_real_ratio', a.seed, folder / 'map')
    try:
        episodes = TwoTermReward(ScheduledEpisodes(env, a.seed, folder / 'episodes.jsonl'))
        obs, _ = episodes.reset()
        stats = [0., 0]
        t0 = time.time()
        for w in range(WARMUP_ROLLOUTS):
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=stats, timeout_is_failure=True)
            _, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            vl = agent.fit_value(ret)
            print(json.dumps(dict(warmup=w, value_loss=vl, sec=round(time.time() - t0))), file=log, flush=True)
        agent.num_timesteps = 0
        episodes.outcomes.clear()
        atomic_torch(dict(schema=SCHEMA, policy=agent.policy.state_dict()), folder / 'policy_0.pt')
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
                atomic_torch(dict(schema=SCHEMA, policy=agent.policy.state_dict()),
                             folder / f'policy_{agent.num_timesteps}.pt')
    finally:
        env.close()
        log.close()
    # holdout evaluation of every checkpoint (deterministic policy, 64 episodes)
    rows = {}
    for ck in sorted(folder.glob('policy_*.pt'), key=lambda f: int(f.stem.split('_')[1])):
        ho = dm.run_batch(ck, False, 4, dm.HOLDOUT_BASE, folder, a.workers)
        rows[ck.stem] = dm.counts(ho)
        (folder / 'holdout.json').write_text(json.dumps(rows, indent=1))
        print(a.chain, a.seed, ck.stem, rows[ck.stem], flush=True)


if __name__ == '__main__':
    main()
