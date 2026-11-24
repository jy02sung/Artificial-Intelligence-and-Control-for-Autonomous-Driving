"""Multi-round DAgger on top of the jys3 BC + DAgger model (pmh, no safety gate).

Start weights: models/jys3/onehop_dagger1.pt (= results/jys3_dagger_one/dagger1.pt,
jys3 commit b5a7241: BC v2 + one DAgger round, 53/64 arrivals, 11 collisions).

Each round
  1. the current student drives 16 routes x REPS high-traffic episodes
     (deterministic policy, start lane alternates 0/1 as in jys3);
  2. the jys3 v2 expert (utils.jys3_bc_v2.expert) labels every visited state;
  3. a supervised fit on the aggregate of
        the BC v2 training demonstrations,
        the jys3_dagger_one rollouts (round 1, BC student) and
        every pmh round so far,
     warm-started from the current student, 12 epochs, the jys3 loss
     (smooth L1 accel + lane cross-entropy, lane-change oversampling) and
     epoch selection on the BC v2 validation demonstrations;
  4. evaluation on the jys3 64-episode set (base 26010000, the same cases as
     results/jys3_dagger_one/evaluation.json) and on a held-out 64-episode set
     (base 29010000) that is used for nothing else.

Two chains share everything but one switch:
  plain    the jys3 DAgger loss as is;
  crash    + every batch also draws 64 states from the last 3 s (15 steps) of
           episodes that ended in a collision, so the expert's correction of
           the states that led to a crash weighs more.

The policy runs alone in every episode: no safety gate, no action filter.
The expert only labels; it never drives in collection or evaluation.

Usage: python -m utils.pmh_jys3_dagger_multi --rounds 5 --reps 8 --chains plain crash
"""
import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.jys3 import environment, m, HP, SCHEMA, OBS_DIM
from utils.jys3_bc_v2 import OUT as BC, expert
from utils.jys3_dagger_one import OUT as ROUND1, load_rollouts
from utils.jys3_fit_bc import load

START = m.ROOT / 'models/jys3/onehop_dagger1.pt'
OUT = m.ROOT / 'results/pmh_jys3_dagger_multi'
EVAL_BASE, HOLDOUT_BASE, COLLECT_BASE = 26010000, 29010000, 27200000
CRASH_STEPS, CRASH_DRAW = 15, 64


def policy(path):
    agent = m.PPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=11, **HP)
    state = torch.load(path, weights_only=True, map_location='cpu')
    assert state['schema'] == SCHEMA
    agent.policy.load_state_dict(state['policy'])
    agent.policy.eval()
    return agent


def drive(job):
    model, collect, rep, base, folder = job
    torch.set_num_threads(1)
    agent = policy(model)
    rows = []
    with tempfile.TemporaryDirectory() as map_dir:
        env = environment('high_real_ratio', 11, Path(map_dir))
        try:
            for n in range(1, 17):
                route = f'route_{n:02d}'
                seed = base + 100 * rep + n
                lane = min(rep % 2, env.counts[env.routes[route][0]] - 1)
                env.ego_depart_lane = str(lane)
                obs, _ = env.reset(seed=seed, options={'route': route})
                states, labels = [], []
                for t in range(1, 1001):
                    action = agent.predict(obs, deterministic=True)
                    if collect:
                        states.append(obs)
                        labels.append(expert(obs, env))
                    obs, _, done, cut, info = env.step(action)
                    if done or cut:
                        break
                outcome = ('arrival' if info['arrived'] else 'collision' if info['collided']
                           else 'stall' if info['stalled'] else 'timeout')
                row = dict(route=n, seed=seed, lane=lane, steps=t, outcome=outcome)
                if collect:
                    row['file'] = f'{route}_{seed}.npz'
                    np.savez_compressed(Path(folder) / row['file'], obs=states, actions=labels)
                rows.append(row)
        finally:
            env.close()
    return rows


def run_batch(model, collect, reps, base, folder, workers):
    with ProcessPoolExecutor(workers) as pool:
        jobs = [(str(model), collect, rep, base, str(folder)) for rep in range(reps)]
        return [r for block in pool.map(drive, jobs) for r in block]


def crash_indices(rows, offset):
    """Indices (into the concatenated rollout tensor) of the last CRASH_STEPS states of collision episodes."""
    idx, pos = [], offset
    for r in rows:
        if r['outcome'] == 'collision':
            idx.extend(range(pos + max(0, r['steps'] - CRASH_STEPS), pos + r['steps']))
        pos += r['steps']
    return idx


def fit(initial, out_path, data, valid, crash):
    torch.set_num_threads(2)
    x, y, crash_ids = data
    vx, vy = valid
    model = policy(initial).policy
    opt = torch.optim.Adam(model.parameters(), lr=.0002)
    cls = (1 - y[:, 1]).long()
    changed = torch.where(cls != 1)[0]
    crash_ids = torch.tensor(crash_ids, dtype=torch.long)
    gen = torch.Generator().manual_seed(12)
    best, selected, history = -float('inf'), None, []
    for epoch in range(1, 13):
        model.train()
        for batch_ids in torch.randperm(len(x), generator=gen).split(512):
            extra = [changed[torch.randint(len(changed), (64,), generator=gen)]]
            if crash and len(crash_ids):
                extra.append(crash_ids[torch.randint(len(crash_ids), (CRASH_DRAW,), generator=gen)])
            ids = torch.cat([batch_ids] + extra)
            mu, _, z, _ = model(x[ids])
            # accel loss on the uniform part plus the crash draw; lane loss on everything
            acc_ids = torch.cat([torch.arange(len(batch_ids))] +
                                ([torch.arange(len(ids) - CRASH_DRAW, len(ids))] if len(extra) > 1 else []))
            loss = (F.smooth_l1_loss(mu[acc_ids, 0], y[ids][acc_ids, 0], beta=.15)
                    + F.cross_entropy(z, cls[ids]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _, z, _ = model(vx)
            vc = (1 - vy[:, 1]).long()
            conf = torch.bincount(vc * 3 + z.argmax(1), minlength=9).reshape(3, 3)
            f1 = 2 * conf.diag() / (conf.sum(0) + conf.sum(1)).clamp_min(1)
            score = float((f1[0] + f1[2]) / 2 - .1 * (mu[:, 0] - vy[:, 0]).abs().mean())
        history.append(dict(epoch=epoch, score=score))
        if score > best:
            best, selected = score, epoch
            torch.save(dict(schema=SCHEMA, policy=model.state_dict()), out_path)
    return dict(samples=len(x), crash_samples=len(crash_ids), selected=selected, history=history)


def counts(rows):
    c = Counter(r['outcome'] for r in rows)
    return dict(n=len(rows), arrival=c['arrival'], collision=c['collision'],
                timeout=c['timeout'] + c['stall'], stall=c['stall'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--reps', type=int, default=8)
    p.add_argument('--chains', nargs='+', default=['plain', 'crash'])
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--eval-reps', type=int, default=4)
    p.add_argument('--no-baseline', action='store_true')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    # fixed data: BC v2 demonstrations and the jys3 round-1 rollouts
    orig = json.loads((BC / 'train.json').read_bytes())
    bx, by = load([r for r in orig if r['seed'] < 24113400 and r['outcome'] == 'arrival'])
    vx, vy = load([r for r in orig if r['seed'] >= 24113400 and r['outcome'] == 'arrival'])
    r1 = json.loads((ROUND1 / 'collection.json').read_bytes())
    r1x, r1y = load_rollouts(r1, ROUND1)

    # baselines on the held-out set
    for name, model in (() if a.no_baseline else
                        (('start', START), ('bc_v2', m.ROOT / 'models/jys3/onehop_bc_v2.pt'))):
        f = out / f'baseline_{name}_holdout.json'
        if not f.exists():
            f.write_text(json.dumps(run_batch(model, False, a.eval_reps, HOLDOUT_BASE, out, a.workers), indent=1))

    for chain in a.chains:
        folder = out / chain
        folder.mkdir(exist_ok=True)
        xs, ys = [bx, r1x], [by, r1y]
        crash = crash_indices(r1, len(bx))
        student, summary = START, []
        summary_path = folder / 'summary.json'
        for k in range(2, 2 + a.rounds):            # jys3_dagger_one was round 1
            rdir = folder / f'round{k}'
            rdir.mkdir(exist_ok=True)
            coll = run_batch(student, True, a.reps, COLLECT_BASE + 10000 * k, rdir, a.workers)
            (rdir / 'collection.json').write_text(json.dumps(coll, indent=1))
            dx, dy = load_rollouts(coll, rdir)
            crash += crash_indices(coll, sum(len(t) for t in xs))
            xs.append(dx)
            ys.append(dy)
            x, y = torch.cat(xs), torch.cat(ys)
            model = rdir / f'dagger{k}.pt'
            fit_info = fit(student, model, (x, y, crash), (vx, vy), chain == 'crash')
            (rdir / 'fit.json').write_text(json.dumps(fit_info, indent=1))
            ev = run_batch(model, False, a.eval_reps, EVAL_BASE, rdir, a.workers)
            ho = run_batch(model, False, a.eval_reps, HOLDOUT_BASE, rdir, a.workers)
            (rdir / 'evaluation.json').write_text(json.dumps(ev, indent=1))
            (rdir / 'holdout.json').write_text(json.dumps(ho, indent=1))
            summary.append(dict(round=k, collection=counts(coll), evaluation=counts(ev), holdout=counts(ho),
                                samples=fit_info['samples'], crash_samples=fit_info['crash_samples']))
            summary_path.write_text(json.dumps(summary, indent=1))
            print(chain, summary[-1], flush=True)
            student = model


if __name__ == '__main__':
    main()
