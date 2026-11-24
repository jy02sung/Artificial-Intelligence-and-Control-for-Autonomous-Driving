"""DAgger rounds 7+ for the jys3 one-hop policy with the two failure fixes (pmh, no safety gate).

Continues the crash chain of utils/pmh_jys3_dagger_multi.py. Start weights:
models/pmh/jys3_dagger6_crash.pt (round 6, 239/256 arrivals, 11 collisions,
6 stalls). Prior data: BC v2 demonstrations, the jys3 round-1 rollouts and the
crash chain's rounds 2-6 (results/pmh_jys3_dagger_multi/crash/).

Chains
  control  the round-6 recipe continued: v2 expert labels, crash draw.
  fixed    the fixes from the 17-failure analysis:
           a. labels from expert v3 (utils/pmh_jys3_expert_v3: lane-end merge
              rule when stopped); all prior data relabelled offline, since the
              expert reads only the observation and env constants;
           b. disagreement mining: every epoch the current model is run on the
              whole aggregate and 64 states per batch are drawn from those where
              its lane command differs from the label (the 11 collisions were
              lane changes the expert would not make);
           c. weave emphasis: each round also collects WEAVE_REPS extra episodes
              of the routes through the 51 m weave 170059994 (3/4/5/13/14/15);
           plus the crash draw.

Everything else (loss, 12 epochs, Adam 2e-4 supervised fit, epoch selection on
the BC v2 validation demonstrations, evaluation sets) is as in
pmh_jys3_dagger_multi. The policy drives alone in every episode.

Training seed (--seed, default 11 = the first run): sets the fit's sampling
generator (seed + 1) and shifts the collection seeds of rounds 7+ by
100000 * (seed - 11). Start weights and prior data (rounds 2-6) are shared by
all seeds. Other seeds write to results/pmh_jys3_dagger_fix/seed<S>/<chain>.

Usage: python -m utils.pmh_jys3_dagger_fix --rounds 4 --chains control fixed [--seed 12]
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

from utils.jys3 import environment, m, SCHEMA, OBS_DIM
from utils.jys3_bc_v2 import OUT as BC, expert as expert_v2
from utils.jys3_dagger_one import OUT as ROUND1, load_rollouts
from utils.jys3_fit_bc import load
from utils import pmh_jys3_dagger_multi as dm
from utils.pmh_jys3_expert_v3 import expert_v3, relabel, constants

START = m.ROOT / 'models/pmh/jys3_dagger6_crash.pt'
PRIOR = dm.OUT / 'crash'
OUT = m.ROOT / 'results/pmh_jys3_dagger_fix'
WEAVE = (3, 4, 5, 13, 14, 15)
WEAVE_REPS = 4
MINE_DRAW = 64


def drive(job):
    model, collect, rep, base, folder, which, routes = job
    torch.set_num_threads(1)
    agent = dm.policy(model)
    label = expert_v3 if which == 'v3' else expert_v2
    rows = []
    with tempfile.TemporaryDirectory() as map_dir:
        env = environment('high_real_ratio', 11, Path(map_dir))
        try:
            for n in routes:
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
                        labels.append(label(obs, env))
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


def collect(model, k, folder, which, weave, workers, seed=11):
    base = dm.COLLECT_BASE + 10000 * k + 100000 * (seed - 11)
    jobs = [(str(model), True, rep, base, str(folder), which, range(1, 17)) for rep in range(8)]
    if weave:
        jobs += [(str(model), True, rep, base + 5000, str(folder), which, WEAVE) for rep in range(WEAVE_REPS)]
    with ProcessPoolExecutor(workers) as pool:
        return [r for block in pool.map(drive, jobs) for r in block]


def fit(initial, out_path, x, y, crash_ids, vx, vy, mine, seed=11):
    torch.set_num_threads(2)
    model = dm.policy(initial).policy
    opt = torch.optim.Adam(model.parameters(), lr=.0002)
    cls = (1 - y[:, 1]).long()
    changed = torch.where(cls != 1)[0]
    crash_ids = torch.tensor(crash_ids, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed + 1)
    best, selected, history = -float('inf'), None, []
    for epoch in range(1, 13):
        hard = torch.empty(0, dtype=torch.long)
        if mine:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(x[i:i + 8192])[2].argmax(1) for i in range(0, len(x), 8192)])
            hard = torch.where(pred != cls)[0]
        model.train()
        for batch_ids in torch.randperm(len(x), generator=gen).split(512):
            parts = [batch_ids, changed[torch.randint(len(changed), (64,), generator=gen)]]
            acc = [torch.arange(len(batch_ids))]
            if len(crash_ids):
                start = sum(len(p) for p in parts)
                parts.append(crash_ids[torch.randint(len(crash_ids), (dm.CRASH_DRAW,), generator=gen)])
                acc.append(torch.arange(start, start + dm.CRASH_DRAW))
            if len(hard):
                parts.append(hard[torch.randint(len(hard), (MINE_DRAW,), generator=gen)])
            ids = torch.cat(parts)
            acc = torch.cat(acc)
            mu, _, z, _ = model(x[ids])
            loss = F.smooth_l1_loss(mu[acc, 0], y[ids][acc, 0], beta=.15) + F.cross_entropy(z, cls[ids])
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
        history.append(dict(epoch=epoch, score=score, hard=len(hard)))
        if score > best:
            best, selected = score, epoch
            torch.save(dict(schema=SCHEMA, policy=model.state_dict()), out_path)
    return dict(samples=len(x), crash_samples=len(crash_ids), selected=selected, history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--chains', nargs='+', default=['control', 'fixed'])
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    out = OUT if a.seed == 11 else OUT / f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        env = environment('high_real_ratio', 11, Path(d))
        consts = constants(env)
        env.close()

    orig = json.loads((BC / 'train.json').read_bytes())
    bx, by = load([r for r in orig if r['seed'] < 24113400 and r['outcome'] == 'arrival'])
    vx, vy = load([r for r in orig if r['seed'] >= 24113400 and r['outcome'] == 'arrival'])
    r1 = json.loads((ROUND1 / 'collection.json').read_bytes())
    parts = [(bx, by, None), (*load_rollouts(r1, ROUND1), r1)]
    for k in range(2, 7):
        rows = json.loads((PRIOR / f'round{k}' / 'collection.json').read_bytes())
        parts.append((*load_rollouts(rows, PRIOR / f'round{k}'), rows))

    for chain in a.chains:
        fixed = chain == 'fixed'
        which = 'v3' if fixed else 'v2'
        folder = out / chain
        folder.mkdir(exist_ok=True)
        xs, ys, crash, offset = [], [], [], 0
        for x, y, rows in parts:
            xs.append(x)
            ys.append(torch.from_numpy(relabel(x.numpy(), consts)) if fixed else y)
            if rows is not None:
                crash += dm.crash_indices(rows, offset)
            offset += len(x)
        if fixed:
            vy_use = torch.from_numpy(relabel(vx.numpy(), consts))
            changed_labels = int((torch.cat(ys)[:, 1] != torch.cat([p[1] for p in parts])[:, 1]).sum())
        else:
            vy_use, changed_labels = vy, 0
        student, summary = START, []
        for k in range(7, 7 + a.rounds):
            rdir = folder / f'round{k}'
            rdir.mkdir(exist_ok=True)
            coll = collect(student, k, rdir, which, fixed, a.workers, a.seed)
            (rdir / 'collection.json').write_text(json.dumps(coll, indent=1))
            dx, dy = load_rollouts(coll, rdir)
            crash += dm.crash_indices(coll, offset)
            offset += len(dx)
            xs.append(dx)
            ys.append(dy)
            model = rdir / f'dagger{k}.pt'
            info = fit(student, model, torch.cat(xs), torch.cat(ys), crash, vx, vy_use, mine=fixed, seed=a.seed)
            (rdir / 'fit.json').write_text(json.dumps(info, indent=1))
            ev = dm.run_batch(model, False, 4, dm.EVAL_BASE, rdir, a.workers)
            ho = dm.run_batch(model, False, 4, dm.HOLDOUT_BASE, rdir, a.workers)
            (rdir / 'evaluation.json').write_text(json.dumps(ev, indent=1))
            (rdir / 'holdout.json').write_text(json.dumps(ho, indent=1))
            summary.append(dict(round=k, collection=dm.counts(coll), evaluation=dm.counts(ev),
                                holdout=dm.counts(ho), samples=info['samples'],
                                crash_samples=info['crash_samples'], relabelled_lane_labels=changed_labels,
                                hard_last_epoch=info['history'][-1]['hard']))
            (folder / 'summary.json').write_text(json.dumps(summary, indent=1))
            print(chain, summary[-1], flush=True)
            student = model


if __name__ == '__main__':
    main()
