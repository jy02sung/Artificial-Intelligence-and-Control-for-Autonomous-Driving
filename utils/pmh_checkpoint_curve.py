"""Evaluate saved checkpoints of pmh_prepare_trial runs to get learning curves.

Uses the same unshaped evaluation environment as pmh_prepare_trial.evaluate
(no prepare/pressure/risk/TTC terms), so every point is comparable with the
final model.summary.json. Evaluation seeds are 40001..40000+count, so count=10
is the first half of the count=20 final evaluation.

Writes <run>/curve/policy-<step>.{evaluation,summary}.json and skips points
that already exist, then <output>/curve.json and curve.csv for all runs.
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('IC_LIBSUMO', '1')
from utils import pmh_prepare_trial as t

f, b = t.f, t.b


def nearest_checkpoints(run, targets):
    saved = {int(p.stem.split('-')[1]): p for p in run.glob('policy-*.pt')}
    picked = []
    for target in targets:
        step = min(saved, key=lambda s: abs(s - target))
        if saved[step] not in picked:
            picked.append(saved[step])
    return picked


def evaluate_checkpoint(spec):
    run, checkpoint, count = Path(spec[0]), Path(spec[1]), spec[2]
    out = run / 'curve'
    out.mkdir(exist_ok=True)
    target = out / f'{checkpoint.stem}.summary.json'
    if target.exists():
        return str(target)
    meta = f.read(run / 'manifest.json')
    b.torch.set_num_threads(1)
    agent = b.PPO(obs_dim=47, act_dim=2, **meta['hp'])
    agent.load(str(checkpoint))
    # Same unshaped evaluation environment as the final evaluation.
    f.environment = lambda mode, seed, load='none', routes=None, variants=(): \
        t.environment(seed, load, False, variants)
    rows = []
    for load in ('none', 'low'):
        for route in f.ROUTES:
            rows.extend(f.evaluate(agent, 'N1', meta['seed'], load, [route], count, ()))
    b.save(out / f'{checkpoint.stem}.evaluation.json', rows)
    b.save(target, t.summary(rows))
    return str(target)


def lane_change_collision(row):
    last = (row.get('diagnostic', {}).get('samples') or [])[-4:]
    return len({x['lane_index'] for x in last if not x['edge'].startswith(':')}) > 1


def point(rows, load):
    rows = [r for r in rows if r['load'] == load]
    n = len(rows)
    arrivals = sum(r['arrival'] for r in rows)
    collisions = [r for r in rows if r['collision']]
    return dict(cases=n, arrivals=arrivals, collisions=len(collisions),
                success_rate=arrivals / n, collision_rate=len(collisions) / n,
                route_lane_blocked=sum(r['failure'] == 'route_lane_blocked' for r in rows),
                lane_change_collisions=sum(map(lane_change_collision, collisions)))


def collect(runs, count):
    records = []
    for run in runs:
        meta = f.read(run / 'manifest.json')
        files = sorted(run.glob('curve/policy-*.evaluation.json'))
        final = run / 'model.evaluation.json'
        sources = [(int(p.name.split('-')[1].split('.')[0]), p) for p in files]
        if final.exists():
            sources.append((meta['budget'], final))
        for step, path in sorted(sources):
            rows = [r for r in f.read(path) if r['seed'] <= 40000 + count]
            for load in ('none', 'low'):
                records.append(dict(mode=meta['mode'], seed=meta['seed'], step=step,
                                    load=load, **point(rows, load)))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', type=Path, nargs='+', help='result folders with runs/*')
    p.add_argument('--every', type=int, default=50000)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = sorted(r for folder in args.runs for r in (folder / 'runs').iterdir() if r.is_dir())
    specs = []
    for run in runs:
        budget = f.read(run / 'manifest.json')['budget']
        targets = range(args.every, budget, args.every)  # final model is already evaluated
        specs += [(str(run), str(c), args.count) for c in nearest_checkpoints(run, targets)]
    print(f'{len(specs)} checkpoint evaluations over {len(runs)} runs', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done in pool.map(evaluate_checkpoint, specs):
            print('done', done, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = collect(runs, args.count)
    b.save(args.output / 'curve.json', records)
    with (args.output / 'curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote {len(records)} curve points to {args.output}', flush=True)


if __name__ == '__main__':
    main()
