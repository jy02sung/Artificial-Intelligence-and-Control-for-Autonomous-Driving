"""Matched, supervised BC comparison of the old and congestion-enriched demonstrations."""
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from utils import jys3_gap_expert as data

prior = data.prior
ROOT = data.ROOT.parent / 'jys3_gap_bc'
DENSE = data.ROOT / 'reference_dense'
SETS = {'old': prior.data.ROOT, 'dense': DENSE}
EVAL_BASE = 95380000


def fit(name):
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    packages = [(SETS[name], 'train.json', 'samples', 10), (DENSE, 'valid.json', 'valid_samples', 2)]
    datasets = []
    seeds = set()
    for root, index, samples, per_route in packages:
        rows = prior.v3.read(root / index)
        assert Counter(r['route'] for r in rows) == {i: per_route for i in range(1, 17)}
        ids = {r['seed'] for r in rows}
        assert len(ids) == len(rows) and seeds.isdisjoint(ids)
        seeds.update(ids)
        for row in rows:
            assert row['outcome'] == 'arrival'
            assert hashlib.sha256((root / samples / row['file']).read_bytes()).hexdigest() == row['sha256']
        datasets.append(prior.dagger.load_rollouts(rows, root / samples))
    assert seeds.isdisjoint({EVAL_BASE+100*i+r for i in range(8) for r in range(1,17)})
    torch.set_num_threads(2)
    model = prior.dagger.m.PPO(obs_dim=50, act_dim=2, buffer=prior.dagger.m.RolloutBuffer(),
                              seed=11, **prior.dagger.HP).policy
    h = hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().numpy().tobytes())
    prior.v3.save(folder/'settings.json', dict(initial_tensor_sha256=h.hexdigest(), validation_sha256=
        hashlib.sha256((DENSE/'valid.json').read_bytes()).hexdigest(), train_episodes=160, valid_episodes=32,
        source='jys3_fit_bc.fit', lr=0.0004, epochs=30, batch=512, extra_lane_changes=64, seed=11))
    prior.bc_fit.fit(*datasets[0], *datasets[1], 160, 32, folder)


if __name__ == '__main__':
    with ProcessPoolExecutor(2) as pool:
        list(pool.map(fit, SETS))
    a,b = [prior.v3.read(ROOT/name/'settings.json') for name in SETS]
    assert a == b
