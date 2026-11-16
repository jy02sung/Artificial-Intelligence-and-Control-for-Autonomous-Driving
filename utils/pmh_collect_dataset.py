"""Collect the course POMDP transition dataset with the best pmh policy (zipper_filter).

Reuses collect_pomdp_data.collect() unchanged (same record format and
required columns: episode, t, observation, action_raw, action.*, reward,
terminated, truncated, state, detected_vehicles, event.*, next_observation,
next_state). Only the environment and the policy are swapped:

  environment  pmh zipper map (maps/toegyewon_zipper), measured traffic
               low_exit25, action safety filter on, evaluation mode (no
               training-only shaping in the reward), as in
               utils/pmh_zipper_trial.environment(train=False)
  policy       results/pmh_zipper_filter/runs/zipper_filter-<seed>/model.pt,
               deterministic

Team columns are attached under event.team (event.update(info['ic_event'])
already copies the IC fields such as event.route):

  zipper_obs             obs[37:45], the merge observation slots
  merge_pass             the ego entered a zipper merge link this step
  merge_simultaneous     ... together with a competing vehicle
  filter_speed_capped    the safety filter lowered the commanded speed
  lane_change_blocked    the lane-change gate refused the command
  policy_seed            training seed of the model that drove

Episode e uses route ROUTES[e % 16] and SUMO seed base_seed + e.

Usage: python -m utils.pmh_collect_dataset --seeds 11 22 33 44 55 --episodes-per-seed 16
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import collect_pomdp_data as cpd
from collect_pomdp_data import collect, iter_transitions
from utils import pmh_zipper_trial as z

MODE = 'zipper_filter'
RUNS = z.ROOT / 'results/pmh_zipper_filter/runs'
ROUTES = list(z.f.ROUTES)


def make_env(seed, out):
    env = z.environment(seed, 'measured', MODE, out, train=False, variants=())
    reset, step = env.reset, env.step
    counter = {'episode': 0}

    def reset_route(seed=None, options=None):
        options = dict(options or {}, route=ROUTES[counter['episode'] % len(ROUTES)])
        counter['episode'] += 1
        return reset(seed=seed, options=options)

    def step_team(action):
        passes, simul = env.episode['passes'], env.episode['simultaneous']
        blocked = env.filter_blocked_lc
        obs, reward, done, cut, info = step(action)
        info.setdefault('ic_event', {})['team'] = dict(
            zipper_obs=[round(float(x), 4) for x in np.asarray(obs)[z.SLOT]],
            merge_pass=env.episode['passes'] > passes,
            merge_simultaneous=env.episode['simultaneous'] > simul,
            filter_speed_capped=bool(info.get('filter_speed_capped', False)),
            lane_change_blocked=env.filter_blocked_lc > blocked,
            policy_seed=seed)
        return obs, reward, done, cut, info

    env.reset, env.step = reset_route, step_team
    return env


def run(seed, episodes, base_seed, out_dir, scenario_out):
    model = RUNS / f'{MODE}-{seed}' / 'model.pt'
    hp = json.loads((model.parent / 'manifest.json').read_text())['hp']
    z.configure(str(scenario_out))
    agent = z.b.PPO(obs_dim=47, act_dim=2, **hp)
    agent.load(str(model))
    cpd.make_policy = lambda args, env, worker_seed: (
        lambda obs: np.clip(agent.predict(obs, deterministic=True), -1, 1).astype(np.float32))
    args = SimpleNamespace(seed=base_seed, workers=1, policy='ppo', model=str(model), device='cpu',
                           stochastic=False, gui=False, ndigits=4, episodes=episodes)
    name = f'pmh_{MODE}_seed{seed}'
    jsonl, npz = out_dir / f'{name}.jsonl.gz', out_dir / f'{name}.npz'
    n = collect(args, None, list(range(episodes)), 0, str(jsonl), str(npz),
                env_factory=lambda: make_env(seed, scenario_out), tag=f'[seed {seed}] ')
    return dict(seed=seed, model=str(model.relative_to(z.ROOT)), transitions=n,
                jsonl=str(jsonl.relative_to(z.ROOT)), npz=str(npz.relative_to(z.ROOT)))


def verify(entry):
    d = np.load(z.ROOT / entry['npz'])
    n = entry['transitions']
    assert all(len(d[k]) == n for k in ('observation', 'action_raw', 'reward', 'next_observation',
                                         'episode', 't', 'terminated', 'truncated', 'lane_change'))
    assert d['observation'].shape[1] == 47 and np.isfinite(d['observation']).all()
    count = 0
    for count, rec in enumerate(iter_transitions(str(z.ROOT / entry['jsonl'])), 1):
        assert {'next_observation', 'next_state', 'state', 'detected_vehicles'} <= rec.keys()
        assert 'team' in rec['event'] and 'route' in rec['event']
    assert count == n
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    p.add_argument('--episodes-per-seed', type=int, default=16)
    p.add_argument('--base-seed', type=int, default=50001)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--out-dir', type=Path, default=z.ROOT / 'data/pmh_zipper_filter')
    a = p.parse_args()
    a.out_dir = a.out_dir.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    scenario_out = a.out_dir / 'scenario'
    for load in z.LOADS:
        z.scenario(load, scenario_out)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        entries = list(pool.map(run, a.seeds, [a.episodes_per_seed] * len(a.seeds),
                                [a.base_seed] * len(a.seeds), [a.out_dir] * len(a.seeds),
                                [scenario_out] * len(a.seeds)))
    for e in entries:
        e['verified'] = verify(e)
    meta = dict(mode=MODE, traffic=z.CASE, network=str(z.NETWORK.relative_to(z.ROOT)),
                routes=ROUTES, base_seed=a.base_seed, episodes_per_seed=a.episodes_per_seed,
                seconds=round(time.time() - t0, 1), shards=entries)
    (a.out_dir / 'manifest.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
