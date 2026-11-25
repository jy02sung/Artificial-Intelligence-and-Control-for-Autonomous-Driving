"""Collect balanced 50-input expert demonstrations in the physical fork environment."""
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from utils import jys3_bc_v2 as bc
from utils import jys3_physical_forks as physical

v3 = physical.v3
ROOT = bc.OUT.parent / "jys3_physical_expert_160"
BASE = 68172500


def episode(env, number, rep, seed):
    route = f"route_{number:02d}"
    lane = min(rep % 2, env.counts[env.routes[route][0]] - 1)
    env.ego_depart_lane = str(lane)
    obs, _ = env.reset(seed=seed, options={"route":route})
    states, actions = [], []
    for step in range(1, env.max_steps+1):
        assert obs.shape == (52,) and tuple(obs[50:]) == (0.,0.)
        state = obs[:50].copy()
        action = bc.expert(state, env)
        states.append(state)
        actions.append(action)
        obs, _, done, cut, info = env.step(action)
        if done or cut:
            assert info['outcome'] != 'arrival' or info['route_intact']
            return (dict(route=number, seed=seed, lane=lane, steps=step, outcome=info['outcome'],
                         departure=info.get('off_route_detail')), np.stack(states), np.stack(actions))
    raise AssertionError("missing terminal outcome")


def collect(number):
    with TemporaryDirectory() as tmp:
        legacy = bc.environment("high_real_ratio", 11, Path(tmp)/"metadata")
        env = physical.Env(v3.ARMS[0], Path(tmp)/"physical", 11)
        env.sim.signs, env.sim.road_flags = legacy.signs, legacy.road_flags
        rows, failures, checks = [], [], []
        try:
            if number in (3,13,14,16):
                rep = 3
                seed = 24112400 + rep*100 + number
                row, obs, actions = episode(env,number,rep,seed)
                name=f"route_{number:02d}_{seed}.npz"
                with np.load(bc.OUT/'samples'/name) as previous:
                    checks.append(dict(**row, same_observations=bool(np.array_equal(obs,previous['obs'])),
                                       same_actions=bool(np.array_equal(actions,previous['actions'])),
                                       old_steps=len(previous['obs'])))
            for rep in range(10):
                attempt = 0
                while True:
                    seed = BASE + number + 100*rep + 10000*attempt
                    row, obs, actions = episode(env,number,rep,seed)
                    name=f"route_{number:02d}_{seed}.npz"
                    success=row['outcome']=='arrival'
                    folder=ROOT/('samples' if success else 'unsuccessful')
                    folder.mkdir(exist_ok=True)
                    with (folder/(name+'.tmp')).open('wb') as fp:
                        np.savez_compressed(fp,obs=obs,actions=actions)
                    (folder/(name+'.tmp')).replace(folder/name)
                    with np.load(folder/name,allow_pickle=False) as saved:
                        assert saved['obs'].shape==(row['steps'],50) and saved['actions'].shape==(row['steps'],2)
                        assert np.array_equal(saved['obs'],obs) and np.array_equal(saved['actions'],actions)
                        assert np.isfinite(obs).all() and np.isfinite(actions).all()
                        assert np.abs(obs).max()<=1 and np.abs(actions).max()<=1
                        assert np.isin(actions[:,1],[-1.,0.,1.]).all()
                    row.update(file=name,rep=rep,sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest())
                    if success:
                        rows.append(row)
                        break
                    failures.append(row)
                    attempt+=1
                    if attempt>=10:
                        raise RuntimeError(f"Repeated failures for route {number}, rep {rep}")
            return rows,failures,checks
        finally:
            env.close()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(6) as pool:
        groups=list(pool.map(collect,range(1,17)))
    rows=[r for a,_,_ in groups for r in a]
    failures=[r for _,b,_ in groups for r in b]
    checks=[r for _,_,c in groups for r in c]
    assert Counter(r['route'] for r in rows)=={n:10 for n in range(1,17)}
    assert len({r['seed'] for r in rows+failures})==len(rows+failures)
    old_seeds={r['seed'] for r in v3.read(bc.OUT/'train.json')}
    assert all(r['seed'] not in old_seeds for r in rows+failures)
    v3.save(ROOT/'train.json',rows)
    v3.save(ROOT/'manifest.json',dict(schema=bc.source.SCHEMA,traffic="high_real_ratio",vph=15000,
             collection_seed_base=BASE,expert="jys3_bc_v2.expert",episodes=len(rows),
             samples=sum(r['steps'] for r in rows),attempts=len(rows)+len(failures),
             outcomes=dict(Counter(r['outcome'] for r in rows+failures)),failures=failures,
             legacy_checks=checks))


if __name__=="__main__":
    main()
