"""Fresh paired PPO comparison: only 300m lane-progress reward differs.

Does not resume historical checkpoints or invoke legacy publication routines.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

os.environ.setdefault('IC_LIBSUMO', '0')  # Installed SUMO binary, no libsumo required.
from env.prepare_reward import PrepareCredit
from utils import train_navigation_clarity as c

f, b = c.f, c.f.base
OUT = f.ROOT / 'results/pmh_prepare300_fresh'


class TrialEnv(c.ClarityEnv):
    def __init__(self, *args, prepare=False, **kwargs):
        self.prepare = prepare
        self.extra_total = 0.0
        self.extra_events = 0
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.credit = PrepareCredit()
        return super().reset(*args, **kwargs)

    def step(self, action):
        v = b.traci.vehicle
        edge, lane = v.getRoadID(self.ego_id), v.getLaneIndex(self.ego_id)
        nav = self.navigation_context()
        obs, reward, done, cut, info = super().step(action)
        extra = 0.0
        if (self.prepare and not done and not cut and not edge.startswith(':')
                and self.ego_id in v.getIDList() and v.getRoadID(self.ego_id) == edge):
            extra = self.credit.award(edge, nav['decision_edge'], nav['prepare_targets'],
                                      lane, v.getLaneIndex(self.ego_id), nav['distance'])
        self.extra_total += extra
        self.extra_events += int(extra > 0)
        info['prepare_reward'] = extra
        return obs, reward + extra, done, cut, info


def environment(seed, load='low', prepare=False, variants=()):
    return TrialEnv(str(f.prepare(load)), copy.deepcopy(b.ROAD_NEW),
                    dict(b.EGO, depart_lane='best', depart_speed='max'),
                    dict(b.SIMULATION, warmup_steps=300), b.OBSERVATION,
                    copy.deepcopy(b.ACTION), copy.deepcopy(b.REWARD),
                    dict(b.TRAFFIC, prefill={'enabled': False}), variants, seed,
                    navigation=b.NAV, clarity='baseline', prepare=prepare)


def summary(rows):
    def group(xs):
        n = len(xs)
        a, col = sum(x['arrival'] for x in xs), sum(x['collision'] for x in xs)
        return dict(cases=n, arrivals=a, collisions=col, other=n-a-col,
                    success_rate=a/n, collision_rate=col/n,
                    failures=dict(Counter(x['failure'] for x in xs if not x['arrival'])))
    return {load: dict(group([x for x in rows if x['load'] == load]),
                       per_route={route: group([x for x in rows if x['load'] == load and x['route'] == route])
                                  for route in f.ROUTES}) for load in ('none', 'low')}


def evaluate(directory, model, count):
    meta = f.read(directory / 'manifest.json')
    b.torch.set_num_threads(1)
    agent = b.PPO(obs_dim=47, act_dim=2, **meta['hp'])
    agent.load(str(model))
    f.environment = lambda mode, seed, load='none', routes=None, variants=(): environment(seed, load, False, variants)
    rows = []
    for load in ('none', 'low'):
        for route in f.ROUTES:
            b.save(directory/'status.json', dict(stage='evaluating', load=load,
                   route=route, completed_cases=len(rows), expected_cases=32*count))
            rows.extend(f.evaluate(agent, 'N1', meta['seed'], load, [route], count, ()))
            b.save(directory / (model.stem + '.evaluation.json'), rows)
    b.save(directory / (model.stem + '.evaluation.json'), rows)
    b.save(directory / (model.stem + '.summary.json'), summary(rows))


class Log(b.Log):
    def log_train(self, step, metrics, verbose=False):
        super().log_train(step, dict(metrics, prepare_total=self.env.extra_total,
                                    prepare_events=self.env.extra_events), False)
        b.save(self.directory / 'status.json', dict(stage='training', step=step,
               budget=self.budget, prepare_total=self.env.extra_total,
               prepare_events=self.env.extra_events, updated=time.time()))
        if step >= getattr(self, 'next_save', 50000) or step == self.budget:
            self.agent.save(str(self.directory / f'policy-{step}.pt'))
            self.next_save = step + 50000


def train(spec):
    mode, seed, budget, count = spec
    b.torch.set_num_threads(1)
    f.OUT = OUT
    directory = OUT / 'runs' / f'{mode}-{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    hp = dict(b.PLAN['hparams'], seed=seed)
    env = environment(seed, prepare=mode == 'prepare300')
    agent = b.PPO(obs_dim=47, act_dim=2, buffer=b.RolloutBuffer(), **hp)
    # Identical per-seed state tensors are verified by the parent before training.
    initial = OUT / 'initial' / f'{seed}.pt'
    agent.load(str(initial))
    b.save(directory / 'manifest.json', dict(mode=mode, seed=seed, budget=budget, hp=hp,
           initialization='fresh, identical per-seed state_dict in both arms',
           initial_sha=b.digest(initial), map_sha=b.digest(b.MAP/'network.net.xml'),
           base_commit=b.COMMIT, action=b.ACTION, reward=b.REWARD,
           simulation=dict(b.SIMULATION, warmup_steps=300), observation=b.OBSERVATION,
           navigation_horizon=b.NAV.horizon, variants=[], train_load='low',
           traffic=f.read(f.prepare('low').parent/'manifest.json'),
           prepare_reward=dict(enabled=mode == 'prepare300', horizon=300, weight=.1,
                               duplicate_credit=False), evaluation_count=count))
    logger = Log(directory, agent, budget, env)
    started = time.monotonic()
    try:
        agent.learn(env, budget, logger=logger, eval_interval=0, exact_budget=True)
        assert agent.num_timesteps == budget and env.route_override_count == 0
        agent.save(str(directory/'model.pt'))
    except Exception as exc:
        b.save(directory/'status.json', dict(stage='error', error=repr(exc)))
        raise
    finally:
        env.close()
        logger.close()
    b.save(directory/'status.json', dict(stage='evaluating', step=budget))
    evaluate(directory, directory/'model.pt', count)
    result = dict(mode=mode, seed=seed, steps=budget, seconds=time.monotonic()-started,
                  model_sha=b.digest(directory/'model.pt'), results=f.read(directory/'model.summary.json'))
    b.save(directory/'complete.json', result)
    b.save(directory/'status.json', dict(stage='complete', step=budget))
    return result


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--seeds', type=int, nargs='+', default=[11,22,33])
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    OUT = args.output.resolve()
    f.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'initial').mkdir()
    b.torch.set_num_threads(1)
    for load in ('none', 'low'):
        f.prepare(load)
    b.save(OUT/'plan.json', dict(steps=args.steps, seeds=args.seeds, count=args.count,
           baseline='fresh; NOT meeting checkpoint continuation',
           backend='libsumo' if os.environ.get('IC_LIBSUMO') == '1' else 'traci',
           sumo_version=subprocess.check_output(['sumo', '--version'], text=True),
           torch_version=b.torch.__version__))
    paths = list(f.ROOT.glob('env/*.py')) + list(f.ROOT.glob('utils/*.py'))
    paths += list(f.ROOT.glob('algorithms/*.py')) + [f.ROOT/'train.py', b.MAP/'network.net.xml',
              b.MAP/'manifest.json', f.ROOT/'docs/experiments/ic-navigation-plan-v1.json']
    hashes = {}
    for path in paths:
        rel = path.relative_to(f.ROOT)
        target = OUT/'snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = b.digest(target)
    b.save(OUT/'source-hashes.json', hashes)
    for seed in args.seeds:
        hp = dict(b.PLAN['hparams'], seed=seed)
        first = b.PPO(obs_dim=47, act_dim=2, **hp)
        second = b.PPO(obs_dim=47, act_dim=2, **hp)
        assert all(b.torch.equal(value, second.policy.state_dict()[name])
                   for name, value in first.policy.state_dict().items())
        first.save(str(OUT/'initial'/f'{seed}.pt'))
    specs = [(mode, seed, args.steps, args.count) for seed in args.seeds
             for mode in ('baseline','prepare300')]
    # macOS spawn workers need the selected output, not a module default.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure,
                             initargs=(str(OUT),)) as pool:
        results = list(pool.map(train, specs))
    b.save(OUT/'comparison.json', results)
    print(json.dumps(results, ensure_ascii=False), flush=True)


def configure(output):
    global OUT
    OUT = Path(output)
    f.OUT = OUT


if __name__ == '__main__':
    main()
