"""AV zipper-merge training on the zipper map with measured HV demand.

Environment
  map     maps/toegyewon_zipper (utils/pmh_zipper_network.py): entryprep
          geometry, all six merges are SUMO zipper junctions, so HV take
          turns with each other and with the AV.
  traffic utils/pmh_zipper_scenario.py, case low_exit25 = 13,949 veh/h
          (observed 2025-10-16 evening-peak mainline, assumed 25% exit share).

Arms (fresh PPO, identical initial weights per seed)
  pressure         prepare credit +0.1 and pressure 0.01 (best so far);
                   the 8 merge slots of the observation stay zero.
  pressure_zobs    + merge observation in the 10 unused ClarityEnv slots.
  pressure_zipper  + merge observation + zipper reward.

Merge observation (obs[37:45], all in [-1, 1])
  0 merge ahead on the route within 200 m (1 / -1)
  1 distance to the merge point / 200
  2 own lane feeds the merge (1), does not (-1), not yet known (0)
  3 competing vehicle on the other feeding lane (1 / -1)
  4 (its time to the merge - mine) / 5 s   (>0: I arrive first)
  5 (its speed - mine) / vmax
  6 (its distance - mine) / 50 m
  7 free space at the start of the shared lane / 50 m

Zipper reward (pressure_zipper only, training only)
  - per step, within 50 m of a merge the ego feeds, with a competitor:
    0.05 * clip((1.5 s - |time difference|) / 1.5 s, 0, 1): settle the order.
  - entering the merge link alone: +0.1; together with a competitor on the
    same merge link: -0.2. Once per merge per episode.
"""
import argparse
import copy
import csv
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault('IC_LIBSUMO', '1')
from utils import pmh_prepare_trial as t
from utils.pmh_zipper_scenario import build, sha
import env.ic_experiment as ic
from env.navigation_observation import RouteNavigation

f, b = t.f, t.b
traci = b.traci
ROOT = f.ROOT
NETWORK = ROOT / 'maps/toegyewon_zipper/network.net.xml'
OUT = ROOT / 'results/pmh_zipper_trial'
MODES = ('pressure', 'pressure_zobs', 'pressure_zipper')
LOADS = ('none', 'measured')
CASE = 'low_exit25'
SLOT = slice(37, 45)
HORIZON = 200.0
NEAR = 50.0
CONFLICT_T = 1.5
CONFLICT_W = 0.05
PASS_BONUS = 0.1
SIMUL_PENALTY = 0.2


def network_context(net=NETWORK):
    xml = ET.parse(net).getroot()
    edges = {e.get('id'): e for e in xml.findall('edge') if not e.get('function')}
    road = dict(b.ROAD, routes=b.MAN['routes'],
                edges=[(eid, e.get('from'), e.get('to'), len(e.findall('lane'))) for eid, e in edges.items()],
                connections=[(c.get('from'), c.get('to'), int(c.get('fromLane')), int(c.get('toLane')))
                             for c in xml.findall('connection') if c.get('from') in edges and c.get('to') in edges],
                solid_lane_edges=[eid for eid, e in edges.items()
                                  if any(l.get('changeLeft') or l.get('changeRight') for l in e.findall('lane'))])
    lengths = {e: max(float(l.get('length')) for l in x.findall('lane')) for e, x in edges.items()}
    internal = {l.get('id'): float(l.get('length')) for x in xml.findall('edge')
                if x.get('function') == 'internal' for l in x.findall('lane')}
    link = {}
    for c in xml.findall('connection'):
        if c.get('from') in edges and c.get('to') in edges:
            key = (c.get('from'), c.get('to'))
            link[key] = max(link.get(key, 0.), internal.get(c.get('via'), 0.))
    nav = RouteNavigation({e: len(x.findall('lane')) for e, x in edges.items()},
                          road['connections'], lengths, link_lengths=link)
    zipper_nodes = {j.get('id') for j in xml.findall('junction') if j.get('type') == 'zipper'}
    feeds = defaultdict(list)
    for c in xml.findall('connection'):
        if c.get('from') in edges and c.get('via') and edges[c.get('from')].get('to') in zipper_nodes:
            feeds[f"{c.get('to')}_{c.get('toLane')}"].append((f"{c.get('from')}_{c.get('fromLane')}", c.get('via')))
    merges = {}
    for target, src in feeds.items():
        if len(src) > 1:
            m = dict(target=target, sources=[s for s, _ in src], vias=[v for _, v in src])
            for s, _ in src:
                merges[s] = m
    by_pair = {}
    for s, m in merges.items():
        by_pair.setdefault((s.rsplit('_', 1)[0], m['target'].rsplit('_', 1)[0]), m)
    ramps = {eid for eid, e in edges.items() if e.get('type', '').endswith('_link')}
    return road, nav, merges, by_pair, ramps


ROAD, NAV, MERGES, MERGE_BY_PAIR, RAMPS = network_context()
VIA_MERGE = {via: m for m in MERGES.values() for via in m['vias']}
ic.RAMPS = RAMPS            # map-dependent ramp membership, as check_toegyewon_only does
EPISODES = []               # per-episode zipper stats, read by evaluate()


class ZipperEnv(t.TrialEnv):
    def __init__(self, *args, zipper_obs=False, zipper_reward=False, **kwargs):
        self.zipper_obs = zipper_obs
        self.zipper_reward = zipper_reward
        self.zip_conflict_total = 0.0
        self.zip_pass_total = 0.0
        self.zip_passes = self.zip_simultaneous = 0
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.passed = set()
        self.last_lane = None
        self.episode = dict(passes=0, simultaneous=0, merge_collision=False, collision_lane=None)
        EPISODES.append(self.episode)
        return result

    # ---- merge context -------------------------------------------------
    def merge_context(self):
        v = traci.vehicle
        ego = self.ego_id
        lane = v.getLaneID(ego)
        speed = v.getSpeed(ego)
        m, d, feeds, own = None, None, 0.0, None
        if lane in VIA_MERGE:
            m, d, feeds = VIA_MERGE[lane], 0.0, 1.0
            own = m['sources'][m['vias'].index(lane)]
        elif lane in MERGES:
            m, feeds, own = MERGES[lane], 1.0, lane
            d = traci.lane.getLength(lane) - v.getLanePosition(ego)
        elif not lane.startswith(':'):
            path = self.routes[self.episode_route]
            idx = v.getRouteIndex(ego)
            end = traci.lane.getLength(lane) - v.getLanePosition(ego)
            for j in range(idx, len(path) - 1):
                if j > idx:
                    end += NAV.link_lengths.get((path[j - 1], path[j]), 0.) + NAV.lengths[path[j]]
                if end > HORIZON:
                    break
                cand = MERGE_BY_PAIR.get((path[j], path[j + 1]))
                if cand:
                    m, d, feeds = cand, end, (-1.0 if j == idx else 0.0)
                    break
        if m is None:
            return None
        foes = []
        for src, via in zip(m['sources'], m['vias']):
            if src == own:
                continue
            for lane_id, at_merge in ((src, False), (via, True)):
                length = traci.lane.getLength(lane_id)
                for x in traci.lane.getLastStepVehicleIDs(lane_id):
                    if x != ego:
                        foes.append((0.0 if at_merge else length - v.getLanePosition(x), v.getSpeed(x), x))
        foe = min(foes) if foes else None
        target_ids = [x for x in traci.lane.getLastStepVehicleIDs(m['target']) if x != ego]
        space = min((v.getLanePosition(x) - v.getLength(x) for x in target_ids), default=HORIZON)
        if any(x != ego for via in m['vias'] for x in traci.lane.getLastStepVehicleIDs(via)):
            space = 0.0
        ctx = dict(merge=m, d=d, feeds=feeds, speed=speed, space=max(space, 0.0), foe=None)
        if foe:
            fd, fv, fid = foe
            ctx['foe'] = dict(id=fid, d=fd, v=fv,
                              dt=fd / max(fv, 0.5) - d / max(speed, 0.5))
        return ctx

    def merge_features(self, ctx):
        if ctx is None:
            return [-1., 1., 0., -1., 1., 0., 1., 1.]
        feats = [1., min(ctx['d'] / HORIZON, 1.), ctx['feeds']]
        foe = ctx['foe']
        if foe:
            feats += [1., float(np.clip(foe['dt'] / 5, -1, 1)),
                      float(np.clip((foe['v'] - ctx['speed']) / self.vmax, -1, 1)),
                      float(np.clip((foe['d'] - ctx['d']) / 50, -1, 1))]
        else:
            feats += [-1., 1., 0., 1.]
        return feats + [min(ctx['space'] / 50, 1.)]

    def _get_obs(self):
        obs = super()._get_obs()
        if self.zipper_obs and self.ego_id in traci.vehicle.getIDList():
            obs[SLOT] = self.merge_features(self.merge_context())
        return obs

    # ---- zipper reward and bookkeeping ---------------------------------
    def step(self, action):
        v = traci.vehicle
        before = v.getLaneID(self.ego_id) if self.ego_id in v.getIDList() else None
        obs, reward, done, cut, info = super().step(action)
        alive = self.ego_id in v.getIDList()
        if info.get('collided'):
            lane = before or ''
            self.episode['collision_lane'] = lane
            self.episode['merge_collision'] = lane in MERGES or lane in VIA_MERGE or any(
                lane == m['target'] for m in MERGES.values())
        extra = 0.0
        if alive and not done and not cut:
            lane = v.getLaneID(self.ego_id)
            if before in MERGES:
                m = MERGES[before]
                if (lane in m['vias'] or lane == m['target']) and m['target'] not in self.passed:
                    self.passed.add(m['target'])
                    together = any(x != self.ego_id for via in m['vias']
                                   for x in traci.lane.getLastStepVehicleIDs(via))
                    self.zip_passes += 1
                    self.episode['passes'] += 1
                    self.zip_simultaneous += together
                    self.episode['simultaneous'] += together
                    if self.zipper_reward:
                        extra += -SIMUL_PENALTY if together else PASS_BONUS
                        self.zip_pass_total += -SIMUL_PENALTY if together else PASS_BONUS
            if self.zipper_reward:
                ctx = self.merge_context()
                if ctx and ctx['feeds'] == 1.0 and ctx['foe'] and ctx['d'] < NEAR:
                    risk = float(np.clip((CONFLICT_T - abs(ctx['foe']['dt'])) / CONFLICT_T, 0, 1))
                    extra -= CONFLICT_W * risk
                    self.zip_conflict_total += CONFLICT_W * risk
        info['zipper_reward'] = extra
        return obs, reward + extra, done, cut, info


def scenario(load, out):
    cfg = out / 'traffic' / load / 'run.sumocfg'
    if not cfg.exists():
        build(cfg.parent, NETWORK, case=CASE, scale=0.0 if load == 'none' else 1.0)
    return cfg


def environment(seed, load, mode, out, train=True, variants=()):
    return ZipperEnv(str(scenario(load, out)), copy.deepcopy(ROAD),
                     dict(b.EGO, depart_lane='best', depart_speed='max'),
                     dict(b.SIMULATION, warmup_steps=300), b.OBSERVATION,
                     copy.deepcopy(b.ACTION), copy.deepcopy(b.REWARD),
                     dict(b.TRAFFIC, prefill={'enabled': False}), variants, seed,
                     navigation=NAV, clarity='baseline', prepare=train, pressure=train,
                     zipper_obs=mode != 'pressure', zipper_reward=train and mode == 'pressure_zipper')


def evaluate(directory, model, counts, out, mode, seed, hp):
    b.torch.set_num_threads(1)
    agent = b.PPO(obs_dim=47, act_dim=2, **hp)
    agent.load(str(model))
    f.environment = lambda m, s, load='none', routes=None, variants=(): \
        environment(s, load, mode, out, train=False, variants=variants)
    rows = []
    for load in LOADS:
        for route in f.ROUTES:
            start = len(EPISODES)
            got = f.evaluate(agent, 'N1', seed, load, [route], counts[load], ())
            stats = iter(EPISODES[start:])
            for row in got:
                if not row.get('insertion_failure'):
                    row['zipper'] = dict(next(stats))
            rows.extend(got)
    b.save(directory / f'{model.stem}.evaluation.json', rows)
    summary_ = summary(rows)
    b.save(directory / f'{model.stem}.summary.json', summary_)
    return summary_


def summary(rows):
    def group(xs):
        n = len(xs)
        a, c = sum(x['arrival'] for x in xs), sum(x['collision'] for x in xs)
        z = [x['zipper'] for x in xs if 'zipper' in x]
        passes = sum(s['passes'] for s in z)
        return dict(cases=n, arrivals=a, collisions=c, other=n - a - c,
                    success_rate=a / n if n else 0, collision_rate=c / n if n else 0,
                    merge_collisions=sum(s['merge_collision'] for s in z),
                    merge_passes=passes, simultaneous=sum(s['simultaneous'] for s in z),
                    failures=dict(Counter(x['failure'] for x in xs if not x['arrival'])))
    return {load: dict(group([x for x in rows if x['load'] == load]),
                       per_route={r: group([x for x in rows if x['load'] == load and x['route'] == r])
                                  for r in f.ROUTES}) for load in LOADS}


class Log(t.Log):
    def log_train(self, step, metrics, verbose=False):
        e = self.env
        super().log_train(step, dict(metrics, zip_passes=e.zip_passes, zip_simultaneous=e.zip_simultaneous,
                                     zip_conflict_total=e.zip_conflict_total,
                                     zip_pass_total=e.zip_pass_total), False)


def train(spec):
    mode, seed, budget, counts, out, every = spec
    out = Path(out)
    b.torch.set_num_threads(1)
    directory = out / 'runs' / f'{mode}-{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    hp = dict(b.PLAN['hparams'], seed=seed)
    env = environment(seed, 'measured', mode, out)
    agent = b.PPO(obs_dim=47, act_dim=2, buffer=b.RolloutBuffer(), **hp)
    initial = out / 'initial' / f'{seed}.pt'
    agent.load(str(initial))
    b.save(directory / 'manifest.json', dict(
        mode=mode, seed=seed, budget=budget, hp=hp, initial_sha=b.digest(initial),
        network=str(NETWORK.relative_to(ROOT)), network_sha=sha(NETWORK),
        traffic=json.loads((out / 'traffic/measured/manifest.json').read_text()),
        action=b.ACTION, reward=b.REWARD, simulation=dict(b.SIMULATION, warmup_steps=300),
        prepare_reward=dict(weight=.1, horizon=300), prepare_pressure=dict(weight=t.PRESSURE_WEIGHT),
        zipper_obs=mode != 'pressure', zipper_reward=(dict(conflict_w=CONFLICT_W, conflict_t=CONFLICT_T,
                                                        near=NEAR, pass_bonus=PASS_BONUS,
                                                        simultaneous_penalty=SIMUL_PENALTY)
                                                   if mode == 'pressure_zipper' else None),
        safety_gate=b.ACTION.get('lane_change_safety', {}).get('enabled'), eval_counts=counts))
    t.PRESSURE_WEIGHT = 0.01
    logger = Log(directory, agent, budget, env)
    logger.next_save = every
    started = time.monotonic()
    try:
        agent.learn(env, budget, logger=logger, eval_interval=0, exact_budget=True)
        agent.save(str(directory / 'model.pt'))
    finally:
        env.close()
        logger.close()
    b.save(directory / 'status.json', dict(stage='evaluating', step=budget))
    result = dict(mode=mode, seed=seed, steps=budget, train_seconds=time.monotonic() - started,
                  results=evaluate(directory, directory / 'model.pt', counts, out, mode, seed, hp))
    b.save(directory / 'complete.json', result)
    b.save(directory / 'status.json', dict(stage='complete', step=budget))
    return result


def evaluate_checkpoint(spec):
    run, checkpoint, counts, out = spec
    run, checkpoint = Path(run), Path(checkpoint)
    target = run / 'curve' / f'{checkpoint.stem}.summary.json'
    if target.exists():
        return str(target)
    target.parent.mkdir(exist_ok=True)
    meta = f.read(run / 'manifest.json')
    evaluate(run / 'curve', checkpoint, counts, Path(out), meta['mode'], meta['seed'], meta['hp'])
    return str(target)


def write_report(out, results):
    lines = ['# pmh 지퍼합류 비교 (지퍼 지도, 실측 교통량)', '',
             '| 교통 | 조건 | 평가 | 완주율 | 충돌률 | 합류 지점 충돌 | 합류 통과 | 동시 진입 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for load in LOADS:
        for mode in MODES:
            rs = [r['results'][load] for r in results if r['mode'] == mode]
            if not rs:
                continue
            n = sum(r['cases'] for r in rs)
            a, c = sum(r['arrivals'] for r in rs), sum(r['collisions'] for r in rs)
            mc, mp, ms = (sum(r[k] for r in rs) for k in ('merge_collisions', 'merge_passes', 'simultaneous'))
            lines.append(f'| {load} | {mode} | {n} | {a/n:.2%} | {c/n:.2%} | {mc} | {mp} | {ms} |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def configure(out):
    t.configure(out, 0.01, 0.5, 0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=300000)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--count-none', type=int, default=1, help='no-traffic eval is deterministic')
    p.add_argument('--count-measured', type=int, default=5)
    p.add_argument('--every', type=int, default=50000, help='checkpoint spacing')
    p.add_argument('--curve', action='store_true', help='also evaluate saved checkpoints')
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    out = a.output.resolve()
    counts = dict(none=a.count_none, measured=a.count_measured)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'initial').mkdir()
    for load in LOADS:
        scenario(load, out)
    b.save(out / 'plan.json', dict(steps=a.steps, seeds=a.seeds, modes=a.modes, counts=counts,
                                    case=CASE, network=str(NETWORK.relative_to(ROOT)), network_sha=sha(NETWORK)))
    for seed in a.seeds:
        hp = dict(b.PLAN['hparams'], seed=seed)
        first, second = b.PPO(obs_dim=47, act_dim=2, **hp), b.PPO(obs_dim=47, act_dim=2, **hp)
        assert all(b.torch.equal(v, second.policy.state_dict()[k]) for k, v in first.policy.state_dict().items())
        first.save(str(out / 'initial' / f'{seed}.pt'))
    specs = [(m, s, a.steps, counts, str(out), a.every) for s in a.seeds for m in a.modes]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=configure, initargs=(str(out),)) as pool:
        results = list(pool.map(train, specs))
    b.save(out / 'comparison.json', results)
    write_report(out, results)
    if a.curve:
        cps = []
        for run in sorted((out / 'runs').iterdir()):
            saved = {int(x.stem.split('-')[1]): x for x in run.glob('policy-*.pt')}
            for target in range(a.every, a.steps, a.every):
                step = min(saved, key=lambda s: abs(s - target))
                cps.append((str(run), str(saved[step]), counts, str(out)))
        with ProcessPoolExecutor(max_workers=a.workers, initializer=configure, initargs=(str(out),)) as pool:
            list(pool.map(evaluate_checkpoint, cps))
    print(f'Completed {len(results)} runs. Report: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
