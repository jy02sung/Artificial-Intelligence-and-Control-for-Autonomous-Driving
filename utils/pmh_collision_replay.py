"""Replay the measured-traffic collisions of a pmh_zipper_trial run and record who hit whom.

Evaluation (utils/navigation_followup.evaluate) is deterministic: one env per
route, reset(seed=40001+i) for i in range(count), policy.predict(deterministic).
This script rebuilds exactly that sequence for every route that had a
collision, checks that the same episodes collide at the same time, and for each
collision stores:

  - SUMO's collision record (collider, victim, type, lane, position, speeds)
  - the ego's last 3 s: lane, speed, accel command, lane-change request /
    applied / blocked by the gate, filter speed cap, leader and follower gaps
  - a cause label (see classify()).

Usage: python -m utils.pmh_collision_replay --run results/pmh_zipper_filter/runs/zipper_filter-11
"""
import argparse
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utils import pmh_zipper_trial as z

b, traci = z.b, z.traci
HISTORY = 15          # 3 s at 0.2 s


def neighbour(env, leader):
    try:
        got = env._neighbor(0, leader=leader)
    except Exception:
        got = None
    return None if got is None else [round(float(got[0]), 2), round(float(got[1]), 2)]


def replay_route(spec):
    run, route, targets, count = spec
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    mode, seed, out = manifest['mode'], manifest['seed'], run.parents[1]
    z.configure(str(out))
    b.torch.set_num_threads(1)
    agent = b.PPO(obs_dim=47, act_dim=2, **manifest['hp'])
    agent.load(str(run / 'model.pt'))
    env = z.environment(seed, 'measured', mode, out, train=False, variants=('MEASURE_ROUTE',))
    v = traci.vehicle
    found = []
    try:
        for i in range(count):
            simseed = 40001 + i
            obs, _ = env.reset(seed=simseed, options={'route': route})
            hist = deque(maxlen=HISTORY)
            while True:
                action = agent.predict(obs, deterministic=True)
                pre = None
                if env.ego_id in v.getIDList():
                    a = float(np.clip(np.asarray(action).reshape(-1)[env.accel_index], -1, 1))
                    pre = dict(t=round(env.step_count * env.step_length, 1), lane=v.getLaneID(env.ego_id),
                               pos=round(v.getLanePosition(env.ego_id), 1), speed=round(v.getSpeed(env.ego_id), 2),
                               accel_cmd=round(a, 2), leader=neighbour(env, True), follower=neighbour(env, False))
                blocked_before = env.filter_blocked_lc
                obs, reward, done, cut, info = env.step(action)
                if pre is not None:
                    pre.update(lc_request=int(info.get('lane_change', 0)),
                               lc_applied=bool(info.get('lane_change_applied')) and info.get('lane_change', 0) != 0,
                               lc_blocked=env.filter_blocked_lc > blocked_before,
                               speed_capped=bool(info.get('filter_speed_capped')))
                    hist.append(pre)
                if info.get('collided'):
                    cols = [dict(collider=c.collider, victim=c.victim, collider_type=c.colliderType,
                                 victim_type=c.victimType, collider_speed=round(c.colliderSpeed, 2),
                                 victim_speed=round(c.victimSpeed, 2), type=c.type, lane=c.lane,
                                 pos=round(c.pos, 1))
                            for c in traci.simulation.getCollisions()
                            if env.ego_id in (c.collider, c.victim)]
                    found.append(dict(run=str(run), ego=env.ego_id, mode=mode, training_seed=seed, route=route, seed=simseed,
                                      seconds=round(env.step_count * env.step_length, 1),
                                      expected=simseed in targets, sumo=cols, history=list(hist)))
                if done or cut:
                    break
    finally:
        env.close()
    return found


def classify(c):
    ego = c['ego']
    s = c['sumo'][0] if c['sumo'] else None
    h = c['history']
    lc_recent = any(x['lc_applied'] for x in h[-10:])           # last 2 s
    on_internal = h[-1]['lane'].startswith(':') if h else False
    if s is None:
        return 'unknown'
    ego_hits = s['collider'] == ego
    kind = s['type']
    if kind == 'frontal':
        return 'frontal'
    if lc_recent:
        return 'lane_change_ego_hits' if ego_hits else 'lane_change_hit_by_other'
    if on_internal or s['lane'].startswith(':'):
        return 'junction_ego_hits' if ego_hits else 'junction_hit_by_other'
    if ego_hits:
        return 'rear_end_ego_hits'
    # ego was hit from behind: did the ego brake hard just before?
    hard = any(x['accel_cmd'] <= -0.5 for x in h[-5:])
    stopped = h[-1]['speed'] < 0.5 if h else False
    if stopped:
        return 'hit_while_stopped'
    return 'hit_from_behind_after_hard_brake' if hard else 'hit_from_behind'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    specs = []
    for run in a.runs:
        rows = json.loads((run / 'model.evaluation.json').read_text())
        by_route = {}
        for r in rows:
            if r['load'] == 'measured' and r['collision']:
                by_route.setdefault(r['route'], {})[r['seed']] = r['seconds']
        specs += [(str(run), route, targets, a.count) for route, targets in sorted(by_route.items())]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        found = [c for got in pool.map(replay_route, specs) for c in got]
    expected = {(s[0], s[1], seed) for s in specs for seed in s[2]}
    got = {(c['run'], c['route'], c['seed']) for c in found}
    for c in found:
        c['cause'] = classify(c)
    report = dict(expected=len(expected), replayed=len(got), matched=len(expected & got),
                  missing=sorted(map(list, expected - got)), extra=sorted(map(list, got - expected)),
                  causes=Counter(c['cause'] for c in found),
                  ego_collider=Counter(c['sumo'][0]['collider'] == c['ego'] for c in found if c['sumo']),
                  sumo_types=Counter(c['sumo'][0]['type'] for c in found if c['sumo']),
                  other_types=Counter((c['sumo'][0]['victim_type'] if c['sumo'][0]['collider'] == c['ego']
                                       else c['sumo'][0]['collider_type']) for c in found if c['sumo']))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(report=report, collisions=found), indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
