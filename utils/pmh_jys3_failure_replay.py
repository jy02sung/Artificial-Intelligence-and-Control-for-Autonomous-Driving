"""Replay the failed episodes of a jys3-schema policy and compare with the expert on the same cases.

Evaluation (utils.pmh_jys3_dagger_multi.drive) runs routes 1..16 one after the
other in one environment per rep, seed = base + 100*rep + route. This script
re-runs exactly those rep sequences with the same policy, checks that every
episode ends the same way, and for each failure keeps the last 10 s:

  edge / lane / position / distance to lane end / speed, the policy action
  (accel, lane command, applied), the expert's label for the same state,
  SUMO's lane-change verdict for the commanded direction, true neighbour gaps
  in the commanded lane, and SUMO's collision record.

It also lets the jys3 v2 expert drive the same rep sequences, so each failure
can be marked "expert also fails" or "imitation gap".

Usage: python -m utils.pmh_jys3_failure_replay --model models/pmh/jys3_dagger6_crash.pt \
          --eval docs/experiments/pmh-jys3-dagger-multi/final256_crash6.json --base 30010000 \
          --output results/pmh_jys3_failures/crash6.json
"""
import argparse
import json
import tempfile
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from utils.jys3 import environment, traci
from utils.jys3_bc_v2 import expert
from utils.pmh_jys3_dagger_multi import policy

UNSAFE = 2560 | 5120 | 8192 | 16384      # blocked by leader / follower, overlapping, insufficient space
KEEP = 50                                 # 10 s at 0.2 s


def replay(job):
    model, rep, base, failing = job
    torch.set_num_threads(1)
    agent = policy(model) if model != 'expert' else None
    out = []
    v = traci.vehicle
    with tempfile.TemporaryDirectory() as map_dir:
        env = environment('high_real_ratio', 11, Path(map_dir))
        try:
            for n in range(1, 17):
                route = f'route_{n:02d}'
                seed = base + 100 * rep + n
                lane = min(rep % 2, env.counts[env.routes[route][0]] - 1)
                env.ego_depart_lane = str(lane)
                obs, _ = env.reset(seed=seed, options={'route': route})
                hist = deque(maxlen=KEEP)
                for t in range(1, 1001):
                    label = expert(obs, env)
                    action = label if agent is None else agent.predict(obs, deterministic=True)
                    rec = None
                    if seed in failing and env.ego_id in v.getIDList():
                        lane_id = v.getLaneID(env.ego_id)
                        cmd = env._quantize_lane_change(float(np.asarray(action).reshape(-1)[1]))
                        verdict = None
                        if cmd and not lane_id.startswith(':'):
                            try:
                                verdict = bool(v.getLaneChangeState(env.ego_id, cmd)[0] & UNSAFE)
                            except traci.TraCIException:
                                pass
                        lead = env._neighbor(cmd, leader=True) if cmd else env._neighbor(0, leader=True)
                        foll = env._neighbor(cmd, leader=False) if cmd else env._neighbor(0, leader=False)
                        rec = dict(t=round(t * env.step_length, 1), lane=lane_id,
                                   pos=round(v.getLanePosition(env.ego_id), 1),
                                   remaining=round(traci.lane.getLength(lane_id) - v.getLanePosition(env.ego_id), 1),
                                   speed=round(v.getSpeed(env.ego_id), 2),
                                   accel=round(float(np.asarray(action).reshape(-1)[0]), 2), cmd=cmd,
                                   expert_accel=round(float(label[0]), 2),
                                   expert_cmd=env._quantize_lane_change(float(label[1])),
                                   sumo_unsafe=verdict,
                                   lead=None if lead is None else [round(lead[0], 1), round(lead[1], 1)],
                                   foll=None if foll is None else [round(foll[0], 1), round(foll[1], 1)])
                    obs, _, done, cut, info = env.step(action)
                    if rec is not None:
                        rec['applied'] = bool(info.get('lane_change_applied')) and info.get('lane_change', 0) != 0
                        hist.append(rec)
                    if done or cut:
                        break
                outcome = ('arrival' if info['arrived'] else 'collision' if info['collided']
                           else 'stall' if info['stalled'] else 'timeout')
                row = dict(route=n, seed=seed, lane=lane, steps=t, outcome=outcome)
                if seed in failing:
                    row['history'] = list(hist)
                    row['sumo'] = [dict(collider=c.collider, victim=c.victim, type=c.type, lane=c.lane,
                                        other_type=c.victimType if c.collider == env.ego_id else c.colliderType)
                                   for c in traci.simulation.getCollisions() if env.ego_id in (c.collider, c.victim)]
                    row['ego'] = env.ego_id
                out.append(row)
        finally:
            env.close()
    return out


def classify(r):
    h = r['history']
    if not h:
        return 'no_history'
    last = h[-1]
    if r['outcome'] == 'collision':
        s = r['sumo'][0] if r['sumo'] else None
        ego_hits = bool(s and s['collider'] == r['ego'])
        lc = [x for x in h[-10:] if x['applied']]
        if lc:
            x = lc[-1]
            where = 'edge_end' if x['remaining'] < 10 else 'edge_start' if x['pos'] < 15 else 'mid_edge'
            agree = 'expert_agrees' if x['expert_cmd'] == x['cmd'] else 'expert_keeps'
            return f"lane_change/{where}/{'unsafe' if x['sumo_unsafe'] else 'ok'}/{agree}/{'ego_hits' if ego_hits else 'hit_by_other'}"
        if s and (s['type'] == 'junction' or s['lane'].startswith(':')):
            return f"junction/{'ego_hits' if ego_hits else 'hit_by_other'}"
        return 'rear_end_ego_hits' if ego_hits else 'hit_from_behind'
    # stall / timeout
    wants = sum(x['expert_cmd'] != 0 for x in h[-25:])
    cmds = sum(x['cmd'] != 0 for x in h[-25:])
    return (f"{r['outcome']}/remaining_{'lt10' if last['remaining'] < 10 else 'ge10'}"
            f"/expert_wants_change_{'yes' if wants else 'no'}/policy_requests_{'yes' if cmds else 'no'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--eval', type=Path, required=True)
    p.add_argument('--base', type=int, default=30010000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.eval.read_text())
    failing = {r['seed'] for r in rows if r['outcome'] != 'arrival'}
    reps = sorted({(s - a.base) // 100 for s in failing})
    with ProcessPoolExecutor(a.workers) as pool:
        pol = [r for block in pool.map(replay, [(a.model, rep, a.base, failing) for rep in reps]) for r in block]
        exp = [r for block in pool.map(replay, [('expert', rep, a.base, failing) for rep in reps]) for r in block]
    orig = {r['seed']: r['outcome'] for r in rows}
    fails = [r for r in pol if r['seed'] in failing]
    exp_out = {r['seed']: r['outcome'] for r in exp}
    for r in fails:
        r['cause'] = classify(r)
        r['expert_outcome'] = exp_out[r['seed']]
    report = dict(failures=len(failing), reproduced=sum(orig[r['seed']] == r['outcome'] for r in fails),
                  replay_matches_all=all(orig[r['seed']] == r['outcome'] for r in pol),
                  causes=Counter(r['cause'] for r in fails),
                  expert_on_failed_cases=Counter(r['expert_outcome'] for r in fails),
                  expert_on_replayed_reps=Counter(r['outcome'] for r in exp))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(report=report, failures=fails), indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
