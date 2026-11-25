"""Follow-up of the 600k fork-reward trial (+0.25 vs +1): pairing, replays, fresh hard starts (pmh).

Models: results/jys3_ppo_comparison/pure_f/terminal_ratio/fork_reward_team_600k/
        collision5_{11,22,33}/policy_600000.pt        (fork +0.25)
        collision5_fork1_{11,22,33}/policy_600000.pt  (fork +1)
No training. Every episode is driven by the deterministic policy, exactly as in
utils.jys3_pure_f.evaluate (same env class via jys3_terminal_ratio.setup, same
start lane rule, same traffic seed rule, observation noise reseeded per reset).

Commands
  aggregate  recompute the 192-episode totals from the raw eval files
             (eval_43110000_600000_{0..3}.json) and pair +0.25 / +1 rows by
             (training seed, route, case, traffic seed, start state).
  replay     re-drive all 24 evaluation sequences (2 rewards x 3 seeds x 4
             cases, 16 routes each); check every outcome matches the raw file;
             keep the end state and a step log of every episode.
  fresh      new hard starts: route 13 from lane 1 of 86198728 (only lane 0
             leads to the first exit 37927508) and route 16 from lane 0 of
             86198728 (lanes 1-4 lead to the main road 86198725); traffic seeds
             FRESH_SEEDS (fixed below, unused elsewhere); 3 training seeds x 2
             routes x 10 seeds = 60 episodes per reward.
  report     write the CSV files and the fresh seed list into docs/.
  gui        drive chosen episodes in sumo-gui for viewing
             (--cases arm:seed:route:lane:traffic_seed ...).

Usage: python -m utils.pmh_fork_reward_followup aggregate|replay|fresh|gui
"""
import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from utils import jys3_terminal_ratio as ratio

e = ratio.exp
j = e.j
RUN = ratio.ROOT / 'fork_reward_team_600k'
ARMS = ('collision5', 'collision5_fork1')
LABEL = {'collision5': '+0.25', 'collision5_fork1': '+1'}
SEEDS = (11, 22, 33)
STEP = 600000
HOLD = 43110000
OUT = j.m.ROOT / 'results/pmh_fork_reward_followup'
DOC = j.m.ROOT / 'docs/experiments/pmh-fork-reward-followup'
# Fresh traffic seeds: 47110000 + 100*k + route. Unused by training (seed*1e6 + n,
# n < 3000), screening (41110000+), the final evaluation (43110000+), the jys3
# references (34110000+, 39110000+) and the pmh DAgger sets (26-30M).
FRESH_BASE = 47110000
FRESH_SEEDS = {r: [FRESH_BASE + 100 * k + r for k in range(10)] for r in (13, 16)}
FRESH_LANE = {13: 1, 16: 0}
FIRST, EXIT13, MAIN16, WEAVE, LAST13 = '86198728', '169728835', '86198725', '170059994', '1063527748'


def outcome_of(info, cut):
    return ('arrival' if info['arrived'] else 'collision' if info['collided'] else 'stall' if info['stalled']
            else 'timeout' if cut else 'other')


def model_for(arm, seed):
    agent = e.agent(seed)
    agent.policy.load_state_dict(j.torch.load(RUN / f'{arm}_{seed}' / f'policy_{STEP}.pt', weights_only=True)['policy'])
    agent.policy.eval()
    return agent


def drive(env, agent, route, lane, traffic_seed, log):
    """One episode as in jys3_pure_f.evaluate; returns a row with end state and route-13/16 events."""
    name = f'route_{route:02d}'
    env.ego_depart_lane = str(min(lane, env.counts[env.routes[name][0]] - 1))
    obs, _ = env.reset(seed=traffic_seed, options={'route': name})
    start = [env.snapshot[i] for i in (0, 1, 7, -1)]
    path = env.routes[name]
    ev = dict(first_left=None, first_move_left=None, first_right=None, first_move_right=None, weave_lane=None)
    forks, steps, lanes_seen = [], [], []
    for t in range(1, 1001):
        edge, cur, _, idx, _, _, _, v, remaining = env.snapshot
        act = agent.predict(obs, deterministic=True)
        cmd = int(round(float(act[1]))) if abs(float(act[1])) > .5 else 0
        if edge == FIRST and idx == 0:
            if cmd > 0 and ev['first_left'] is None:
                ev['first_left'] = dict(t=t, remaining=round(remaining, 1), speed=round(v, 2), lane=cur)
            if cmd < 0 and ev['first_right'] is None:
                ev['first_right'] = dict(t=t, remaining=round(remaining, 1), speed=round(v, 2), lane=cur)
        if edge == WEAVE and ev['weave_lane'] is None:
            ev['weave_lane'] = cur
        obs, _, done, cut, info = env.step(act)
        post = env.snapshot
        moved = 0
        if post and not done and not edge.startswith(':') and post[0] == edge and post[1] != cur:
            moved = post[1] - cur
            if edge == FIRST and idx == 0:
                key = 'first_move_left' if moved > 0 else 'first_move_right'
                if ev[key] is None:
                    ev[key] = dict(t=t, remaining=round(remaining, 1), speed=round(v, 2), lane_from=cur, lane_to=post[1])
        if not edge.startswith(':') and (not lanes_seen or lanes_seen[-1][:2] != [edge, cur]):
            lanes_seen.append([edge, cur, round(remaining, 1)])
        if info['fork_passed']:
            forks.append(env.last_external[1])
        if log:
            steps.append([t, edge, cur, round(remaining, 1), round(v, 2), round(float(act[0]), 3), cmd, moved])
        if done or cut:
            break
    out = outcome_of(info, cut)
    end = env.snapshot
    nxt_lanes = None
    if end and not end[0].startswith(':') and end[3] + 1 < len(path):
        nxt_lanes = sorted(env.links.get(end[0], {}).get(path[end[3] + 1], ()))
    row = dict(route=route, lane=int(env.ego_depart_lane), traffic_seed=traffic_seed, start=start, outcome=out, steps=t,
               end_edge=end[0] if end else None, end_lane=end[1] if end else None,
               end_remaining=round(end[-1], 1) if end else None, end_speed=round(end[7], 2) if end else None,
               end_next_lanes=nxt_lanes, forks=forks, lanes=lanes_seen, **ev)
    if log:
        row['step_log'] = steps
    return row


def sequence(job):
    """Re-drive one evaluation sequence (16 routes) exactly as the raw eval file was produced."""
    arm, seed, case, gui = job
    j.torch.set_num_threads(1)
    ratio.setup(arm)
    agent = model_for(arm, seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        env = ratio.Env(arm, Path(tmp), seed)
        env.sim.gui = gui
        try:
            for route in range(1, 17):
                r = drive(env, agent, route, case % 2, HOLD + (case // 2) * 100 + route, log=True)
                rows.append(dict(arm=arm, seed=seed, case=case, **r))
        finally:
            env.close()
    return rows


def single(job):
    arm, seed, route, lane, traffic_seed, gui = job
    j.torch.set_num_threads(1)
    ratio.setup(arm)
    agent = model_for(arm, seed)
    with tempfile.TemporaryDirectory() as tmp:
        env = ratio.Env(arm, Path(tmp), seed)
        if gui:
            env.sim.gui = True
            env.sim.gui_delay = 60.
            env.sim.gui_zoom = 900.
        try:
            r = drive(env, agent, route, lane, traffic_seed, log=True)
        finally:
            env.close()
    return dict(arm=arm, seed=seed, **r)


# ---------------------------------------------------------------- aggregate
def raw_rows(arm, seed):
    return [r for c in range(4) for r in json.loads((RUN / f'{arm}_{seed}' / f'eval_{HOLD}_{STEP}_{c}.json').read_bytes())]


def key(r):
    return (r['route'], r['case'], r['traffic_seed'], tuple(r['start'][:2]), round(r['start'][2], 3), round(r['start'][3], 3))


def aggregate():
    OUT.mkdir(parents=True, exist_ok=True)
    final = json.loads((RUN / 'final.json').read_bytes())
    report, pairs = {}, []
    for arm in ARMS:
        tot = Counter()
        for s in SEEDS:
            rows = raw_rows(arm, s)
            c = Counter(r['outcome'] for r in rows)
            assert dict(c) == final['hold'][f'{arm}_{s}_{STEP}']['outcomes'], (arm, s)
            tot.update(c)
        report[arm] = dict(tot)
    for s in SEEDS:
        a, b = raw_rows(ARMS[0], s), raw_rows(ARMS[1], s)
        assert [key(r) for r in a] == [key(r) for r in b]
        for p, q in zip(a, b):
            pairs.append(dict(seed=s, route=p['route'], case=p['case'], start_lane=p['start'][1], traffic_seed=p['traffic_seed'],
                              outcome_025=p['outcome'], outcome_1=q['outcome']))
    by_route = {}
    for r in pairs:
        d = by_route.setdefault(r['route'], Counter())
        d['n'] += 1
        d['arr_025'] += r['outcome_025'] == 'arrival'
        d['arr_1'] += r['outcome_1'] == 'arrival'
        d['gained'] += r['outcome_025'] != 'arrival' and r['outcome_1'] == 'arrival'
        d['lost'] += r['outcome_025'] == 'arrival' and r['outcome_1'] != 'arrival'
    trans = Counter((r['seed'], r['outcome_025'] + '->' + r['outcome_1']) for r in pairs)
    result = dict(totals=report, by_route={k: dict(v) for k, v in sorted(by_route.items())},
                  transitions={f'{s}:{t}': n for (s, t), n in sorted(trans.items())})
    (OUT / 'aggregate.json').write_text(json.dumps(result, indent=1))
    with (OUT / 'paired_192.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- replay
def replay(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s, c, False) for a in ARMS for s in SEEDS for c in range(4)]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for block in pool.map(sequence, jobs) for r in block]
    mismatch = []
    for a in ARMS:
        for s in SEEDS:
            raw = {(r['route'], r['case']): r for r in raw_rows(a, s)}
            for r in rows:
                if r['arm'] == a and r['seed'] == s:
                    o = raw[r['route'], r['case']]
                    assert o['traffic_seed'] == r['traffic_seed']
                    if o['outcome'] != r['outcome'] or o['steps'] != r['steps']:
                        mismatch.append([a, s, r['route'], r['case'], o['outcome'], r['outcome'], o['steps'], r['steps']])
    (OUT / 'replay_check.json').write_text(json.dumps(dict(episodes=len(rows), mismatch=mismatch)))
    (OUT / 'replay_full_steps.json').write_text(json.dumps(rows))
    print('episodes', len(rows), 'mismatch', mismatch)


# ---------------------------------------------------------------- fresh hard starts
def fresh(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s, r, FRESH_LANE[r], ts, False) for a in ARMS for s in SEEDS for r in (13, 16) for ts in FRESH_SEEDS[r]]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(single, jobs))
    starts = {}
    for r in rows:
        k = (r['seed'], r['route'], r['traffic_seed'])
        starts.setdefault(k, set()).add((tuple(r['start'][:2]), round(r['start'][2], 4), round(r['start'][3], 3)))
    assert all(len(v) == 1 for v in starts.values()), 'start state differs between rewards'
    (OUT / 'fresh.json').write_text(json.dumps(rows))
    for a in ARMS:
        for route in (13, 16):
            c = Counter(r['outcome'] for r in rows if r['arm'] == a and r['route'] == route)
            print(LABEL[a], route, dict(c))


def gui(cases):
    os.environ.pop('IC_LIBSUMO', None)
    for spec in cases:
        arm, seed, route, lane, ts = spec.split(':')
        r = single((arm, int(seed), int(route), int(lane), int(ts), True))
        print(spec, r['outcome'], 'first_left', r['first_left'], 'move_left', r['first_move_left'], 'forks', r['forks'], flush=True)


def _ev(x, k):
    return None if not x else x[k]


def _row(r):
    last = r.get('step_log') or []
    tail = last[-150:]
    return dict(reward=LABEL[r['arm']], seed=r['seed'], case=r.get('case'), route=r['route'], start_lane=r['start'][1],
                start_speed=round(r['start'][2], 2), traffic_seed=r['traffic_seed'], outcome=r['outcome'], steps=r['steps'],
                end_edge=r['end_edge'], end_lane=r['end_lane'], end_remaining=r['end_remaining'], end_speed=r['end_speed'],
                end_next_lanes=' '.join(map(str, r['end_next_lanes'] or [])),
                first_fork_37927508='37927508' in r['forks'], exit13_169728835=EXIT13 in r['forks'],
                main16_86198725=MAIN16 in r['forks'], weave_lane=r['weave_lane'],
                first_left_rem=_ev(r['first_left'], 'remaining'), first_left_speed=_ev(r['first_left'], 'speed'),
                first_move_left_rem=_ev(r['first_move_left'], 'remaining'), first_move_left_speed=_ev(r['first_move_left'], 'speed'),
                first_right_rem=_ev(r['first_right'], 'remaining'), first_move_right_rem=_ev(r['first_move_right'], 'remaining'),
                first_move_right_speed=_ev(r['first_move_right'], 'speed'),
                last30s_left=sum(x[6] > 0 for x in tail) if tail else None,
                last30s_right=sum(x[6] < 0 for x in tail) if tail else None,
                lane_moves=sum(x[7] != 0 for x in last) if last else None)


def report():
    """CSV files for the README (needs replay_full_steps.json from the replay with step logs, and fresh.json)."""
    DOC.mkdir(parents=True, exist_ok=True)
    full = json.loads((OUT / 'replay_full_steps.json').read_bytes())
    fr = json.loads((OUT / 'fresh.json').read_bytes())
    for name, rows in (('episodes_192x2.csv', full), ('fresh_hard_60x2.csv', fr)):
        out = [_row(r) for r in rows]
        with (DOC / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    (DOC / 'fresh_seeds.json').write_text(json.dumps(dict(base=FRESH_BASE, rule='base + 100*k + route, k=0..9',
                                                          seeds=FRESH_SEEDS, start_lane=FRESH_LANE,
                                                          start_edge=FIRST), indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['aggregate', 'replay', 'fresh', 'gui', 'report'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cases', nargs='*', default=[])
    a = p.parse_args()
    if a.command == 'aggregate':
        aggregate()
    elif a.command == 'replay':
        replay(a.workers)
    elif a.command == 'fresh':
        fresh(a.workers)
    elif a.command == 'report':
        report()
    else:
        gui(a.cases)


if __name__ == '__main__':
    main()
