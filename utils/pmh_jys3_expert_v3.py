"""jys3 expert v3: the v2 expert plus a lane-end merge rule for a stopped vehicle (pmh).

Why: the round-6 DAgger student stalls at lane ends (6 of 17 failures in
docs/experiments/pmh-jys3-dagger-multi). There the v2 expert keeps labelling
"wait": its gap rule needs rear gap > 3 + 0.5*max(0, v + dv) + max(0, dv) m,
about 25 m when stopped next to 15 m/s traffic, which a stream almost never
offers. The expert itself never reaches that state, so DAgger only ever
teaches waiting.

v3 changes only the case "stopped (< 3 m/s) within LANE_END m of the lane end,
current lane does not lead where the route goes, no solid line":
  - accept the gap to the nearest useful lane if the front gap > 2.5 m, the
    rear gap > 2.0 m + 1.0 s * (follower closing speed) and the overlap flag
    for that side is clear; then change and accelerate gently (0.3);
  - otherwise keep the v2 label. (A first version crept towards the lane
    end instead; driving 256 cases it stalled 5 times against v2's 0, since
    v2 waits ~14 m back while creeping enters the 5 m stall zone.)
Every other state gets exactly the v2 label. The expert reads only the
observation and the fixed env constants (vmax, visibility, max accel/decel),
so stored demonstrations can be relabelled offline with relabel().

The policy never sees the expert at run time; it is a training label only.

Checks (python -m utils.pmh_jys3_expert_v3):
  takeover  the 6 stalled cases: the student drives until it has stood at the
            lane end for 2 s, then v2 or v3 takes over; outcome of each.
  drive     v2 and v3 drive the same 256 high-traffic cases (base 30010000).
"""
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.jys3 import environment, traci, MAX_LANES
from utils.jys3_bc_v2 import expert as expert_v2

LANE_END, STOPPED = 15.0, 3.0
FRONT_MIN, REAR_MIN, REAR_HEADWAY = 2.5, 2.0, 1.0
GO_ACCEL = 0.3


def expert_v3(obs, env):
    action = expert_v2(obs, env).copy()
    if action[1] or obs[38]:
        return action
    speed = float(obs[0]) * env.vmax
    remaining = 400. * float(obs[39])
    lane, count = np.rint(obs[32:34] * MAX_LANES).astype(int)
    links = obs[40:].reshape(MAX_LANES, 2)[:count]
    eligible = links[:, int(obs[34] > 0)] if obs[35] >= 0 else links.max(axis=1)
    if not eligible.any() or eligible[lane] or speed >= STOPPED or remaining > LANE_END:
        return action
    options = np.flatnonzero(eligible)
    desired = min(options, key=lambda i: abs(i - lane))
    direction = int(np.sign(desired - lane))
    index = 1 + 4 * (2 - direction)
    front_gap = float(obs[index]) * env.visibility
    rear_gap = -float(obs[index + 2]) * env.visibility
    rear_closing = float(obs[index + 3]) * env.vmax
    clear = obs[37 if direction == 1 else 36] == 0.
    if clear and front_gap > FRONT_MIN and rear_gap > REAR_MIN + REAR_HEADWAY * max(0., rear_closing):
        return np.array([GO_ACCEL, direction], np.float32)
    return action                       # no acceptable gap: keep the v2 label (wait where v2 waits)


def constants(env):
    return SimpleNamespace(vmax=env.vmax, visibility=env.visibility,
                           max_accel=env.max_accel, max_decel=env.max_decel)


def relabel(obs, consts):
    """v3 labels for stored observations (N, OBS_DIM)."""
    return np.stack([expert_v3(o, consts) for o in obs]).astype(np.float32)


# ---------------------------------------------------------------- checks
def _outcome(info):
    return ('arrival' if info['arrived'] else 'collision' if info['collided']
            else 'stall' if info['stalled'] else 'timeout')


def takeover(job):
    model, rep, base, target, which = job
    from utils.pmh_jys3_dagger_multi import policy
    torch.set_num_threads(1)
    agent = policy(model)
    driver = expert_v3 if which == 'v3' else expert_v2
    v = traci.vehicle
    with tempfile.TemporaryDirectory() as d:
        env = environment('high_real_ratio', 11, Path(d))
        try:
            for n in range(1, 17):
                route = f'route_{n:02d}'
                seed = base + 100 * rep + n
                env.ego_depart_lane = str(min(rep % 2, env.counts[env.routes[route][0]] - 1))
                obs, _ = env.reset(seed=seed, options={'route': route})
                stopped, switched = 0, None
                for t in range(1, 1001):
                    if seed == target and switched is None and env.ego_id in v.getIDList():
                        rem = traci.lane.getLength(v.getLaneID(env.ego_id)) - v.getLanePosition(env.ego_id)
                        stopped = stopped + 1 if (v.getSpeed(env.ego_id) < 0.5 and rem < 5) else 0
                        if stopped >= 10:
                            switched = t
                    if switched is not None:
                        action = driver(obs, env)
                    else:
                        action = agent.predict(obs, deterministic=True)
                    obs, _, done, cut, info = env.step(action)
                    if done or cut:
                        break
                if seed == target:
                    return dict(seed=seed, route=n, driver=which, switched_at=switched, outcome=_outcome(info))
        finally:
            env.close()


def drive(job):
    which, rep, base = job
    driver = expert_v3 if which == 'v3' else expert_v2
    rows = []
    with tempfile.TemporaryDirectory() as d:
        env = environment('high_real_ratio', 11, Path(d))
        try:
            for n in range(1, 17):
                route = f'route_{n:02d}'
                seed = base + 100 * rep + n
                env.ego_depart_lane = str(min(rep % 2, env.counts[env.routes[route][0]] - 1))
                obs, _ = env.reset(seed=seed, options={'route': route})
                for t in range(1, 1001):
                    obs, _, done, cut, info = env.step(driver(obs, env))
                    if done or cut:
                        break
                rows.append(dict(route=n, seed=seed, driver=which, outcome=_outcome(info)))
        finally:
            env.close()
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/pmh_jys3_expert_v3'
    out.mkdir(parents=True, exist_ok=True)
    fails = json.loads((root / 'docs/experiments/pmh-jys3-dagger-multi/failures_crash6.json').read_text())['failures']
    stalls = [r for r in fails if r['outcome'] != 'collision']
    base = 30010000
    model = str(root / 'models/pmh/jys3_dagger6_crash.pt')
    jobs = [(model, (r['seed'] - base) // 100, base, r['seed'], w) for r in stalls for w in ('v2', 'v3')]
    with ProcessPoolExecutor(10) as pool:
        take = list(pool.map(takeover, jobs))
        drives = [r for block in pool.map(drive, [(w, rep, base) for w in ('v2', 'v3') for rep in range(16)])
                  for r in block]
    report = dict(takeover=take,
                  drive={w: Counter(r['outcome'] for r in drives if r['driver'] == w) for w in ('v2', 'v3')})
    (out / 'checks.json').write_text(json.dumps(dict(report=report, drives=drives), indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
