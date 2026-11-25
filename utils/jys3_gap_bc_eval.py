"""Measure gap acceptance on the student's trajectory and on common expert observations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib

import numpy as np
import torch

from utils.jys3_gap_bc import ROOT, SETS, EVAL_BASE, prior


def required_direction(x):
    lane, count = np.rint(x[32:34] * 5).astype(int)
    links = x[40:].reshape(5, 2)[:count]
    eligible = links[:, int(x[34] > 0)] if x[35] >= 0 else links.max(axis=1)
    choices = np.flatnonzero(eligible)
    if not len(choices) or eligible[lane]:
        return 0
    return int(np.sign(choices[np.argmin(abs(choices-lane))] - lane))


def new_event(edge, lane, direction, step):
    return dict(edge=edge, lane=int(lane), direction=direction, start=step, last=step,
                blocked=False, opened=None, after_wait=False, correct=None, keep=True,
                next_lane=False, windows=[], shadow={})


def drive(env, models, name, route, rep):
    seed = EVAL_BASE + rep*100 + route
    path = f'route_{route:02d}'
    env.ego_depart_lane = str(min(rep % 2, env.counts[env.routes[path][0]]-1))
    obs, _ = env.reset(seed=seed, options={'route': path})
    initial = [env.snapshot[i] for i in (0, 1, 7, -1)]
    obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()
    events = []
    active = None
    open_window = None
    last_external = None
    for step in range(1,1001):
        assert obs.shape == (52,) and not np.any(obs[50:])
        x = obs[:50]
        edge, lane = env.snapshot[:2]
        remaining = float(env.snapshot[-1])
        if not edge.startswith(':'):
            last_external = (edge, int(lane), remaining)
        direction = required_direction(x)
        expert = prior.data.bc.expert(x,env)
        action = expert if name == 'expert' else models[name].predict(x,deterministic=True)
        cmd = env.sim._quantize_lane_change(action[1])
        if active and ((edge,lane) != (active['edge'],active['lane']) or
                       (direction and active['direction'] != direction)):
            active = None
            open_window = None
        if active is None and direction and not x[38] and not edge.startswith(':'):
            active = new_event(edge,lane,direction,step)
            events.append(active)
        if active is not None:
            active['last'] = step
            allowed = bool(direction and expert[1] == active['direction'])
            if direction and not x[38] and not allowed:
                active['blocked'] = True
            if allowed:
                if active['opened'] is None:
                    active['opened'] = step
                    active['after_wait'] = active['blocked']
                    if name == 'expert':
                        active['shadow'] = {n:env.sim._quantize_lane_change(m.predict(x,deterministic=True)[1]) for n,m in models.items()}
                if open_window is None:
                    open_window = dict(start=step, end=step, response=None)
                    active['windows'].append(open_window)
                open_window['end'] = step
                if cmd == direction:
                    if active['correct'] is None:
                        active['correct'] = step
                    if open_window['response'] is None:
                        open_window['response'] = step
            else:
                open_window = None
            if active['opened'] is not None and cmd != 0:
                active['keep'] = False
        obs, _, done, cut, info = env.step(action)
        post = env.snapshot
        if active and post and post[0] == edge and post[1] != lane:
            active['next_lane'] = True
        if done or cut:
            break
    assert done or cut
    reason = info['outcome']
    src = (info.get('off_route_detail') or {}).get('src')
    if reason == 'off_route':
        assert src is not None
    if events:
        e = events[-1]
        same_last = last_external and (e['edge'],e['lane'])==last_external[:2]
        failed_here = ((reason=='off_route' and src==e['edge']) or
                       (reason=='stall') or (reason=='timeout' and last_external and last_external[2]<=12))
        if same_last and failed_here and e['opened'] is not None and e['keep'] and not e['next_lane']:
            e['missed_to_end'] = True
    return dict(route=route, rep=rep, seed=seed, start=initial, obs_sha256=obs_hash,
                steps=step,outcome=reason,events=events)


def load_project_model(folder,name):
    return prior.dagger.policy(folder/name/'bc.pt')


def shard(job):
    name,rep,folder,loader = job
    torch.set_num_threads(1)
    names = SETS if name=='expert' else [name]
    models = {n:loader(folder,n) for n in names}
    with prior.simulator() as env:
        assert env.sim.physical_forks and not env.sim.lc_safety_enabled
        return [drive(env,models,name,route,rep) for route in range(1,17)]


def summarize(rows):
    events = [e for r in rows for e in r['events'] if e['opened'] is not None]
    def metrics(group):
        delays = [round((e['correct']-e['opened'])*.2, 2) for e in group if e['correct'] is not None]
        windows = [w for e in group for w in e['windows']]
        return dict(opportunities=len(group), responses=len(delays),
                    immediate=sum(t==0 for t in delays), delay_mean_s=float(np.mean(delays)) if delays else None,
                    delay_median_s=float(np.median(delays)) if delays else None,
                    missed_to_end=sum(e.get('missed_to_end',False) for e in group),
                    windows=len(windows),windows_without_response=sum(w['response'] is None for w in windows))
    return dict(outcomes=dict(Counter(r['outcome'] for r in rows)), all=metrics(events),
                after_wait=metrics([e for e in events if e['after_wait']]),
                missed_routes=[r['route'] for r in rows if any(e.get('missed_to_end') for e in r['events'])])


def main(folder=ROOT,loader=load_project_model):
    names = [*SETS,'expert']
    results = {name:[] for name in names}
    jobs = [(name,rep,folder,loader) for rep in range(8) for name in names]
    with ProcessPoolExecutor(8) as pool:
        for (name,_,_,_), rows in zip(jobs,pool.map(shard,jobs)):
            results[name].extend(rows)
            prior.v3.save(folder/f'eval_{name}.json',results[name])
    starts = [[(r['seed'],r['start'],r['obs_sha256']) for r in results[n]] for n in names]
    assert starts[0] == starts[1] == starts[2] and len(set(r[0] for r in starts[0]))==128
    summary = {n:summarize(results[n]) for n in names}
    events = [e for row in results['expert'] for e in row['events'] if e['opened'] is not None]
    for name in SETS:
        summary[name]['common_first_gap']={}
        for key,group in [('all',events),('after_wait',[e for e in events if e['after_wait']])]:
            summary[name]['common_first_gap'][key]=dict(opportunities=len(group),
                immediate=sum(e['shadow'][name]==e['direction'] for e in group))
    prior.v3.save(folder/'evaluation.json',summary)
    print(summary,flush=True)


if __name__ == '__main__':
    main()
