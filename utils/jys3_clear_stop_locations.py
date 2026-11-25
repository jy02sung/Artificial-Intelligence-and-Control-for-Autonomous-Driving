"""Replay the 300k screening panel and localize every stopped episode."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from utils import jys3_clear_stop_trial as trial

OUTPUT = trial.ROOT / 'stop_locations_300k.json'


def location(env):
    edge, lane, path, idx, _, _, _, speed, remaining = env.snapshot
    tr = trial.v3.j.traci
    lane_id = tr.vehicle.getLaneID(env.ego_id)
    position = tr.vehicle.getLanePosition(env.ego_id)
    nxt = path[idx + 1] if idx + 1 < len(path) else None
    targets = sorted(env.links.get(edge, {}).get(nxt, ()))
    fork = None
    if not edge.startswith(':'):
        distance = remaining
        for a, b in zip(path[idx:], path[idx+1:]):
            if (a, b) in env.signs:
                fork = [a, b, round(distance, 1)]
                break
            distance += env.navigation.link_lengths.get((a, b), 0) + env.navigation.lengths[b]
    gaps = {}
    for offset, side in [(1, 'left'), (-1, 'right')]:
        if edge.startswith(':') or not (0 <= lane+offset < tr.edge.getLaneNumber(edge)):
            continue
        gaps[side] = [round(x[0], 1) if x and x[0] <= 120 else None
                      for x in env._scan_lane_neighbors(offset)]
    return dict(edge=edge,lane=lane,lane_id=lane_id,position=round(position,1),
                lane_end=round(remaining,1),speed=round(speed,2),mission=list(path),
                mission_index=idx,next_edge=nxt,connected_lanes=targets,next_fork=fork,
                side_gaps=gaps)


def replay(job):
    seed, rep = job
    v3 = trial.v3
    folder = trial.ROOT / f'{trial.ARM}_{seed}'
    refs = json.loads((folder/f'eval_{v3.SCREEN}_300000_{rep}.json').read_text())
    model = v3.core.agent(seed)
    model.policy.load_state_dict(v3.j.torch.load(folder/'policy_300000.pt',weights_only=True)['policy'])
    model.policy.eval()
    with TemporaryDirectory() as temp:
        env = trial.Env(trial.ARM,Path(temp),seed)
        try:
            for row in refs:
                row['model_seed'] = seed
                name = f"route_{row['route']:02d}"
                env.ego_depart_lane = str(min(rep%2,env.counts[env.routes[name][0]]-1))
                obs,_ = env.reset(seed=row['traffic_seed'],options={'route':name})
                assert [env.snapshot[i] for i in (0,1,7,-1)] == row['start']
                stopping = row['outcome'] in ('timeout','clear_stop','stall')
                window = 150 if row['outcome']=='timeout' else 50
                commands = Counter()
                gaps = []
                forks = []
                for t in range(1,1001):
                    act = model.predict(obs,deterministic=True)
                    tail = stopping and t > row['steps']-window
                    if tail:
                        commands['brake' if act[0]<0 else 'accelerate' if act[0]>0 else 'zero'] += 1
                        commands[{-1:'right',0:'keep',1:'left'}[int(act[1])]] += 1
                    obs,_,done,cut,info = env.step(act)
                    if info['fork_passed']:
                        forks.extend(info.get('passed_edges',[env.last_external[1]]))
                    if tail:
                        gaps.append(info['gap'])
                        if len(gaps) == 1:
                            row['window_start'] = location(env)
                    if done or cut:
                        assert (t, info['outcome'], forks) == (row['steps'],row['outcome'],row['forks'])
                        if stopping:
                            row['end'] = location(env)
                            row['window_commands'] = dict(commands)
                            row['window_front_min'] = min((v for v in gaps if v is not None),default=None)
                            row['end_front'] = gaps[-1]
                            row['window_front_le25'] = sum(v is not None and v<=25 for v in gaps)
                        break
                else:
                    raise AssertionError('unterminated replay')
                row.pop('track',None)
        finally:
            env.close()
    return refs


if __name__ == '__main__':
    with ProcessPoolExecutor(8) as pool:
        parts = pool.map(replay,[(seed,rep) for seed in trial.SEEDS for rep in range(4)])
        rows = [row for part in parts for row in part]
    assert len(rows)==128
    OUTPUT.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
    print(OUTPUT,Counter(row['outcome'] for row in rows))
