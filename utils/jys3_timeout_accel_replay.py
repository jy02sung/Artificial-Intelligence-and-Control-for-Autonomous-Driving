"""Paired acceleration-only replay of 300k screen timeouts, with original lateral commands."""
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_physical_scratch as scratch

v3 = scratch.v3
ROOT = scratch.ROOT.parent / "timeout_acceleration_300k"
STEP = 300000
FIELDS = "step edge lane speed lane_end front_gap accel_original accel_applied lateral mission_lane next_edge fork_distance".split()


def lane_state(env):
    edge, lane, path, idx, _, _, _, _, _ = env.snapshot
    expected = path[idx+1] if idx+1 < len(path) else None
    if edge.startswith(":"):
        key = v3.j.traci.vehicle.getLaneID(env.ego_id)
        dests = {v3.j.traci.lane.getEdgeID(link[0]) for link in v3.j.traci.lane.getLinks(key)}
        return ("internal_ok" if expected in dests else "internal_other"), expected
    if expected is None:
        return "final_road", None
    exits = env.links.get(edge, {})
    if lane in exits.get(expected, ()):
        return "connected", expected
    if any(lane in lanes for lanes in exits.values()):
        return "other_road_only", expected
    return "lane_drops", expected


def replay(env, model, row, original=None):
    name = f"route_{row['route']:02d}"
    env.ego_depart_lane = str(min(row["case"] % 2, env.counts[env.routes[name][0]]-1))
    obs, _ = env.reset(seed=row["traffic_seed"], options={"route":name})
    assert row["start"] == [env.snapshot[i] for i in (0,1,7,-1)]
    meter = v3.core.TimeoutDistances()
    trace = []
    slow = changes = 0
    enabled = False
    state_cache = {}
    for t in range(1,1001):
        edge,lane,_,idx,_,fork_dist,_,speed,left = env.snapshot
        cache_key=(edge,lane,idx)
        if cache_key not in state_cache:
            state_cache[cache_key] = lane_state(env)
        status, nxt = state_cache[cache_key]
        leader = v3.j.traci.vehicle.getLeader(env.ego_id, env.visibility)
        gap = float(leader[1]) if leader and leader[0] else None
        act = model.predict(obs, deterministic=True).copy()
        lon = float(act[0])
        slow = slow+1 if speed <= 0.5 else 0
        enabled |= slow >= 25
        if original is not None:
            act[1] = original[t-1][8]
            if enabled and speed < 5 and (gap is None or gap > 25):
                target = min(1.5, (5-speed)/env.step_length) / env.max_accel
                act[0] = max(lon, min(1., target))
                changes += abs(float(act[0])-lon) > 1e-6
        trace.append([t,edge,lane,round(speed,4),round(left,3),None if gap is None else round(gap,3),
                      lon,float(act[0]),float(act[1]),status,nxt,None if fork_dist is None else round(fork_dist,2)])
        obs, _, done, cut, info = env.step(act)
        metrics = meter.observe(env,t,info)
        if done or cut:
            break
    return trace,dict(outcome=info['outcome'],steps=t,changes=int(changes),distances=metrics,
                      intact=bool(info['route_intact']),off_route=info.get('off_route_detail'))


def run_batch(args):
    folder, rep = args
    arm, seed_str = folder.name.rsplit('_',1)
    seed = int(seed_str)
    rows = v3.read(folder/f"eval_{v3.SCREEN}_{STEP}_{rep}.json")
    model = v3.core.agent(seed)
    model.policy.load_state_dict(v3.j.torch.load(folder/f"policy_{STEP}.pt",weights_only=True)["policy"])
    model.policy.eval()
    with TemporaryDirectory() as tmp:
        env = scratch.physical.Env(arm,Path(tmp),seed)
        try:
            for row in rows:
                if row['outcome'] != 'timeout':
                    continue
                label=f"{folder.name}_{rep}_{row['route']:02d}"
                target=ROOT/(label+'.json')
                if target.exists():
                    continue
                baseline, result = replay(env,model,row)
                assert result['outcome']=='timeout' and result['distances']==row['timeout_distances'], (label,result,row['timeout_distances'])
                alternate, changed = replay(env,model,row,baseline)
                assert all(b[8]==a[8] for b,a in zip(baseline,alternate))
                with gzip.open(ROOT/(label+'.json.gz'),'wt',encoding='utf8') as fp:
                    json.dump(dict(fields=FIELDS,baseline=baseline,accelerated=alternate),fp,separators=(',',':'))
                v3.save(target,dict(arm=arm,seed=seed,route=row['route'],case=rep,traffic=row['traffic_seed'],
                                    baseline=result,accelerated=changed))
        finally:
            env.close()


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[(d,rep) for rep in range(4) for d in sorted(scratch.ROOT.iterdir()) if d.is_dir() and d.name.startswith(('S1_','S2_'))]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for _ in pool.map(run_batch,jobs):
            pass
