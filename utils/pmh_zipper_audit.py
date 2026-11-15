"""HV-only zipper-merge audit on measured-demand scenarios (no AV).

Finds every merge in a network and measures, per merge, whether vehicles
from the two sides enter the shared lane alternately (zipper) and how long
the merging side waits:

* junction merge: one outgoing lane fed from two incoming edges.
* lane drop: a lane that ends on an edge while its neighbour continues
  (acceleration lane). Merges are the changes from the ending lane into
  its left neighbour.

Alternation is measured only inside dense platoons (consecutive entries
less than PLATOON_GAP seconds apart) where both sides compete. Random
interleaving of a p/q mix switches side 2pq of the time; a zipper switches
close to 2*min(p, q). A strict "main first" merge switches far less.

Usage:
  python -m utils.pmh_zipper_audit --scenario <dir with run.sumocfg> --seeds 1 2 3
"""
import argparse
import collections
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import statistics

import sumolib

PLATOON_GAP = 3.0      # s between successive entries counted as competing
STOP_SPEED = 0.5       # m/s
NEAR_END = 30.0        # m before a lane end counted as "waiting at the end"


def find_merges(net_file):
    net = sumolib.net.readNet(str(net_file))
    merges = []
    for node in net.getNodes():
        feeds = collections.defaultdict(set)
        for edge in node.getIncoming():
            for lane in edge.getLanes():
                for c in lane.getOutgoing():
                    feeds[c.getToLane().getID()].add(lane.getID())
        for target, lanes in feeds.items():
            if len(lanes) > 1:   # two lanes (same or different edge) share one lane
                merges.append(dict(kind='junction', id=f'J:{node.getID()}', node=node.getID(),
                                   node_type=node.getType(), target=target,
                                   sources=sorted(lanes)))
    for edge in net.getEdges():
        lanes = edge.getLanes()
        cont = [bool(l.getOutgoing()) for l in lanes]
        if any(cont) and not all(cont):
            for lane, ok in zip(lanes, cont):
                if not ok:
                    neighbour = lanes[lane.getIndex() + 1]
                    merges.append(dict(kind='lane_drop', id=f'D:{edge.getID()}', edge=edge.getID(),
                                       ending=lane.getID(), target=neighbour.getID(),
                                       length=lane.getLength()))
    return merges


def run(spec):
    scenario, seed, horizon, out = spec
    import libsumo as traci
    scenario, out = Path(scenario), Path(out)
    merges = find_merges(scenario / 'network.net.xml')
    traci.start(['sumo', '-c', str(scenario / 'run.sumocfg'), '--seed', str(seed),
                 '--no-step-log', 'true', '--no-warnings', 'true', '--duration-log.disable', 'true'])
    step = traci.simulation.getDeltaT()
    arrived = collisions = 0
    lane_of = {}                                   # vehicle -> last lane id
    events = collections.defaultdict(list)          # merge id -> [(t, side)]
    positions = collections.defaultdict(list)       # lane drop -> distance to end at change
    waits = collections.defaultdict(float)          # merge id -> vehicle-seconds stopped near end
    head_waits = collections.defaultdict(lambda: collections.defaultdict(float))
    mergers = collections.defaultdict(set)          # lane drop -> vehicles seen on the ending lane
    on_target = collections.defaultdict(set)        # lane drop -> vehicles on the shared lane last step
    watched = {}
    for m in merges:
        lanes = m['sources'] + [m['target']] if m['kind'] == 'junction' else [m['ending'], m['target']]
        for lane in lanes:
            watched.setdefault(lane, []).append(m)
    try:
        while traci.simulation.getTime() < horizon:
            traci.simulationStep()
            t = traci.simulation.getTime()
            arrived += traci.simulation.getArrivedNumber()
            collisions += traci.simulation.getCollidingVehiclesNumber()
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            active_now = set(traci.vehicle.getIDList())
            for m in merges:
                if m['kind'] == 'junction':
                    for src in m['sources']:
                        length = traci.lane.getLength(src)
                        for v in traci.lane.getLastStepVehicleIDs(src):
                            if traci.vehicle.getSpeed(v) < STOP_SPEED and length - traci.vehicle.getLanePosition(v) < NEAR_END:
                                waits[m['id']] += step
                                head_waits[m['id']][v] += step
                    for v in traci.lane.getLastStepVehicleIDs(m['target']):
                        prev = lane_of.get(v)
                        if prev in m['sources']:
                            events[m['id']].append((t, prev))
                else:
                    # Order is taken where the shared lane leaves the edge, so a
                    # merger counts at the point it actually passes in the queue.
                    length = m['length']
                    for v in traci.lane.getLastStepVehicleIDs(m['ending']):
                        mergers[m['id']].add(v)
                        if traci.vehicle.getSpeed(v) < STOP_SPEED and length - traci.vehicle.getLanePosition(v) < NEAR_END:
                            waits[m['id']] += step
                            head_waits[m['id']][v] += step
                    now = set(traci.lane.getLastStepVehicleIDs(m['target']))
                    for v in now - on_target[m['id']]:
                        if lane_of.get(v) == m['ending']:
                            positions[m['id']].append(length - traci.vehicle.getLanePosition(v))
                    for v in on_target[m['id']] - now:
                        gone = v not in active_now or traci.vehicle.getRoadID(v) != m['edge']
                        if gone:
                            events[m['id']].append((t, 'merge' if v in mergers[m['id']] else 'through'))
                            mergers[m['id']].discard(v)
                    on_target[m['id']] = now
            for lane in watched:
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    lane_of[v] = lane
            # forget vehicles that left the watched lanes long ago to bound memory
            if int(t / step) % 500 == 0:
                active = set(traci.vehicle.getIDList())
                lane_of = {v: l for v, l in lane_of.items() if v in active}
        stats = dict(end=traci.simulation.getTime(),
                     arrived=arrived, collisions=collisions,
                     pending=len(traci.simulation.getPendingVehicles()),
                     running=traci.vehicle.getIDCount())
    finally:
        traci.close()
    result = dict(scenario=str(scenario), seed=seed, horizon=horizon, sim=stats, merges=[])
    for m in merges:
        seq = events[m['id']]
        result['merges'].append(dict(m, **summarize(seq, waits[m['id']], head_waits[m['id']],
                                                    positions.get(m['id'], []))))
    out.mkdir(parents=True, exist_ok=True)
    (out / f'zipper-{seed}.json').write_text(json.dumps(result, indent=1))
    return result


def summarize(seq, wait, heads, positions):
    sides = collections.Counter(s for _, s in seq)
    switches = pairs = 0
    for (t0, a), (t1, b) in zip(seq, seq[1:]):
        if t1 - t0 <= PLATOON_GAP:
            pairs += 1
            switches += a != b
    n = sum(sides.values())
    shares = [c / n for c in sides.values()] if n else []
    p = min(shares) if len(shares) > 1 else 0.0
    out = dict(entries=n, sides=dict(sides), platoon_pairs=pairs,
               switch_rate=switches / pairs if pairs else None,
               random_switch=2 * p * (1 - p), zipper_switch=2 * p,
               stopped_vehicle_seconds=round(wait, 1),
               max_head_wait=round(max(heads.values()), 1) if heads else 0.0,
               vehicles_stopped_near_end=len(heads))
    if out['switch_rate'] is not None and p > 0:
        # 0 = random interleaving, 1 = perfect zipper, <0 = one side blocks the other
        span = out['zipper_switch'] - out['random_switch']
        out['zipper_index'] = (out['switch_rate'] - out['random_switch']) / span if span > 0 else None
    if positions:
        out['merge_distance_to_end_median'] = statistics.median(positions)
        out['late_merge_share'] = sum(d < 0.3 * max(positions) for d in positions) / len(positions)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--horizon', type=float, default=1800)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    specs = [(str(a.scenario.resolve()), s, a.horizon, str(a.output.resolve())) for s in a.seeds]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(run, specs))
    report(results, a.output)


def report(results, out):
    rows = []
    for r in results:
        for m in r['merges']:
            rows.append(dict(seed=r['seed'], id=m['id'], kind=m['kind'], entries=m['entries'],
                             switch=m['switch_rate'], random=m['random_switch'], zipper=m['zipper_switch'],
                             index=m.get('zipper_index'), stopped_s=m['stopped_vehicle_seconds'],
                             max_wait=m['max_head_wait'], late=m.get('late_merge_share'),
                             dist=m.get('merge_distance_to_end_median')))
    (Path(out) / 'summary.json').write_text(json.dumps(dict(
        runs=[dict(seed=r['seed'], **r['sim']) for r in results], merges=rows), indent=1))
    for r in results:
        print('seed', r['seed'], r['sim'])
    fmt = lambda v: '-' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
    print('seed id kind entries switch random zipper index stopped_s max_wait late dist')
    for x in rows:
        print(' '.join(fmt(v) for v in x.values()))


if __name__ == '__main__':
    main()
