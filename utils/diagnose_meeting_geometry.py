"""Read-only M5/M6/M8 topology audit; no SUMO, training, or production mutation."""
import ast
import csv
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/meeting_diagnosis_20261114/geometry'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def shortest(graph, start, goals):
    queue, seen = [(0., start, [start])], set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in seen:
            continue
        seen.add(node)
        if node in goals:
            return {'distance_m': round(cost, 2), 'path': path}
        for nxt, weight in graph[node]:
            heapq.heappush(queue, (cost + weight, nxt, path + [nxt]))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'scenarios/hv_high15000_entryprep'
    tree = ET.parse(source / 'network.net.xml')
    edges = {e.get('id'): e for e in tree.findall('edge') if not e.get('function')}
    lanes = {l.get('id'): l for e in tree.findall('edge') for l in e.findall('lane')}
    paths = {r.get('id'): r.get('edges').split() for r in ET.parse(source / 'highway.rou.xml').findall('route') if r.get('id').startswith('route_')}
    lengths = {k: max(float(l.get('length')) for l in e.findall('lane')) for k, e in edges.items()}
    counts = {k: len(e.findall('lane')) for k, e in edges.items()}
    links, incoming, outgoing, via = defaultdict(list), defaultdict(set), defaultdict(set), {}
    graph, edgegraph = defaultdict(list), defaultdict(list)
    connections = []
    def allowed(lane):
        return ('allow' not in lane.attrib or 'passenger' in lane.get('allow').split()) and 'passenger' not in lane.get('disallow', '').split()
    for c in tree.findall('connection'):
        a, b = c.get('from'), c.get('to')
        if a not in edges or b not in edges:
            continue
        x, y = int(c.get('fromLane')), int(c.get('toLane'))
        connections.append((a, b, x, y)); links[a, b].append((x, y))
        incoming[b].add(a); outgoing[a].add(b)
        internal = float(lanes[c.get('via')].get('length')) if c.get('via') in lanes else 0.
        via[a, b] = max(via.get((a, b), 0), internal)
        left, right = f'{a}_{x}', f'{b}_{y}'
        if allowed(lanes[left]) and allowed(lanes[right]):
            graph[left].append((right, float(lanes[left].get('length')) + internal))
    for a in edges:
        for b in outgoing[a]:
            edgegraph[a].append((b, lengths[a] + via[a, b]))
        for l in edges[a].findall('lane'):
            i = int(l.get('index'))
            for delta, permission in [(1, 'changeLeft'), (-1, 'changeRight')]:
                target = f'{a}_{i + delta}'
                if target in lanes and allowed(l) and allowed(lanes[target]) and (permission not in l.attrib or 'passenger' in l.get(permission).split()):
                    graph[l.get('id')].append((target, 0.))
    # Execute the actual pure navigation class, avoiding environment / TraCI imports.
    module = ast.parse((ROOT / 'env/navigation_observation.py').read_text(encoding='utf-8'))
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'RouteNavigation')
    scope = {'defaultdict': defaultdict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'RouteNavigation', 'exec'), scope)
    nav = scope['RouteNavigation'](counts, connections, lengths, link_lengths=via)
    events, alternatives, routes = [], [], []
    for rid, path in paths.items():
        row = {'route': rid, 'edges': path, 'entry_length_m': lengths[path[0]], 'entry_lane_count': counts[path[0]], 'entry_targets': sorted({a for a, b in links[path[0], path[1]]}), 'all_adjacent_edges_connected': all(links[a, b] for a, b in zip(path, path[1:]))}
        row['entry_lane_changes_by_start_lane'] = {str(i): min(abs(i-t) for t in row['entry_targets']) for i in range(counts[path[0]])}
        row['nav_at_start'] = nav.query(path, 0, 0, lengths[path[0]])
        row['start_navigation_by_horizon'] = {}
        for horizon in (100., 300., 500.):
            probe = scope['RouteNavigation'](counts, connections, lengths, horizon=horizon, link_lengths=via)
            row['start_navigation_by_horizon'][str(int(horizon))] = probe.query(path, 0, 0, lengths[path[0]])
        routes.append(row)
        for i, (a, b) in enumerate(zip(path, path[1:])):
            targets = sorted({x for x, y in links[a, b]})
            constrained = len(targets) < counts[a]
            events.append({'route': rid, 'index': i, 'edge': a, 'next': b, 'length_m': lengths[a], 'lanes': counts[a], 'target_lanes': targets, 'split': len(outgoing[a]) > 1, 'merge_into_next': len(incoming[b]) > 1, 'lane_count_drop': counts[b] < counts[a], 'navigation_constraint': constrained, 'route100_candidate': constrained and len(outgoing[a]) > 1, 'legacy_prep_candidate': len(outgoing[a]) > 1 and targets == [0] and counts[a] > 1, 'nav_at_edge_start': nav.query(path, i, 0, lengths[a]), 'lane_links': links[a, b]})
            for wrong in sorted(outgoing[a] - {b}):
                goals = {l.get('id') for l in edges[path[-1]].findall('lane') if allowed(l)}
                starts = sorted({f'{wrong}_{y}' for x, y in links[a, wrong]})
                alternatives.append({'route': rid, 'split_edge': a, 'prescribed_next': b, 'wrong_next': wrong, 'destination': path[-1], 'edge_topology_alternative': shortest(edgegraph, wrong, {path[-1]}), 'passenger_lane_alternatives_by_arrival_lane': {s: shortest(graph, s, goals) for s in starts}})
    original = ET.parse(ROOT / 'scenarios/hv_high15000/network.net.xml')
    oldedges = {e.get('id'): e for e in original.findall('edge') if not e.get('function')}
    changes = [{'edge': k, 'before_m': max(float(l.get('length')) for l in oldedges[k].findall('lane')), 'after_m': lengths[k]} for k in edges if k in oldedges and lengths[k] != max(float(l.get('length')) for l in oldedges[k].findall('lane'))]
    restrictions = {k: [{x: l.get(x) for x in ('id','allow','disallow','changeLeft','changeRight') if l.get(x)} for l in e.findall('lane')] for k,e in edges.items() if any(l.get('changeLeft') or l.get('changeRight') for l in e.findall('lane'))}
    def signature(t):
        return sorted(tuple(sorted(c.attrib.items())) for c in t.findall('connection'))
    summary = dict(route_count=len(routes), route_transition_count=len(events), navigation_constraint_count=sum(e['navigation_constraint'] for e in events), split_count=sum(e['split'] for e in events), merge_count=sum(e['merge_into_next'] for e in events), route100_count=sum(e['route100_candidate'] for e in events), legacy_prep_count=sum(e['legacy_prep_candidate'] for e in events), wrong_branch_count=len(alternatives), edge_alternative_count=sum(a['edge_topology_alternative'] is not None for a in alternatives), lane_alternative_count=sum(any(a['passenger_lane_alternatives_by_arrival_lane'].values()) for a in alternatives), connection_attributes_unchanged=signature(tree)==signature(original), length_changes=changes, lane_change_restricted_edges=restrictions, hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'network.net.xml', source/'highway.rou.xml']})
    save('summary.json', summary); save('routes.json', routes); save('events.json', events); save('alternatives.json', alternatives)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
