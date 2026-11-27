import math
import utils.driving_env as j
from collections import defaultdict
import xml.etree.ElementTree as ET


def fork_offsets(net, connections):
    """Rank road geometry using edge IDs and lane-shape averages only."""
    shapes = {}
    for edge in ET.parse(net).findall("edge"):
        if edge.get("function"):
            continue
        lanes = [[tuple(map(float, point.split(",")[:2])) for point in lane.get("shape").split()]
                 for lane in edge.findall("lane")]
        def center(i):
            return tuple(sum(lane[i][axis] for lane in lanes) / len(lanes) for axis in (0, 1))
        shapes[edge.get("id")] = center(0), center(-2), center(-1)
    exits = defaultdict(set)
    for link in connections:
        exits[link[0]].add(link[1])
    offsets = {}
    for src, destinations in exits.items():
        if len(destinations) < 2:
            continue
        _, (px, py), (x, y) = shapes[src]
        norm = math.hypot(x-px, y-py)
        if not norm:
            raise ValueError(f"Degenerate incoming road tangent: {src}")
        for dst in destinations:
            dx, dy = shapes[dst][0]
            offsets[src, dst] = (-(y-py)*(dx-x) + (x-px)*(dy-y)) / norm
    return offsets


def lateral_directions(net, connections):
    offsets = fork_offsets(net, connections)
    exits = defaultdict(list)
    for src, dst in offsets:
        exits[src].append(dst)
    signs = {}
    for src, destinations in exits.items():
        ordered = sorted(destinations, key=lambda dst: offsets[src, dst])
        for i, dst in enumerate(ordered):
            if i and math.isclose(offsets[src, dst], offsets[src, ordered[i-1]], abs_tol=1e-6):
                raise ValueError(f"Indistinguishable road branches: {src}")
            signs[src, dst] = 2*i/(len(ordered)-1)-1
    return signs


def configure_roads(sim, net):
    # The direction function sees only ordered pairs of roads, never the ego or a lane ID.
    pairs = [(src, dst) for src, exits in sim.links.items() for dst in exits]
    sim.signs = lateral_directions(net, pairs)
    flags = {}
    for src, exits in sim.links.items():
        slots = sim.no_links.copy().reshape(j.MAX_LANES, 2)
        for dst, lanes in exits.items():
            sign = sim.signs.get((src, dst))
            for lane in lanes:
                if sign is None:
                    slots[lane, :] = 1.
                else:
                    slots[lane, int(sign > 0)] = 1.
        flags[src] = slots.ravel()
    sim.road_flags = flags
