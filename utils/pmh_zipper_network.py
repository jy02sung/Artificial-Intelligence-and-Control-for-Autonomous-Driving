"""Turn the three lane drops of the HV network into SUMO zipper merges.

In the source network (scenarios/hv_high15000_entryprep, which already has
zipper junctions at the three two-road merges) lane 0 of 206426298,
844484877 and 86198708 simply ends. HV then merge wherever SUMO's
lane-change model finds a gap (measured: mostly in the first 20-40% of the
lane) and nothing makes the through lane let a merging car in by turns.

Here the ending lane is connected to lane 0 of the next edge and the end
node becomes type="zipper", so both lanes feed the shared lane and SUMO's
zipper right-of-way alternates between them - for HV-HV and for an AV
approaching on either lane.

netconvert also re-derives geometry for the whole map (42 external lanes,
including the 51.07 m weave 170059994 -> 29.32 m). So, as in
utils/hv_flow_repair.py, only the three end nodes are transplanted from the
netconvert output: their junction element (type, internal lanes, request
table), their internal edge and their connections. Everything else is the
source file unchanged, and validate() proves it.

Usage: python -m utils.pmh_zipper_network [--source ...] [--output ...]
"""
import argparse
import copy
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scenarios/hv_high15000_entryprep/network.net.xml'
OUTPUT = ROOT / 'maps/toegyewon_zipper/network.net.xml'
NETCONVERT = Path('/Users/minho/src/sumo-1.27.1-mesa/bin/netconvert')
LANE_DROPS = {  # edge whose lane 0 ends -> (end node, next edge)
    '206426298': ('206426298.205', '206426298#1'),
    '844484877': ('844484877.189', '844484877#1'),
    '86198708': ('86198708.-50', '86198708#1'),
}


def connection_key(c):
    return tuple(c.get(k) for k in ('from', 'to', 'fromLane', 'toLane'))


def generate(source, work):
    work.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element('nodes')
    cons = ET.Element('connections')
    for edge, (node, nxt) in LANE_DROPS.items():
        ET.SubElement(nodes, 'node', id=node, type='zipper')
        ET.SubElement(cons, 'connection', {'from': edge, 'to': nxt, 'fromLane': '0', 'toLane': '0'})
    ET.ElementTree(nodes).write(work / 'zip.nod.xml', encoding='utf8')
    ET.ElementTree(cons).write(work / 'zip.con.xml', encoding='utf8')
    with (work / 'netconvert.log').open('w') as log:
        subprocess.run([str(NETCONVERT), '--sumo-net-file', str(source),
                        '--node-files', str(work / 'zip.nod.xml'),
                        '--connection-files', str(work / 'zip.con.xml'),
                        '--output-file', str(work / 'generated.net.xml')],
                       check=True, stdout=log, stderr=log)
    return work / 'generated.net.xml'


def transplant(source, generated, output):
    tree = ET.parse(source)
    net = tree.getroot()
    gen = ET.parse(generated).getroot()
    children = list(net)
    for edge, (node, nxt) in LANE_DROPS.items():
        internal = f':{node}_0'
        old_j, new_j = net.find(f"junction[@id='{node}']"), gen.find(f"junction[@id='{node}']")
        assert old_j.get('incLanes') == new_j.get('incLanes') and old_j.get('shape') == new_j.get('shape')
        net.insert(children.index(old_j), copy.deepcopy(new_j))
        net.remove(old_j)
        old_e, new_e = net.find(f"edge[@id='{internal}']"), gen.find(f"edge[@id='{internal}']")
        net.insert(list(net).index(old_e), copy.deepcopy(new_e))
        net.remove(old_e)
        for c in [c for c in net.findall('connection') if c.get('from') in (edge, internal)]:
            net.remove(c)
        for c in gen.findall('connection'):
            if c.get('from') in (edge, internal):
                net.append(copy.deepcopy(c))
        children = list(net)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='utf8', xml_declaration=True)
    return output


def validate(source, output):
    a, b = ET.parse(source).getroot(), ET.parse(output).getroot()
    touched = {node for node, _ in LANE_DROPS.values()}
    ea = {e.get('id'): e for e in a.findall('edge')}
    eb = {e.get('id'): e for e in b.findall('edge')}
    assert ea.keys() == eb.keys(), 'edge set changed'
    for eid, e in ea.items():
        if any(eid.startswith(f':{n}_') for n in touched):
            continue
        assert ET.tostring(e) == ET.tostring(eb[eid]), ('edge changed', eid)
    for j in a.findall('junction'):
        k = b.find(f"junction[@id='{j.get('id')}']")
        if j.get('id') in touched:
            assert k.get('type') == 'zipper' and k.get('shape') == j.get('shape')
        else:
            assert ET.tostring(j) == ET.tostring(k), ('junction changed', j.get('id'))
    before = {connection_key(c) for c in a.findall('connection') if not c.get('from').startswith(':')}
    after = {connection_key(c) for c in b.findall('connection') if not c.get('from').startswith(':')}
    added = after - before
    assert before <= after and added == {(e, n, '0', '0') for e, (_, n) in LANE_DROPS.items()}, added
    weave = b.find("edge[@id='170059994']/lane").get('length')
    assert weave == '51.07', weave
    return dict(external_edges_unchanged=True, other_junctions_unchanged=True,
                added_connections=sorted(added), weave_170059994_length=weave,
                zipper_junctions=sorted(j.get('id') for j in b.findall('junction')
                                        if j.get('type') == 'zipper'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    generated = generate(a.source, ROOT / 'results/pmh_zipper/netgen')
    transplant(a.source, generated, a.output)
    print(validate(a.source, a.output))


if __name__ == '__main__':
    main()
