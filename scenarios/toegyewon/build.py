"""Rebuild the standalone Toegyewon scenario from the vendored OSM snapshot.

No downloads, edits to the original highway, or training are performed.
"""
from pathlib import Path
import copy
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import sumolib

HERE = Path(__file__).resolve().parent

def binary(name):
    found = shutil.which(name)
    if found:
        return found
    for home in [os.environ.get('SUMO_HOME', ''), r'C:\Program Files (x86)\Eclipse\Sumo']:
        candidate = Path(home) / 'bin' / (name + ('.exe' if os.name == 'nt' else ''))
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f'{name}: install SUMO and set SUMO_HOME or PATH')

def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def clipped_source(params):
    """Clip segments at the simulation boundary, retaining interior OSM topology."""
    source = ET.parse(HERE / 'source.osm.xml').getroot()
    nodes = {n.get('id'): n for n in source.findall('node')}
    output = ET.Element('osm', version='0.6', generator='Toegyewon scenario')
    used, ways = {}, []
    xmin,ymin,xmax,ymax = params['bbox_lon_lat']
    serial = 90000000000
    for way in source.findall('way'):
        refs = [n.get('ref') for n in way.findall('nd')]
        chain = []
        chunks = []
        for a,b in zip(refs, refs[1:]):
            x,y = float(nodes[a].get('lon')),float(nodes[a].get('lat'))
            xx,yy = float(nodes[b].get('lon')),float(nodes[b].get('lat'))
            dx,dy = xx-x,yy-y
            lo,hi = 0.,1.
            for p,q in [(-dx,x-xmin),(dx,xmax-x),(-dy,y-ymin),(dy,ymax-y)]:
                if p == 0:
                    if q < 0: lo,hi=1.,0.
                elif p < 0: lo=max(lo,q/p)
                else: hi=min(hi,q/p)
            if lo >= hi:
                if chain: chunks.append(chain);chain=[]
                continue
            segment=[]
            for t,original in [(lo,a),(hi,b)]:
                if t in (0.,1.):
                    ident=original;used[ident]=copy.deepcopy(nodes[ident])
                else:
                    serial+=1;ident=str(serial)
                    used[ident]=ET.Element('node',id=ident,lon=str(x+t*dx),lat=str(y+t*dy))
                segment.append(ident)
            if chain and chain[-1] != segment[0]: chunks.append(chain);chain=[]
            chain.extend(segment if not chain else segment[1:])
        if chain: chunks.append(chain)
        for index,chain in enumerate(chunks):
            ident=way.get('id') if index==0 else str(int(way.get('id'))*100+index)
            w=ET.Element('way',id=ident)
            for ref in chain: ET.SubElement(w,'nd',ref=ref)
            tags={t.get('k'):t.get('v') for t in way.findall('tag')}
            if tags.get('highway','').endswith('_link'):
                tags.setdefault('maxspeed',str(params['default_link_speed_kmh']))
                tags.setdefault('lanes','1')
            elif tags.get('highway')=='trunk':
                tags.setdefault('maxspeed',str(params['default_trunk_speed_kmh']))
                tags.setdefault('lanes',str(params['default_trunk_lanes']))
            for k,v in tags.items(): ET.SubElement(w,'tag',k=k,v=v)
            ways.append(w)
    for n in used.values(): output.append(n)
    for w in ways: output.append(w)
    write_xml(output,HERE/'clipped.osm.xml')

def build():
    p=json.loads((HERE/'parameters.json').read_text(encoding='utf-8'))
    clipped_source(p)
    subprocess.run([binary('netconvert'),'--osm-files','clipped.osm.xml',
        '--output-file','toegyewon.net.xml','--no-turnarounds',
        '--keep-edges.by-vclass','passenger','--junctions.corner-detail','5',
        '--output.street-names','true','--output.original-names','true'],cwd=HERE,check=True)
    net=sumolib.net.readNet(str(HERE/'toegyewon.net.xml'))
    # Anchors are OSM way IDs from the attached map, not arbitrary shortest paths.
    # Entry/exit edges are selected geometrically on the four open boundaries.
    motorway=[e for e in net.getEdges() if e.getType()=='highway.motorway']
    nw_in=min((e for e in motorway if not e.getIncoming()),key=lambda e:e.getFromNode().getCoord()[0])
    south_in=min((e for e in motorway if not e.getIncoming()),key=lambda e:e.getFromNode().getCoord()[1])
    nw_out=min((e for e in motorway if not e.getOutgoing()),key=lambda e:e.getToNode().getCoord()[0])
    south_out=min((e for e in motorway if not e.getOutgoing()),key=lambda e:e.getToNode().getCoord()[1])
    def anchor(prefix):
        matches=[e for e in net.getEdges() if e.getID()==prefix] or [e for e in net.getEdges() if e.getID().split('#')[0]==prefix]
        if len(matches)!=1: raise ValueError(f'Ambiguous/missing anchor {prefix}: {[e.getID() for e in matches]}')
        return matches[0]
    # Road 47 can be split at ramp connections. Select its outermost pieces.
    eastbound=[e for e in net.getEdges() if e.getID().split('#')[0]=='781184662']
    westbound=[e for e in net.getEdges() if e.getID().split('#')[0]=='169929503']
    west_in=min(eastbound,key=lambda e:e.getFromNode().getCoord()[0])
    east_out=max(eastbound,key=lambda e:e.getToNode().getCoord()[0])
    east_in=max(westbound,key=lambda e:e.getFromNode().getCoord()[0])
    west_out=min(westbound,key=lambda e:e.getToNode().getCoord()[0])
    anchors={
        'nw_to_east':[nw_in,anchor('170059988'),anchor('170059994'),anchor('1063527748'),east_out],
        'south_to_west':[south_in,anchor('37927512'),anchor('170059994'),anchor('170059992'),west_out],
        'nw_to_south':[nw_in,anchor('37927509'),south_out],
        'south_to_nw':[south_in,anchor('844484878#0'),nw_out],
        'west_to_east':[west_in,east_out],
        'east_to_west':[east_in,west_out],
        'west_to_south':[west_in,anchor('544293888#0'),south_out],
    }
    routes={}
    for name,stops in anchors.items():
        edges=[]
        for a,b in zip(stops,stops[1:]):
            path,_=net.getShortestPath(a,b,vClass='passenger')
            if path is None: raise ValueError(f'No route {name}: {a.getID()} -> {b.getID()}')
            edges.extend(path if not edges else path[1:])
        routes[name]=' '.join(e.getID() for e in edges)
    root=ET.Element('routes')
    ET.SubElement(root,'vType',id='human',carFollowModel='Krauss',color='0.3,0.65,1',
        accel='2.6',decel='4.5',sigma='0.5',tau='1.2',length='5',minGap='2.5',lcStrategic='1')
    ET.SubElement(root,'vType',id='bus',vClass='bus',guiShape='bus',carFollowModel='IDM',
        color='1,0.75,0.2',length='12',minGap='3',accel='1.5',decel='4',tau='1.5')
    ET.SubElement(root,'vTypeDistribution',id='background',vTypes='human bus',probabilities='0.95 0.05')
    ET.SubElement(root,'vType',id='ego_baseline',carFollowModel='IDM',color='1,0,0',
        accel='2.6',decel='4.5',tau='1.2',minGap='2.5')
    for name,edges in routes.items(): ET.SubElement(root,'route',id=name,edges=edges)
    for name,rate in p['traffic_vehs_per_hour'].items():
        ET.SubElement(root,'flow',id=name,type='background',route=name,begin='0',
            end=str(p['duration_s']),vehsPerHour=str(rate),departLane='best',departSpeed='max')
    ET.SubElement(root,'vehicle',id='ego',type='ego_baseline',route='nw_to_east',
        depart=str(p['ego_depart_s']),departLane='best',departSpeed='max')
    write_xml(root,HERE/'toegyewon.rou.xml')
    cfg=ET.Element('configuration')
    inp=ET.SubElement(cfg,'input')
    ET.SubElement(inp,'net-file',value='toegyewon.net.xml')
    ET.SubElement(inp,'route-files',value='toegyewon.rou.xml')
    time=ET.SubElement(cfg,'time');ET.SubElement(time,'step-length',value=str(p['step_length_s']))
    ET.SubElement(time,'end',value=str(p['duration_s']+300))
    process=ET.SubElement(cfg,'processing')
    ET.SubElement(process,'collision.check-junctions',value='true')
    ET.SubElement(process,'time-to-teleport',value='120')
    gui=ET.SubElement(cfg,'gui_only');ET.SubElement(gui,'gui-settings-file',value='view.xml')
    write_xml(cfg,HERE/'toegyewon.sumocfg')
    view=ET.Element('viewsettings')
    scheme=ET.SubElement(view,'scheme',name='Toegyewon')
    ET.SubElement(scheme,'background',backgroundColor='232,237,229')
    ET.SubElement(scheme,'edges',laneShowBorders='true')
    ET.SubElement(scheme,'vehicles',vehicleMode='0',vehicleQuality='2',vehicle_minSize='0',vehicle_exaggeration='1')
    ET.SubElement(view,'viewport',zoom='100',x='450',y='610')
    ET.SubElement(view,'delay',value='30')
    write_xml(view,HERE/'view.xml')
    (HERE/'routes.json').write_text(json.dumps(routes,indent=2),encoding='utf-8')
    print(f'Built {len(net.getEdges())} edges and {len(routes)} routes; exactly one ego.')

if __name__=='__main__': build()
