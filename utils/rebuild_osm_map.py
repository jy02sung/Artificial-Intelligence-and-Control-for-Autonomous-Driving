"""Rebuild the review map directly from the user-supplied OSM, no hand-shaped roads."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'maps/toegyewon_osm'
ROAD_TYPES={'motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service','road'}

def build():
    from env.sumo_env import traci  # installed SUMO discovery only
    from sumolib import checkBinary
    original=ET.parse(DEST/'source.osm').getroot()
    ways={w.get('id'):w for w in original.findall('way') if any(t.get('k')=='highway' and t.get('v') in ROAD_TYPES for t in w.findall('tag'))}
    referenced={n.get('ref') for w in ways.values() for n in w.findall('nd')}
    nodes={n.get('id'):n for n in original.findall('node') if n.get('id') in referenced}
    prepared=ET.Element('osm',version='0.6',generator='project road-only input filter')
    bounds=original.find('bounds');prepared.append(bounds)
    for node in nodes.values():prepared.append(node)
    for way in ways.values():prepared.append(way)
    dropped=[];kept=[]
    for relation in original.findall('relation'):
        tags={t.get('k'):t.get('v') for t in relation.findall('tag')}
        if tags.get('type')!='restriction':continue
        complete=all((m.get('ref') in ways if m.get('type')=='way' else m.get('ref') in nodes if m.get('type')=='node' else False) for m in relation.findall('member'))
        if complete:prepared.append(relation);kept.append(relation.get('id'))
        else:dropped.append(relation.get('id'))
    ET.ElementTree(prepared).write(DEST/'road-input.osm',encoding='utf8',xml_declaration=True)
    bbox=','.join(bounds.get(k) for k in ('minlon','minlat','maxlon','maxlat'))
    args=[checkBinary('netconvert'),'--osm-files','road-input.osm','--output-file','full-import.net.xml','--keep-edges.by-vclass','passenger,bus','--keep-edges.in-geo-boundary',bbox,'--geometry.remove','false','--ramps.guess','false','--osm.layer-elevation','0','--osm.turn-lanes','true','--osm.lane-access','true','--no-turnarounds','true','--junctions.corner-detail','8','--junctions.internal-link-detail','12','--output.original-names','true']
    result=subprocess.run(args,cwd=DEST,capture_output=True,text=True,encoding='utf8',errors='replace')
    (DEST/'conversion-warnings.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    result.check_returncode()
    scope=json.loads((DEST/'scope.json').read_text(encoding='utf8'))
    full=ET.parse(DEST/'full-import.net.xml').getroot()
    existing={e.get('id') for e in full.findall('edge')}
    missing=set(scope['edges'])-existing
    if missing:raise ValueError(f'Scope edges missing from source import: {missing}')
    overrides=json.loads((DEST/'lane-overrides.json').read_text(encoding='utf8'))
    markings=json.loads((DEST/'markings.json').read_text(encoding='utf8'))
    corrections=ET.Element('edges')
    for eid,item in overrides.items():
        if eid not in scope['edges']:raise ValueError(f'Override outside selected scope: {eid}')
        ET.SubElement(corrections,'edge',id=eid,numLanes=str(item['num_lanes']))
    full_edges={e.get('id'):e for e in full.findall('edge')}
    for eid in markings['solid']:
        if eid not in scope['edges']:raise ValueError(f'Marking outside scope: {eid}')
        edge=next((e for e in corrections if e.get('id')==eid),None)
        if edge is None:edge=ET.SubElement(corrections,'edge',id=eid)
        count=overrides.get(eid,{}).get('num_lanes',len(full_edges[eid].findall('lane')))
        for i in range(count):
            attrs={'index':str(i)}
            if i<count-1:attrs['changeLeft']=markings['solid_allowed_classes']
            if i>0:attrs['changeRight']=markings['solid_allowed_classes']
            ET.SubElement(edge,'lane',**attrs)
    ET.ElementTree(corrections).write(DEST/'lane-overrides.edg.xml',encoding='utf8',xml_declaration=True)
    # A net.xml import keeps obsolete lane links unless explicitly reset (SUMO PlainXML).
    affected=set(overrides)|set(markings['solid'])
    incoming={c.get('from') for c in full.findall('connection') if c.get('to') in affected}
    reset_edges=(affected|incoming)&set(scope['edges'])
    links=ET.Element('connections')
    for eid in sorted(reset_edges):ET.SubElement(links,'connection',{'from':eid,'reset':'true'})
    ET.ElementTree(links).write(DEST/'connection-resets.con.xml',encoding='utf8',xml_declaration=True)
    crop_args=[checkBinary('netconvert'),'--sumo-net-file','full-import.net.xml','--edge-files','lane-overrides.edg.xml','--connection-files','connection-resets.con.xml','--keep-edges.explicit',','.join(scope['edges']),'--output-file','network.net.xml','--no-turnarounds','true','--geometry.remove','false']
    crop=subprocess.run(crop_args,cwd=DEST,capture_output=True,text=True,encoding='utf8',errors='replace')
    with (DEST/'conversion-warnings.txt').open('a',encoding='utf8') as log:log.write('\nScope crop:\n'+crop.stdout+'\n'+crop.stderr)
    crop.check_returncode()
    net=ET.parse(DEST/'network.net.xml').getroot();external=[e for e in net.findall('edge') if e.get('function')!='internal']
    audit=[]
    for e in external:
        way_id=e.get('id').lstrip('-').split('#')[0];way=ways.get(way_id)
        tags={t.get('k'):t.get('v') for t in way.findall('tag')} if way is not None else {}
        audit.append(dict(edge=e.get('id'),osm_way=way_id,lanes=len(e.findall('lane')),osm_lanes=tags.get('lanes'),osm_directional_lanes={k:v for k,v in tags.items() if k.startswith('lanes:')},oneway=tags.get('oneway'),highway=tags.get('highway'),maxspeed=tags.get('maxspeed'),layer=tags.get('layer'),bridge=tags.get('bridge'),turn_lanes={k:v for k,v in tags.items() if k.startswith('turn:lanes')},lane_override=overrides.get(e.get('id')),marking_status='solid_verified' if e.get('id') in markings['solid'] else 'dashed_verified' if e.get('id') in markings['dashed'] else 'mixed_unresolved' if e.get('id') in markings['unresolved'] else 'not_applicable' if len(e.findall('lane'))==1 else 'unverified',lane_source='Project override ('+overrides[e.get('id')].get('confidence','visible')+')' if e.get('id') in overrides else 'OSM tag (directional split may apply)' if 'lanes' in tags or any(k.startswith('lanes:') for k in tags) else 'SUMO road-type default, unverified'))
    report=dict(source_sha256=hashlib.sha256((DEST/'source.osm').read_bytes()).hexdigest(),source_bounds=bounds.attrib,scope=scope,crop_command=crop_args[1:],command=args[1:],sumo_version=subprocess.check_output([checkBinary('netconvert'),'--version'],text=True).splitlines()[0],external_edges=len(external),lanes=sum(len(e.findall('lane')) for e in external),restriction_relations_kept=kept,incomplete_restrictions_omitted=dropped,edges=audit)
    (DEST/'import-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Native lane rendering avoids the old crossing rectangles and hand-drawn stripes.
    view=ET.Element('viewsettings');scheme=ET.SubElement(view,'scheme',name='OSM source geometry - review')
    ET.SubElement(scheme,'background',backgroundColor='235,239,233')
    ET.SubElement(scheme,'edges',laneShowBorders='true',showDirection='false',showLinkDecals='true',widthExaggeration='1')
    ET.SubElement(view,'delay',value='100')
    anchors=[j for j in net.findall('junction') if j.get('id') in ('498837378','1000493460','436249310')]
    x=sum(float(j.get('x')) for j in anchors)/len(anchors)
    y=sum(float(j.get('y')) for j in anchors)/len(anchors)
    ET.SubElement(view,'viewport',x=str(x),y=str(y),zoom='150')
    ET.indent(view);ET.ElementTree(view).write(DEST/'view.xml',encoding='utf8',xml_declaration=True)
    cfg=ET.Element('configuration');inp=ET.SubElement(cfg,'input');ET.SubElement(inp,'net-file',value='network.net.xml')
    gui=ET.SubElement(cfg,'gui_only');ET.SubElement(gui,'gui-settings-file',value='view.xml')
    ET.indent(cfg);ET.ElementTree(cfg).write(DEST/'review.sumocfg',encoding='utf8',xml_declaration=True)
    print(json.dumps({k:report[k] for k in ('external_edges','lanes','source_sha256')},indent=2))
    return DEST/'review.sumocfg'
if __name__=='__main__':build()
