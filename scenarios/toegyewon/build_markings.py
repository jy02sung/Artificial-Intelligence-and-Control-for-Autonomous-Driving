"""Build a separate image-interpreted network, preserving the baseline files."""
import json
import copy
import xml.etree.ElementTree as ET
from sumolib.geomhelper import indexAtShapeOffset, positionAtShapeOffset, polyLength
from build import HERE, binary, write_xml

RULE_DIR = HERE / 'rules_v2'

def build():
    RULE_DIR.mkdir(exist_ok=True)
    rules=json.loads((HERE/'marking_rules.json').read_text(encoding='utf-8'))
    base=ET.parse(HERE/'toegyewon.net.xml').getroot()
    weave=base.find("edge[@id='170059994']")
    split_at=float(weave.find('lane').get('length'))-rules['weaving_solid_tail_m']
    # Split the actual lane polylines, preserving existing junction shapes.
    # netconvert's inferred junction centers are only 33m apart although this lane is 86.86m.
    root=base; first=copy.deepcopy(weave);second=copy.deepcopy(weave)
    first.set('id','170059994.dashed');first.set('to','marking_transition')
    second.set('id','170059994.solid');second.set('from','marking_transition')
    cuts=[]
    for i,(a,b) in enumerate(zip(first.findall('lane'),second.findall('lane'))):
        shape=[tuple(map(float,p.split(','))) for p in a.get('shape').split()]
        total=float(a.get('length'));offset=polyLength(shape)*split_at/total
        j,_=indexAtShapeOffset(shape,offset);point=positionAtShapeOffset(shape,offset);cuts.append(point)
        a.set('id',first.get('id')+'_'+str(i));b.set('id',second.get('id')+'_'+str(i))
        a.set('length',str(split_at));b.set('length',str(total-split_at))
        a.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in shape[:j+1]+[point]))
        b.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in [point]+shape[j+1:]))
    index=list(root).index(weave);root.remove(weave);root.insert(index,second);root.insert(index,first)
    for c in root.findall('connection'):
        if c.get('from')=='170059994':c.set('from',second.get('id'))
        if c.get('to')=='170059994':c.set('to',first.get('id'))
    for j in root.findall('junction'):
        j.set('incLanes',j.get('incLanes','').replace('170059994_','170059994.solid_'))
    x=sum(p[0] for p in cuts)/len(cuts);y=sum(p[1] for p in cuts)/len(cuts)
    junction=ET.Element('junction',id='marking_transition',type='priority',x=str(x),y=str(y),
        incLanes=' '.join(a.get('id') for a in first.findall('lane')),intLanes='',shape=f'{x-7},{y-0.01} {x+7},{y-0.01} {x+7},{y+0.01} {x-7},{y+0.01}')
    for i in range(4):ET.SubElement(junction,'request',index=str(i),response='0000',foes='0000',cont='0')
    root.insert(list(root).index(root.find('connection')),junction)
    for i in range(4):ET.SubElement(root,'connection',attrib={'from':first.get('id'),'to':second.get('id'),'fromLane':str(i),'toLane':str(i),'dir':'s','state':'M'})
    inventory={}
    for edge in root.findall('edge'):
        ident=edge.get('id');lanes=edge.findall('lane')
        if ident.startswith(':'):continue
        boundaries=[]
        if ident in rules['solid_edges'] or ident=='170059994.solid':
            boundaries=list(range(len(lanes)-1))
        elif ident in rules['solid_boundaries']:
            boundaries=rules['solid_boundaries'][ident]['boundaries']
        for idx in boundaries:
            lanes[idx].set('changeLeft','custom1 custom2')
            lanes[idx+1].set('changeRight','custom1 custom2')
        inventory[ident]={'lanes':len(lanes),'solid_boundaries':boundaries,
            'evidence_status': 'interpreted_approximate' if boundaries else 'observed_dashed' if ident in rules['dashed_observed_edges'] or ident=='170059994.dashed' else 'single_lane' if len(lanes)==1 else 'unverified_dashed'}
    # Custom classes must have the same road access as passenger vehicles, including connectors.
    for lane in root.findall('.//lane'):
        if lane.get('allow') and 'passenger' in lane.get('allow').split():
            lane.set('allow',' '.join(dict.fromkeys(lane.get('allow').split()+['custom1','custom2'])))
    write_xml(root,RULE_DIR/'network.net.xml')
    # Explicit paint overlay keeps line style legible regardless of saved GUI schemes.
    paint=ET.Element('additional')
    for edge in root.findall('edge'):
        ident=edge.get('id')
        if ident.startswith(':'):continue
        lanes=edge.findall('lane')
        for boundary in range(len(lanes)-1):
            shapes=[[tuple(map(float,p.split(','))) for p in lane.get('shape').split()] for lane in lanes[boundary:boundary+2]]
            length=min(polyLength(s) for s in shapes)
            def point(offset):
                a,b=[positionAtShapeOffset(s,min(offset,polyLength(s))) for s in shapes]
                return ((a[0]+b[0])/2,(a[1]+b[1])/2)
            def line(start,end):
                offsets=[start]+[float(j) for j in range(int(start)+1,int(end))]+[end]
                return ' '.join('%.4f,%.4f'%point(x) for x in offsets)
            ET.SubElement(paint,'poly',id=f'mask_{ident}_{boundary}',color='0,0,0',fill='false',layer='8',lineWidth='0.4',shape=line(0,length))
            solid=boundary in inventory[ident]['solid_boundaries']
            spans=[(0,length)] if solid else [(float(i),min(i+3.,length)) for i in range(0,int(length),9)]
            for k,(a,b) in enumerate(spans):
                ET.SubElement(paint,'poly',id=f'paint_{ident}_{boundary}_{k}',color='255,255,255',fill='false',layer='9',lineWidth='0.15',shape=line(a,b))
    write_xml(paint,RULE_DIR/'markings.add.xml')
    routes=ET.parse(HERE/'toegyewon.rou.xml').getroot()
    for route in routes.findall('route'):
        ids=[]
        for ident in route.get('edges').split():
            ids.extend(['170059994.dashed','170059994.solid'] if ident=='170059994' else [ident])
        route.set('edges',' '.join(ids))
    write_xml(routes,RULE_DIR/'routes.rou.xml')
    (RULE_DIR/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    cfg=ET.Element('configuration');inp=ET.SubElement(cfg,'input')
    ET.SubElement(inp,'net-file',value='network.net.xml');ET.SubElement(inp,'route-files',value='routes.rou.xml')
    ET.SubElement(inp,'additional-files',value='markings.add.xml')
    gui=ET.SubElement(cfg,'gui_only');ET.SubElement(gui,'gui-settings-file',value='../view.xml')
    write_xml(cfg,RULE_DIR/'preview.sumocfg')
    return inventory

if __name__=='__main__':build()
