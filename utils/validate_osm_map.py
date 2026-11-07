"""Static source fidelity and route connectivity checks; no driving/learning experiment."""
import json,math,os,xml.etree.ElementTree as ET
os.environ['IC_LIBSUMO']='1'
from env.sumo_env import traci
from sumolib import checkBinary
import sumolib
from utils.rebuild_osm_map import DEST

def main():
 source=ET.parse(DEST/'source.osm').getroot();ways={w.get('id'):w for w in source.findall('way')};nodes={n.get('id'):(float(n.get('lon')),float(n.get('lat'))) for n in source.findall('node')}
 root=ET.parse(DEST/'network.net.xml').getroot();junctions={j.get('id'):j for j in root.findall('junction')};edges={e.get('id'):e for e in root.findall('edge') if e.get('function')!='internal'}
 audit=json.loads((DEST/'import-audit.json').read_text(encoding='utf8'));lane_checks=0
 for row in audit['edges']:
  if row['oneway']=='yes' and row['osm_lanes'] and row['osm_lanes'].isdigit():
   assert row['lanes']==int(row['osm_lanes']),row
   lane_checks+=1
 for wid,count in [('86198728',5),('52873356',4),('37927512',3),('170059994',4)]:
  found=[e for k,e in edges.items() if k.split('#')[0]==wid];assert found and all(len(e.findall('lane'))==count for e in found)
 assert edges['169929488#0'].get('from')=='1808903166'
 assert edges['544292616'].get('to')==edges['844484877'].get('from')=='1000493465'
 assert not any(c.get('from')=='86198728' and c.get('to','').startswith('169929488') for c in root.findall('connection'))
 # Audit raw centerlines against OSM nodes. SUMO overlap avoidance can shift endpoints; report rather than conceal these deviations.
 max_error=0.;checked=0;cache={};deviations=[]
 traci.start([checkBinary('sumo'),'-c',str(DEST/'review.sumocfg'),'--no-step-log','true'])
 try:
  for eid,e in edges.items():
   wid=eid.lstrip('-').split('#')[0];w=ways[wid]
   if wid not in cache:cache[wid]=[traci.simulation.convertGeo(*nodes[n.get('ref')],fromGeo=True) for n in w.findall('nd')]
   if e.get('shape'):points=[tuple(map(float,p.split(',')[:2])) for p in e.get('shape').split()]
   else:points=[(float(junctions[e.get(side)].get('x')),float(junctions[e.get(side)].get('y'))) for side in ('from','to')]
   for p in points:
    error=min(math.hypot(p[0]-q[0],p[1]-q[1]) for q in cache[wid]);max_error=max(max_error,error)
    if error>=.05:deviations.append(dict(edge=eid,error_m=error,point=p))
   checked+=1
 finally:traci.close()
 net=sumolib.net.readNet(str(DEST/'network.net.xml'))
 pairs=[('south_to_east','86198728','1063527748'),('north_to_east','52873356','1063527748'),('north_to_west','52873356','170059992'),('outer_to_south','169929488#0','844484877'),('return_to_north','170059993','86198708'),('return_to_south','170059993','844484877')]
 paths={}
 for label,start,end in pairs:
  path,cost=net.getShortestPath(net.getEdge(start),net.getEdge(end),vClass='passenger')
  assert path,label
  paths[label]=dict(edges=[e.getID() for e in path],length_m=cost)
 report=dict(status='review_required' if deviations else 'passed',geometry_deviations=deviations,no_simulation_steps_or_training=True,source_geometry_edges_checked=checked,max_projected_source_error_m=max_error,explicit_oneway_lane_tag_checks=lane_checks,roads_with_unverified_lane_defaults=sum(r['lane_source'].startswith('SUMO') for r in audit['edges']),routes=paths,limitations=['Lane-to-lane mapping and signals inferred by SUMO where OSM lacks detail','No measured road elevations; layer topology preserved without physical Z inference','Conversion warnings retained; no traffic-capacity or AV certification'])
 (DEST/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
