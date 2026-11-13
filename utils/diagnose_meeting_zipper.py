"""M3/M4 diagnostic only: matched native HV priority pairs, no AV or training."""
import collections
import concurrent.futures
import copy
import csv
import json
import shutil
import time
import xml.etree.ElementTree as E
from utils import hv_flow_audit as a
from utils.hv_flow_repair import JUNCTIONS, connection_key, validate_geometry
from utils.hv_measured_traffic import allocate

ROOT=a.ROOT
OUT=ROOT/'results/meeting_diagnosis_20261114/zipper'
SOURCE=ROOT/'scenarios/hv_high15000_entryprep'
ORIGINAL=ROOT/'maps/toegyewon_only/network.net.xml'

def prepare():
    original=E.parse(ORIGINAL)
    source=E.parse(SOURCE/'network.net.xml')
    original_links={connection_key(c):c for c in original.findall('connection')}
    restored=copy.deepcopy(source)
    provenance=[]
    for jid in JUNCTIONS:
        old=original.find(f"junction[@id='{jid}']")
        cur=restored.find(f"junction[@id='{jid}']")
        assert old.get('type')=='priority' and cur.get('type')=='zipper'
        for attr in ('incLanes','intLanes'):
            assert old.get(attr)==cur.get(attr)
        assert [r.get('index') for r in old.findall('request')]==[r.get('index') for r in cur.findall('request')]
        cur.set('type',old.get('type'))
        for req in list(cur):cur.remove(req)
        for req in old:cur.append(copy.deepcopy(req))
        incoming={e.get('id') for e in source.findall('edge') if e.get('to')==jid}
        for c in restored.findall('connection'):
            if c.get('from') in incoming:c.set('state',original_links[connection_key(c)].get('state'))
        provenance.append(dict(junction=jid,original_type=old.get('type'),incoming_edges=sorted(incoming),incoming_lanes=cur.get('incLanes')))
    OUT.mkdir(parents=True,exist_ok=True)
    restored.write(OUT/'restored-priority.net.xml',encoding='utf8',xml_declaration=True)
    geometry=validate_geometry(SOURCE/'network.net.xml',OUT/'restored-priority.net.xml')
    routes=E.parse(SOURCE/'highway.rou.xml')
    assert not routes.findall('vehicle')
    original_counts=[int(f.get('number')) for f in routes.findall('flow')]
    assert sum(original_counts)==15000
    for density,scale in [('low',.1),('base',1.)]:
        counts=allocate(round(15000*.25*scale),original_counts)
        for mode in ('on','off'):
            d=OUT/'inputs'/f'{density}_{mode}';d.mkdir(parents=True,exist_ok=True)
            shutil.copy2(SOURCE/'network.net.xml' if mode=='on' else OUT/'restored-priority.net.xml',d/'network.net.xml')
            shutil.copy2(SOURCE/'run.sumocfg',d/'run.sumocfg')
            tree=copy.deepcopy(routes)
            for f,n in zip(tree.findall('flow'),counts):f.set('number',str(n));f.set('begin','0');f.set('end','900')
            tree.write(d/'highway.rou.xml',encoding='utf8',xml_declaration=True)
    for density in ('low','base'):
        assert a.sha(OUT/'inputs'/f'{density}_on/highway.rou.xml')==a.sha(OUT/'inputs'/f'{density}_off/highway.rou.xml')
    a.save(OUT/'manifest.json',dict(seed=9101,demand_seconds=900,horizon_seconds=3600,subset_not_full_hour=True,
        density_vph=dict(low=1500,base=15000),planned=dict(low=375,base=3750),sample_period_seconds=1,
        approach_distance_m=100,stopped_threshold_mps=.1,front_wait_max_remaining_m=15,
        source_hashes={str(p):a.sha(p) for p in (SOURCE/'network.net.xml',SOURCE/'highway.rou.xml',ORIGINAL)},
        provenance=provenance,geometry=geometry,priority_provenance='hv_flow_repair.preserve_geometry_zipper copied right-of-way only from maps/toegyewon_only; OFF restores original type, request tables, connection states with lane index checks; keeps later speeds and entry extensions',
        code_sha256=a.sha(ROOT/'utils/diagnose_meeting_zipper.py')))

def run(mode):
    d=OUT/'runs'/mode;d.mkdir(parents=True,exist_ok=True)
    inp=OUT/'inputs'/mode
    net=E.parse(inp/'network.net.xml')
    approaches={j:{e.get('id'):[l.get('id') for l in e.findall('lane')] for e in net.findall('edge') if e.get('to')==j} for j in JUNCTIONS}
    rows=[];counts=collections.Counter();per=collections.defaultdict(collections.Counter)
    entrants={};previous={};signal_counts=collections.Counter();changes=[]
    a.traci.start([str(a.SUMO/'bin/sumo.exe'),'-c',str(inp/'run.sumocfg'),'--seed','9101','--no-step-log','true','--tripinfo-output',str(d/'tripinfo.xml'),'--tripinfo-output.write-unfinished','true','--error-log',str(d/'sumo.log')])
    tr=a.traci;v=tr.vehicle;start=time.time()
    try:
        for i in range(18000):
            tr.simulationStep();t=round((i+1)*.2,1)
            for vid in tr.simulation.getDepartedIDList():per[v.getRouteID(vid)]['inserted']+=1
            counts['arrived']+=tr.simulation.getArrivedNumber()
            counts['collision_vehicle_events']+=tr.simulation.getCollidingVehiclesNumber()
            counts['teleport_events']+=tr.simulation.getStartingTeleportNumber()
            if (i+1)%5==0:
                for jid,edges in approaches.items():
                    near={}
                    for edge,lanes in edges.items():
                        near[edge]=[(vid,l,tr.lane.getLength(l)-v.getLanePosition(vid)) for l in lanes for vid in tr.lane.getLastStepVehicleIDs(l) if tr.lane.getLength(l)-v.getLanePosition(vid)<=100]
                    for edge,vs in near.items():
                        opponents=sum(len(xs) for e,xs in near.items() if e!=edge)
                        for vid,l,remaining in vs:
                            speed=v.getSpeed(vid);sig=v.getSignals(vid);route=v.getRouteID(vid)
                            signal_counts[f'{jid}|{route}|{sig}']+=1
                            key=(jid,vid)
                            if previous.get(key)!=sig:changes.append(dict(t=t,junction=jid,id=vid,route=route,edge=edge,remaining=remaining,signals=sig,previous=previous.get(key)))
                            previous[key]=sig
                            if key not in entrants:entrants[key]=dict(first_seen=t,side=edge,route=route)
                            if speed<.1 and remaining<15:
                                lead=v.getLeader(vid,50)
                                rows.append(dict(t=t,junction=jid,id=vid,route=route,edge=edge,remaining=remaining,speed=speed,opponents_within_100m=opponents,waiting=v.getWaitingTime(vid),leader=lead[0] if lead else '',leader_gap=lead[1] if lead else '',signals=sig))
                # Entry order sampled at 1s; ties and missed fast traversals are not compliance labels.
                for (jid,vid),record in entrants.items():
                    if 'entry_time' not in record and vid in v.getIDList() and v.getRoadID(vid).startswith(':'+jid+'_'):record['entry_time']=t
            if (i+1)%250==0:a.save(d/'progress.json',dict(t=t,wall_seconds=time.time()-start,arrived=counts['arrived']))
            if t>=900 and tr.simulation.getMinExpectedNumber()==0:break
        remaining=len(v.getIDList());pending=len(tr.simulation.getPendingVehicles())
    finally:tr.close()
    with (d/'front-wait-samples.csv').open('w',encoding='utf8',newline='') as f:
        fields=['t','junction','id','route','edge','remaining','speed','opponents_within_100m','waiting','leader','leader_gap','signals']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    trips=E.parse(d/'tripinfo.xml').findall('tripinfo');done=[x for x in trips if float(x.get('arrival'))>=0]
    a.save(d/'signals.json',dict(sample_counts=dict(signal_counts),transitions=changes))
    a.save(d/'merge-entry-observations.json',[dict(junction=j,id=vid,**rec) for (j,vid),rec in entrants.items()])
    result=dict(mode=mode,ended=t,inserted=sum(c['inserted'] for c in per.values()),**counts,active=remaining,pending=pending,
        front_wait_sample_vehicle_seconds=len(rows),front_wait_no_opponent_sample_vehicle_seconds=sum(r['opponents_within_100m']==0 for r in rows),
        front_wait_no_opponent_no_leader_sample_vehicle_seconds=sum(r['opponents_within_100m']==0 and not r['leader'] for r in rows),
        mean_completed_wait=sum(float(x.get('waitingTime')) for x in done)/max(1,len(done)),
        mean_completed_duration=sum(float(x.get('duration')) for x in done)/max(1,len(done)),wall_seconds=time.time()-start)
    a.save(d/'complete.json',result)
    return result

if __name__=='__main__':
    prepare()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,['low_on','low_off','base_on','base_off']))
    a.save(OUT/'summary.json',results)
    print(json.dumps(results,indent=2))
