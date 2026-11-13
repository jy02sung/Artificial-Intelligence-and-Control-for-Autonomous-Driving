"""HV-only matched demand audit; preserve original and isolate exit split."""
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hv_flow_audit_20261113'
SOURCE=ROOT/'results/resume_validation_20261112/traffic_v2/base-hv'
SUMO=Path(os.environ.get('SUMO_HOME',r'C:/Program Files (x86)/Eclipse/Sumo'))
sys.path.append(str(SUMO/'tools'))
import traci


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(mode):
    d=OUT/'inputs'/mode;d.mkdir(parents=True,exist_ok=True)
    for name in ['network.net.xml','highway.rou.xml','run.sumocfg']:shutil.copy2(SOURCE/name,d/name)
    tree=ET.parse(d/'highway.rou.xml')
    if mode=='lane_split':
        rates={'route_13':200,'route_14':200,'route_15':200,'route_16':2400}
        for flow in tree.findall('flow'):
            if flow.get('route') in rates:flow.set('vehsPerHour',str(rates[flow.get('route')]))
        tree.write(d/'highway.rou.xml',encoding='utf8',xml_declaration=True)
    assert not tree.findall('vehicle'), 'HV-only requires no explicit ego/demo vehicles'
    assert sum(float(x.get('vehsPerHour')) for x in tree.findall('flow'))==10200
    save(d/'manifest.json',dict(mode=mode,total_vph=10200,source=str(SOURCE),
        changes='13/14/15:750->200 each;16:750->2400; total unchanged' if mode=='lane_split' else 'none',
        files={n:sha(d/n) for n in ['network.net.xml','highway.rou.xml','run.sumocfg']},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))


def run(spec):
    mode,seed=spec;d=OUT/'runs'/f'{mode}-{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return str(d)
    inp=OUT/'inputs'/mode
    flows=ET.parse(inp/'highway.rou.xml').findall('flow')
    planned={x.get('route'):math.ceil(float(x.get('vehsPerHour'))*float(x.get('end'))/3600-1e-9) for x in flows}
    net=ET.parse(inp/'network.net.xml')
    links=collections.defaultdict(set)
    for c in net.findall('connection'):links[c.get('from'),c.get('to')].add(int(c.get('fromLane')))
    perroute={k:dict(planned=v,inserted=0,arrived=0) for k,v in planned.items()}
    vehicle_route={};collisions=teleports=0;stops=collections.Counter();evidence={};snapshots=[];checkpoints={}
    traci.start([str(SUMO/'bin/sumo.exe'),'-c',str(inp/'run.sumocfg'),'--seed',str(seed),
        '--step-length','.2','--no-step-log','true','--collision.action','remove',
        '--tripinfo-output',str(d/'tripinfo.xml'),'--tripinfo-output.write-unfinished','true',
        '--error-log',str(d/'sumo.log')])
    started=time.time()
    try:
        for i in range(18000):
            traci.simulationStep();t=round((i+1)*.2,1)
            for vid in traci.simulation.getDepartedIDList():
                route=traci.vehicle.getRouteID(vid);vehicle_route[vid]=route;perroute[route]['inserted']+=1
            for vid in traci.simulation.getArrivedIDList():perroute[vehicle_route[vid]]['arrived']+=1
            collisions+=traci.simulation.getCollidingVehiclesNumber();teleports+=traci.simulation.getStartingTeleportNumber()
            if (i+1)%25==0:
                ids=traci.vehicle.getIDList();halted=[]
                for vid in ids:
                    v=traci.vehicle
                    if v.getSpeed(vid)>=.5:continue
                    edge=v.getRoadID(vid);lane=v.getLaneID(vid);path=v.getRoute(vid);idx=v.getRouteIndex(vid)
                    nxt=path[idx+1] if idx+1<len(path) else None
                    allowed=links.get((edge,nxt),set());remaining=traci.lane.getLength(lane)-v.getLanePosition(vid)
                    lead=v.getLeader(vid,50)
                    reason=('wrong_lane_end' if allowed and v.getLaneIndex(vid) not in allowed and remaining<10 else
                            'queue_behind_vehicle' if lead and v.getSpeed(lead[0])<2 else
                            'junction_front_wait' if remaining<15 else 'other_slow_or_stopped')
                    key=(edge,reason);stops[key]+=5
                    ev=dict(t=t,id=vid,route=vehicle_route[vid],lane=lane,remaining=remaining,
                            waiting=v.getWaitingTime(vid),next_edge=nxt,allowed=sorted(allowed),
                            leader=lead,leader_speed=v.getSpeed(lead[0]) if lead else None)
                    if key not in evidence or ev['waiting']>evidence[key]['waiting']:evidence[key]=ev
                    halted.append(vid)
                pending=list(traci.simulation.getPendingVehicles())
                snapshots.append(dict(t=t,active=len(ids),stopped=len(halted),pending=len(pending),
                    pending_routes=dict(collections.Counter('.'.join(v.split('.')[:-1]) for v in pending))))
            if t in [900,1800,2700,3600]:
                checkpoints[str(int(t))]=dict(inserted=sum(v['inserted'] for v in perroute.values()),
                    arrived=sum(v['arrived'] for v in perroute.values()),active=len(traci.vehicle.getIDList()),
                    pending=len(traci.simulation.getPendingVehicles()),collisions=collisions,teleports=teleports)
            if (i+1)%250==0:
                save(d/'progress.json',dict(t=t,pid=os.getpid(),wall_seconds=time.time()-started,
                    inserted=sum(v['inserted'] for v in perroute.values()),arrived=sum(v['arrived'] for v in perroute.values())))
            if t>=900 and traci.simulation.getMinExpectedNumber()==0:break
        leftovers=[dict(id=vid,route=vehicle_route[vid],lane=traci.vehicle.getLaneID(vid),
            speed=traci.vehicle.getSpeed(vid),waiting=traci.vehicle.getWaitingTime(vid)) for vid in traci.vehicle.getIDList()]
        pending=list(traci.simulation.getPendingVehicles())
    finally:traci.close()
    trips=ET.parse(d/'tripinfo.xml').findall('tripinfo')
    stats={}
    for route in perroute:
        ts=[x for x in trips if vehicle_route.get(x.get('id'))==route and float(x.get('arrival'))>=0]
        stats[route]=dict(perroute[route],mean_depart_delay=sum(float(x.get('departDelay')) for x in ts)/max(1,len(ts)),
             max_depart_delay=max([float(x.get('departDelay')) for x in ts] or [0]),
             mean_wait=sum(float(x.get('waitingTime')) for x in ts)/max(1,len(ts)),
             mean_duration=sum(float(x.get('duration')) for x in ts)/max(1,len(ts)))
    passed=all(x['planned']==x['inserted']==x['arrived'] for x in stats.values()) and not(collisions or teleports or leftovers or pending)
    save(d/'samples.json',snapshots)
    save(d/'complete.json',dict(mode=mode,seed=seed,ended=t,checkpoints=checkpoints,routes=stats,
        collisions=collisions,teleports=teleports,leftovers=leftovers,pending=pending,passed=passed,
        stop_locations=[dict(edge=k[0],reason=k[1],sampled_vehicle_seconds=n,evidence=evidence[k]) for k,n in stops.most_common()],
        wall_seconds=time.time()-started))
    return str(d)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for mode in ['original','lane_split']:prepare(mode)
        save(OUT/'status.json',dict(stage='running',pid=os.getpid()))
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
            for result in pool.map(run,[(m,s) for m in ['original','lane_split'] for s in [1001,1002,1003]]):print(result,flush=True)
        save(OUT/'status.json',dict(stage='complete'))
    except Exception as exc:save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
