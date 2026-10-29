"""Run the IC baseline, record real TraCI telemetry and validate traffic.

This does not load a trained policy. Telemetry is diagnostic ground truth,
not the original POMDP dataset or a PPO evaluation result.
"""
import argparse
import gzip
import json
from pathlib import Path
import traci
import traci.constants as tc
from build import HERE, binary

def run(seed=0, gui=False, record=False):
    params=json.loads((HERE/'parameters.json').read_text(encoding='utf-8'))
    out=HERE/'results'/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    args=[binary('sumo-gui' if gui else 'sumo'),'-c',str(HERE/'toegyewon.sumocfg'),
          '--seed',str(seed),'--no-step-log','--duration-log.disable','true',
          '--tripinfo-output',str(out/'tripinfo.xml'),
          '--lanechange-output',str(out/'lanechanges.xml'),
          '--error-log',str(out/'warnings.log')]
    if gui: args+=['--start','--quit-on-end','--delay','0','--window-size','1400,1000']
    traci.start(args)
    stream=gzip.open(out/'telemetry.jsonl.gz','wt',encoding='utf-8') if record else None
    report={'seed':seed,'controller':'SUMO IDM ego (not trained RL)',
            'departed':0,'arrived':0,'collision_events':0,'teleports':0,
            'max_active_ego':0,'ego_arrived':False,'ego_weaving_lanes':[],
            'weaving_lane_changes':0,'duration_s':0}
    previous={}
    variables=[tc.VAR_ROAD_ID,tc.VAR_LANE_INDEX]
    if record: variables += [tc.VAR_TYPE,tc.VAR_ROUTE_ID,tc.VAR_LANEPOSITION,tc.VAR_POSITION,tc.VAR_SPEED,tc.VAR_ACCELERATION,tc.VAR_ANGLE]
    if gui:
        traci.gui.setBoundary('View #0',-30,-30,940,1250)
    try:
        while traci.simulation.getTime()<params['duration_s']+300:
            traci.simulationStep()
            t=traci.simulation.getTime()
            ids=traci.vehicle.getIDList()
            for ident in traci.simulation.getDepartedIDList():
                if ident in ids: traci.vehicle.subscribe(ident,variables)
            states=traci.vehicle.getAllSubscriptionResults()
            report['departed']+=traci.simulation.getDepartedNumber()
            report['arrived']+=traci.simulation.getArrivedNumber()
            report['collision_events']+=traci.simulation.getCollidingVehiclesNumber()
            report['teleports']+=traci.simulation.getStartingTeleportNumber()
            report['max_active_ego']=max(report['max_active_ego'],int('ego' in ids))
            report['ego_arrived']|='ego' in traci.simulation.getArrivedIDList()
            rows=[];current={}
            for ident in ids:
                state=states[ident]
                edge=state[tc.VAR_ROAD_ID]
                lane=state[tc.VAR_LANE_INDEX]
                current[ident]=(edge,lane)
                if edge=='170059994':
                    if ident=='ego' and (not report['ego_weaving_lanes'] or report['ego_weaving_lanes'][-1]!=lane):
                        report['ego_weaving_lanes'].append(lane)
                    old=previous.get(ident)
                    if old and old[0]==edge and old[1]!=lane:report['weaving_lane_changes']+=1
                if stream:
                    x,y=state[tc.VAR_POSITION]
                    rows.append({'id':ident,'type':state[tc.VAR_TYPE],
                        'route':state[tc.VAR_ROUTE_ID],'edge':edge,'lane':lane,
                        'lane_position_m':state[tc.VAR_LANEPOSITION],
                        'x_m':x,'y_m':y,'speed_mps':state[tc.VAR_SPEED],
                        'accel_mps2':state[tc.VAR_ACCELERATION],
                        'angle_deg':state[tc.VAR_ANGLE]})
            if stream:stream.write(json.dumps({'time_s':t,'vehicles':rows},ensure_ascii=False)+'\n')
            previous=current
            if t>=params['duration_s'] and traci.simulation.getMinExpectedNumber()==0:break
        report['duration_s']=traci.simulation.getTime()
        report['remaining']=traci.simulation.getMinExpectedNumber()
    finally:
        if stream:stream.close()
        traci.close(wait=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['collision_events'] or report['teleports'] or report['remaining'] or not report['ego_arrived']:
        raise RuntimeError(f'Simulation acceptance failed; inspect {out}')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--record',action='store_true')
    a=p.parse_args()
    run(a.seed,a.gui,a.record)
