"""Collect baseline-controller transitions; never trains or loads an AI model."""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import sumolib
import traci
import traci.constants as tc
from build import HERE, binary, write_xml

ROUTES=('nw_to_east','south_to_west')
FIELDS={tc.VAR_ROAD_ID:'edge',tc.VAR_LANE_INDEX:'lane',tc.VAR_SPEED:'speed',
        tc.VAR_ANGLE:'angle',tc.VAR_LANEPOSITION:'position',tc.VAR_TYPE:'type',
        tc.VAR_ROUTE_ID:'route',tc.VAR_ACCELERATION:'acceleration'}

def snapshot():
    ids=set(traci.vehicle.getIDList())
    for ident in traci.simulation.getDepartedIDList():
        if ident in ids:traci.vehicle.subscribe(ident,list(FIELDS)+[tc.VAR_POSITION])
    states={}
    for ident,v in traci.vehicle.getAllSubscriptionResults().items():
        if ident not in ids:continue
        states[ident]={name:v[key] for key,name in FIELDS.items()}
        states[ident]['x'],states[ident]['y']=v[tc.VAR_POSITION]
    return states

def observation(states,route,lengths,lanes):
    if 'ego' not in states:return np.zeros(54,np.float32),np.empty((0,2),np.float32)
    ego=states['ego'];edge=ego['edge'];lane=ego['lane']
    remaining=max(0,lengths.get(f'{edge}_{lane}',ego['position'])-ego['position'])
    vec=[ego['speed']/33.33,math.sin(math.radians(ego['angle'])),
         lane/max(lanes.get(edge,1)-1,1),min(remaining/200,1),
         float(route==ROUTES[0]),float(route==ROUTES[1])]
    nearby=[]
    for ident,v in states.items():
        if ident=='ego':continue
        dx,dy=v['x']-ego['x'],v['y']-ego['y']
        distance=math.hypot(dx,dy)
        if distance<=50:nearby.append((distance,ident,v,dx,dy))
    nearby.sort(key=lambda row:(row[0],row[1]))
    for _,_,v,dx,dy in nearby[:8]:
        da=math.radians(v['angle']-ego['angle'])
        vec += [dx/50,dy/50,(v['speed']-ego['speed'])/33.33,math.sin(da),math.cos(da),1.]
    vec += [0.]*(54-len(vec))
    xy=np.asarray([[v['x'],v['y']] for _,_,v,_,_ in nearby],np.float32).reshape(-1,2)
    return np.asarray(vec,np.float32),xy

def applied_action(before,after,dt):
    if after is None:return [0.,0.],[False,False]
    same=before['edge']==after['edge'] and not before['edge'].startswith(':')
    return [(after['speed']-before['speed'])/dt,
            after['lane']-before['lane'] if same else 0],[True,same]

def collect(out,repeats=2,seed=100,rules=False,ego_controller=None,paired=False):
    if paired and (not rules or repeats!=5 or ego_controller is None):
        raise ValueError('Paired study requires rules, 5 repeats, and an explicit ego controller')
    out.mkdir(parents=True,exist_ok=False)
    net_source=HERE/'rules_v2/network.net.xml' if rules else HERE/'toegyewon.net.xml'
    net=sumolib.net.readNet(str(net_source),withInternal=True)
    lengths={lane.getID():lane.getLength() for e in net.getEdges(withInternal=True) for lane in e.getLanes()}
    lanes={e.getID():e.getLaneNumber() for e in net.getEdges(withInternal=True)}
    params=json.loads((HERE/'parameters.json').read_text(encoding='utf-8'))
    config=json.loads((HERE/'collection_config.json').read_text(encoding='utf-8'))
    dt=params['step_length_s']
    source=HERE/'rules_v2/routes.rou.xml' if rules else HERE/'toegyewon.rou.xml'
    for name in ['parameters.json','collection_config.json','toegyewon.net.xml','toegyewon.rou.xml','collect.py']:
        shutil.copy2(HERE/name,out/name)
    if rules:
        shutil.copy2(net_source,out/'toegyewon.net.xml');shutil.copy2(source,out/'toegyewon.rou.xml')
        for name in ['marking_rules.json','rule_policy.py','build_markings.py']:
            shutil.copy2(HERE/name,out/name)
        shutil.copy2(HERE/'rules_v2/inventory.json',out/'marking_inventory.json')
    manifest={'schema':'toegyewon-baseline-v1','controller':'SUMO IDM + LC2013, no learned policy',
        'step_length_s':dt,'visibility_m':50,'observation_dim':54,
        'action_semantics':'realized [delta_speed/dt m/s2, same-edge lane index delta]; NOT command/action_raw',
        'reward':'0.01*speed_t*dt - 0.001*dt - 5*ego_collision + 2*arrival',
        'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['toegyewon.net.xml','toegyewon.rou.xml','parameters.json']},
        'collection_config':config,'sumo_version':subprocess_version(),
        'episodes':[],'complete':False}
    if ego_controller:
        manifest['ego_controller']=ego_controller
        manifest['controller']=f'SUMO {ego_controller} + LC2013, no learned policy'
    if paired:
        shutil.copy2(HERE/'comparison_setup.py',out/'comparison_setup.py')
        manifest.update(experiment='matched-controllers-v1',matched_seed_base=seed,
                        demand_protocol='explicit deterministic vehicle types/departures; all speed factors 1; endogenous trajectories may differ',
                        split_protocol='per route/load: repeats 0/1/2=train,3=validation,4=test')
    if rules:
        manifest.update(schema='toegyewon-rules-v2',controller='SUMO native car following + marked lane permissions + brake-first emergency supervisor',
                        rule_context_fields=['left_exists','right_exists','left_solid','right_solid'],
                        marking_rules=json.loads((HERE/'marking_rules.json').read_text(encoding='utf-8')))
        manifest['source_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['toegyewon.net.xml','toegyewon.rou.xml','parameters.json','marking_rules.json','rule_policy.py','marking_inventory.json']}
    arrays={k:[] for k in ['obs','next_obs','action_applied','action_valid','reward','terminated','truncated','episode','t','next_obs_valid']}
    if rules:arrays.update({k:[] for k in ['rule_context','next_rule_context','emergency_requested','emergency_braking']})
    detected=[];offsets=[0]
    schedule=[(scale,route,rep) for scale in config['traffic_scales'] for route in ROUTES for rep in range(repeats)]
    with gzip.open(out/'transitions.jsonl.gz','wt',encoding='utf-8') as stream:
        for episode,(scale,route,rep) in enumerate(schedule):
            epseed=seed+episode;root=ET.parse(source).getroot()
            for ident,model in [('car_idm','IDM'),('car_eidm','EIDM'),('car_acc','ACC')]:
                root.insert(2,ET.Element('vType',id=ident,carFollowModel=model,
                    accel='2.0',decel='4.5',tau='1.5',minGap='2.5',length='5',color='0.2,0.7,0.7'))
            distribution=root.find("vTypeDistribution[@id='background']")
            distribution.set('vTypes',' '.join(config['background_distribution']))
            distribution.set('probabilities',' '.join(str(v) for v in config['background_distribution'].values()))
            for flow in root.findall('flow'):flow.set('vehsPerHour',str(float(flow.get('vehsPerHour'))*scale))
            ego=root.find("vehicle[@id='ego']");ego.set('route',route);ego.set('depart',str(config['ego_depart_s']))
            if ego_controller:
                from comparison_setup import configure_ego
                manifest['ego_parameters']=configure_ego(root,ego_controller)
            demand_digest=None
            if paired:
                from comparison_setup import freeze_demand,demand_hash,split_for
                freeze_demand(root,epseed);demand_digest=demand_hash(root)
            routefile=out/f'episode_{episode:02d}.rou.xml';write_xml(root,routefile)
            traci.start([binary('sumo'),'-n',str(out/'toegyewon.net.xml'),'-r',str(routefile),
                '--step-length',str(dt),'--seed',str(epseed),'--no-step-log','--duration-log.disable','true',
                '--collision.check-junctions','true','--time-to-teleport','120',
                '--lanechange-output',str(out/f'episode_{episode:02d}.lanechanges.xml'),
                '--error-log',str(out/f'episode_{episode:02d}.warnings.log')])
            steps=0;total_reward=0.;reason='spawn_timeout';collisions=0;teleports=0
            policy=None
            if rules:
                from rule_policy import RulePolicy
                policy=RulePolicy(epseed,out/f'episode_{episode:02d}.rules.jsonl')
            try:
                states={}
                for _ in range(int(180/dt)):
                    decision=policy.before_step(states) if policy else None
                    traci.simulationStep();nextstates=snapshot()
                    if policy:policy.after_step(states,nextstates,decision)
                    states=nextstates
                    if 'ego' in states:break
                if 'ego' not in states:raise RuntimeError('ego failed to depart within 180 seconds')
                ego_start_time=traci.simulation.getTime()
                observed_ego_parameters={k:f('ego') for k,f in dict(accel=traci.vehicle.getAccel,decel=traci.vehicle.getDecel,
                    emergencyDecel=traci.vehicle.getEmergencyDecel,tau=traci.vehicle.getTau,minGap=traci.vehicle.getMinGap,
                    length=traci.vehicle.getLength,width=traci.vehicle.getWidth,speedFactor=traci.vehicle.getSpeedFactor).items()}
                obs,xy=observation(states,route,lengths,lanes)
                for step in range(int(config['episode_limit_s']/dt)):
                    decision=policy.before_step(states) if policy else None
                    time=traci.simulation.getTime();traci.simulationStep();nxt=snapshot()
                    if policy:policy.after_step(states,nxt,decision)
                    collided='ego' in traci.simulation.getCollidingVehiclesIDList()
                    teleported='ego' in traci.simulation.getStartingTeleportIDList()
                    arrived='ego' in traci.simulation.getArrivedIDList()
                    collisions+=int(collided);teleports+=int(teleported)
                    term=arrived or collided
                    missing='ego' not in nxt
                    trunc=(teleported or (missing and not term) or step+1==int(config['episode_limit_s']/dt)) and not term
                    nextobs,nextxy=observation(nxt,route,lengths,lanes)
                    action,valid=applied_action(states['ego'],nxt.get('ego'),dt)
                    reward=.01*states['ego']['speed']*dt-.001*dt-5*collided+2*arrived
                    reason='collision' if collided else 'arrival' if arrived else 'teleport' if teleported else 'missing' if missing else 'time_limit'
                    record=dict(episode=episode,t=step,time_s=time,route=route,traffic_scale=scale,
                        observation=obs.tolist(),next_observation=nextobs.tolist(),action_applied=action,
                        action_valid=valid,reward=reward,terminated=term,truncated=trunc,
                        next_obs_valid=not missing,state=states,next_state=nxt,
                        detected_xy=xy.tolist(),ego_collision=collided,ego_teleport=teleported)
                    if policy:
                        record.update(rule_context=policy.context(states),next_rule_context=policy.context(nxt),
                            ego_rule_decision=decision,violator_ids=sorted(policy.violators.intersection(states)))
                        for k,v in dict(rule_context=record['rule_context'],next_rule_context=record['next_rule_context'],
                                        emergency_requested=decision['emergency'],emergency_braking=decision['brake']).items():arrays[k].append(v)
                    stream.write(json.dumps(record,separators=(',',':'))+'\n')
                    for key,value in dict(obs=obs,next_obs=nextobs,action_applied=action,action_valid=valid,
                        reward=reward,terminated=term,truncated=trunc,episode=episode,t=step,next_obs_valid=not missing).items():arrays[key].append(value)
                    detected.extend(xy.tolist());offsets.append(len(detected))
                    steps+=1;total_reward+=reward
                    if term or trunc:break
                    states,obs,xy=nxt,nextobs,nextxy
            finally:
                if policy:policy.close()
                traci.close(wait=False)
            manifest['episodes'].append(dict(episode=episode,seed=epseed,route=route,traffic_scale=scale,
                nominal_vehs_per_hour=sum(params['traffic_vehs_per_hour'].values())*scale,steps=steps,return_=total_reward,end_reason=reason,
                ego_collisions=collisions,ego_teleports=teleports,ego_start_time_s=ego_start_time,
                observed_ego_parameters=observed_ego_parameters))
            if paired:
                manifest['episodes'][-1].update(pair_id=episode,repeat=rep,split=split_for(rep),demand_sha256=demand_digest)
            if policy:
                manifest['episodes'][-1]['rule_counts']=policy.counts
                assert policy.counts['ordinary_solid_crossings']==0,policy.counts
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'episode {episode+1}/{len(schedule)} route={route} traffic={scale} transitions={steps} end={reason}',flush=True)
    for key,value in arrays.items():
        dtype=np.bool_ if key in ['action_valid','terminated','truncated','next_obs_valid'] else np.int32 if key in ['episode','t'] else np.float32
        arrays[key]=np.asarray(value,dtype=dtype)
    arrays['detected_xy']=np.asarray(detected,np.float32).reshape(-1,2)
    arrays['detected_offsets']=np.asarray(offsets,np.int64)
    np.savez_compressed(out/'dataset.npz',**arrays)
    manifest.update(complete=True,transitions=len(arrays['reward']))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    validate(out)
    from summarize_collection import summarize
    summarize(out)
    return manifest

def subprocess_version():
    import subprocess
    return subprocess.check_output([binary('sumo'),'--version'],text=True).splitlines()[0]

def validate(out):
    with np.load(out/'dataset.npz',allow_pickle=False) as archive:
        d={key:archive[key] for key in archive.files}
    n=len(d['reward']);assert n>0 and d['obs'].shape==(n,54)
    for k in d:assert np.isfinite(d[k]).all(),k
    assert len(d['detected_offsets'])==n+1
    assert d['detected_offsets'][-1]==len(d['detected_xy'])
    assert np.all(np.diff(d['detected_offsets'])>=0)
    assert not np.any(d['terminated'] & d['truncated'])
    for ep in np.unique(d['episode']):
        ix=np.flatnonzero(d['episode']==ep)
        assert np.array_equal(d['t'][ix],np.arange(len(ix)))
        assert not np.any((d['terminated']|d['truncated'])[ix[:-1]])
        assert (d['terminated']|d['truncated'])[ix[-1]]
        np.testing.assert_allclose(d['next_obs'][ix[:-1]],d['obs'][ix[1:]])
        if 'rule_context' in d:
            np.testing.assert_array_equal(d['next_rule_context'][ix[:-1]],d['rule_context'][ix[1:]])
    with gzip.open(out/'transitions.jsonl.gz','rt',encoding='utf-8') as f:
        count=0
        for count,line in enumerate(f,1):
            r=json.loads(line);i=count-1
            np.testing.assert_allclose(r['observation'],d['obs'][i])
            np.testing.assert_allclose(r['next_observation'],d['next_obs'][i])
            np.testing.assert_allclose(r['action_applied'],d['action_applied'][i],rtol=1e-5,atol=1e-6)
            assert r['episode']==int(d['episode'][i])
            assert r['terminated']==bool(d['terminated'][i]) and r['truncated']==bool(d['truncated'][i])
            assert r['action_valid']==d['action_valid'][i].tolist()
            assert math.isclose(r['reward'],float(d['reward'][i]),rel_tol=1e-5,abs_tol=1e-6)
            if 'rule_context' in d:
                np.testing.assert_array_equal(r['rule_context'],d['rule_context'][i])
                np.testing.assert_array_equal(r['next_rule_context'],d['next_rule_context'][i])
                assert bool(d['emergency_requested'][i])==r['ego_rule_decision']['emergency']
                assert bool(d['emergency_braking'][i])==r['ego_rule_decision']['brake']
        assert count==n
    print(f'VALIDATED {n} transitions in {out}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--validate',type=Path)
    parser.add_argument('--rules',action='store_true',help='Use the separately built v2 markings and exception policy')
    parser.add_argument('--ego-controller',choices=['Krauss','IDM','EIDM','ACC'])
    parser.add_argument('--paired',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.validate:validate(args.validate)
    else:
        if args.repeats<1:parser.error('--repeats must be positive')
        # Fixed UTC+9 also works on Windows without the optional tzdata package.
        from datetime import timezone,timedelta
        name=datetime.now(timezone(timedelta(hours=9))).strftime(('rules_v2_' if args.rules else 'dataset_')+'%Y%m%d_%H%M%S')
        collect(args.out or HERE/'results'/name,args.repeats,args.seed,args.rules,args.ego_controller,args.paired)
