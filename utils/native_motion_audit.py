"""Paired same-start motion traces for junctions with observed stalls."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from utils.completion_focus_v1 import BASE, agent_for, make_c, torch, traci

OUT = BASE.parents[2] / 'native_motion_audit'
EDGES = {5:'170059994', 7:'169929488#0', 13:'170059994', 14:'170059994', 16:'86198728'}

def run(job):
    number, lane = job
    route = f'route_{number:02d}'
    torch.set_num_threads(1)
    agent = agent_for(33)
    agent.load(str(BASE))
    agent.policy.eval()
    env = make_c('high_real_ratio','C0',33,OUT/'maps'/str(os.getpid()))
    rows = []
    try:
        for mode in ('sumo','ppo'):
            env.ego_depart_lane = str(min(lane,env.counts[env.routes[route][0]]-1))
            obs,_ = env.reset(seed=992100+lane,options={'route':route})
            ego = env.ego_id
            row = dict(route=number, start_lane=lane, mode=mode, start=[*env.snapshot[:2],traci.vehicle.getLanePosition(ego),env.snapshot[7],traci.simulation.getTime()],events=[])
            if mode=='sumo':
                traci.vehicle.setSpeedMode(ego,31)
                traci.vehicle.setLaneChangeMode(ego,1621)
                traci.vehicle.setSpeed(ego,-1)
            for t in range(3000):
                edge,l,path,i,j,d,sides,v,rem = env.snapshot
                action = agent.predict(obs,deterministic=True)
                cmd = env._quantize_lane_change(action[env.lane_change_index])
                if edge==EDGES[number]:
                    targets=sorted(env.links[edge][path[i+1]])
                    if not row['events']:
                        row.update(targets=targets, solid=edge in env.solid)
                    if t%5==0 or not row['events'] or row['events'][-1]['lane']!=l:
                        row['events'].append(dict(step=t,lane=l,m=round(rem,1),v=round(v,2),ppo_cmd=cmd,accel=float(action[env.accel_index]),gaps=sides))
                if mode=='sumo':
                    traci.simulationStep()
                    done=ego in traci.simulation.getCollidingVehiclesIDList() or ego in traci.simulation.getStartingTeleportIDList() or ego in traci.simulation.getArrivedIDList()
                    if done:
                        row['outcome']='arrival' if ego in traci.simulation.getArrivedIDList() else 'collision_or_teleport'
                        break
                    obs=env._get_obs()
                else:
                    obs,_,done,cut,info=env.step(action)
                    if done or cut:
                        row['outcome']='arrival' if info['arrived'] else 'collision' if info['collided'] else 'timeout'
                        break
            rows.append(row)
        assert rows[0]['start']==rows[1]['start']
        return rows
    finally:
        env.close()

if __name__=='__main__':
    with ProcessPoolExecutor(5) as pool:
        rows=[r for pair in pool.map(run, [(n,l) for n in EDGES for l in (0,1)]) for r in pair]
    (OUT/'paired.json').write_text(json.dumps(rows),encoding='utf-8')
