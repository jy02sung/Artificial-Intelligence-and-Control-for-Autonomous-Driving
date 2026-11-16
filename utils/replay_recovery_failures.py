"""Replay validation failures with collision parties and nearby physical states."""
import collections
from utils import run_traffic_recovery_validation as v
s=v.s;t=v.t
OUT=v.ROOT/'results/traffic_recovery_failure_replay_20261116'

def one(seed,route):
    s.a.b.torch.set_num_threads(1)
    m=t.read(v.MODEL.parent/'manifest.json');agent=s.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(v.MODEL))
    env=s.make_env('route',evaluation=True,none=False);tr=s.a.b.traci;veh=tr.vehicle
    tail=collections.deque(maxlen=150)
    try:
        obs,_=env.reset(seed=seed,options={'route':route})
        while True:
            ego=env.ego_id;speed=veh.getSpeed(ego);lead=veh.getLeader(ego,200)
            front=None
            if lead and lead[0]:
                vid,gap=lead;front=dict(id=vid,gap=gap,speed=veh.getSpeed(vid),acceleration=veh.getAcceleration(vid),edge=veh.getRoadID(vid),lane=veh.getLaneIndex(vid))
            action=agent.predict(obs,deterministic=True)
            row=dict(t=env.step_count*.2,edge=veh.getRoadID(ego),lane=veh.getLaneIndex(ego),speed=speed,position=veh.getLanePosition(ego),front=front,action=action.tolist(),obs=obs.tolist(),safety=env.safety_context(action),neighbors={str((o,f)):env._neighbor(o,f) for o in (-1,0,1) for f in (True,False)})
            obs,_,done,cut,info=env.step(action)
            row['lane_applied']=info.get('lane_change_applied');tail.append(row)
            if done or cut:
                collisions=[{k:getattr(c,k,None) for k in ('collider','victim','colliderSpeed','victimSpeed','type','lane','pos')} for c in tr.simulation.getCollisions()]
                result=dict(seed=seed,route=route,seconds=env.step_count*.2,outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout',tail=list(tail),collisions=collisions,overrides=env.route_override_count)
                t.save(OUT/f'{seed}-{route}.json',result);print({k:result[k] for k in ('seed','route','seconds','outcome','collisions')},flush=True);return
    finally:env.close()

if __name__=='__main__':
    for spec in ((69101,'route_01'),(69103,'route_01'),(69103,'route_13')):one(*spec)
