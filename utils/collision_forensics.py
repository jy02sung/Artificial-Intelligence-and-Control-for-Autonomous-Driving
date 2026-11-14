"""Read-only collision replay with per-step observation/action/physical evidence."""
import collections,concurrent.futures as cf,argparse,json
from utils import run_route_repeat_transfer as r
OUT=r.t.ROOT/'results/collision_forensics_20261114'

def one(spec):
    seed,c,route=spec;dest=OUT/f'{seed}-{c}-{route}.json'
    if dest.exists():return r.t.read(dest)
    t=r.t;t.OUT=r.OUT/f'seed{seed}';t.SEED=seed;t.EVAL_SEEDS=r.EVAL_SEEDS;t.a.b.torch.set_num_threads(1)
    source=(r.x.OUT if seed==11 else t.OUT)/'runs'/c
    m=t.read(source/'manifest.json');model=source/'policy-200000.pt'
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(model))
    env=r.make_env(c,evaluation=True,none=False);v=t.a.b.traci.vehicle;tr=t.a.b.traci
    tail=collections.deque(maxlen=150);first=[];stats=collections.Counter();riskfirst=None
    try:
        obs,_=env.reset(seed=29101,options={'route':route})
        for k in range(54000):
            speed=v.getSpeed(env.ego_id);leader=v.getLeader(env.ego_id,env.visibility)
            lead=None
            if leader and leader[0]:
                vid,gap=leader;lv=v.getSpeed(vid);physical=gap+v.getMinGap(env.ego_id);closing=speed-lv
                lead=dict(id=vid,gap=gap,physical_gap=physical,speed=lv,edge=v.getRoadID(vid),lane=v.getLaneIndex(vid),acceleration=v.getAcceleration(vid),ttc=physical/closing if closing>0 else None)
            action=agent.predict(obs,deterministic=True)
            accel=float(action[0])*(env.max_accel if action[0]>=0 else env.max_decel)
            target=max(0,min(env.vmax,speed+accel*env.step_length))
            i=1+4*list(env.obs_offsets).index(0)
            expected=[1,1] if lead is None or lead["gap"]>env.visibility else [max(0,lead['gap'])/env.visibility,max(-1,min(1,(lead['speed']-speed)/env.vmax))]
            mismatch=max(abs(float(obs[i+j])-expected[j]) for j in (0,1))
            row=dict(t=k*.2,edge=v.getRoadID(env.ego_id),lane=v.getLaneIndex(env.ego_id),position=v.getLanePosition(env.ego_id),speed=speed,leader=lead,front_obs=obs[i:i+2].tolist(),front_expected=expected,obs_error=mismatch,action=action.tolist(),accel_command=accel,target_speed=target,limit=v.getAllowedSpeed(env.ego_id),speedmode=v.getSpeedMode(env.ego_id),lc_mode=v.getLaneChangeMode(env.ego_id),obs=obs.tolist())
            stats['steps']+=1;stats['observation_mismatch']+=int(mismatch>1e-5)
            if lead:
                stats['front_present']+=1
                if lead['ttc'] is not None and 0<lead['ttc']<3:
                    stats['ttc_under3_steps']+=1;stats['ttc_under3_braking']+=int(action[0]<0)
                    if riskfirst is None:riskfirst=row.copy()
            obs,reward,done,cut,info=env.step(action)
            row.update(lane_command_applied=info['lane_change_applied'],reward=reward)
            if env.ego_id in v.getIDList():row.update(after_speed=v.getSpeed(env.ego_id),after_accel=v.getAcceleration(env.ego_id),after_edge=v.getRoadID(env.ego_id),after_lane=v.getLaneIndex(env.ego_id))
            if len(first)<10:first.append(row)
            tail.append(row)
            if done or cut:
                outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout'
                collisions=[{key:getattr(x,key,None) for key in ('collider','victim','colliderSpeed','victimSpeed','type','lane','pos')} for x in tr.simulation.getCollisions()]
                break
        result=dict(seed=seed,condition=c,route=route,eval_seed=29101,outcome=outcome,seconds=(k+1)*.2,policy_sha=t.a.b.digest(model),source_commit=m['source_commit'],stats=dict(stats),first=first,first_risk=riskfirst,tail=list(tail),collisions=collisions,route_overrides=env.route_override_count,visibility=env.visibility,vmax=env.vmax,max_decel=env.max_decel)
        assert env.route_override_count==0
        t.save(dest,result);return {k:result[k] for k in ('seed','condition','route','outcome','seconds')}
    finally:env.close()

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--control',action='store_true');args=parser.parse_args()
    if args.control:
        OUT=r.t.ROOT/'results/collision_traffic_control_20261114'
        r.x.OUT=r.t.ROOT/'results/route_following_trials_20261114'
    OUT.mkdir(exist_ok=True,parents=True)
    specs=[(s,c,f'route_{i:02}') for i in range(1,17) for s in (11,22,33) for c in ('baseline','guidance','dwell')]
    if args.control:specs=[(11,'baseline',f'route_{i:02}') for i in (1,4,13,14)]
    with cf.ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(one,s) for s in specs]
        for i,f in enumerate(cf.as_completed(fs),1):
            print(f.result(),flush=True);r.t.save(OUT/'status.json',dict(completed=i,total=len(specs),stage='running'))
    r.t.save(OUT/'status.json',dict(completed=len(specs),total=len(specs),stage='complete'))
if __name__=='__main__':main()
