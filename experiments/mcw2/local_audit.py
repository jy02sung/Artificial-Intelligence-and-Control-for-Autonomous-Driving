"""Frozen mcw2 holdout and deterministic incident replays. Run from the checkout root."""
import os
os.environ['IC_LIBSUMO']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, collections, hashlib, json, math, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import torch
from utils import mcw_train as m

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'local_holdout_1600'
REF=ROOT/'results/mcw_runs/eval_mcw2_occlusion_12288_80ep'
SPEC=json.loads((REF/'manifest.json').read_bytes())
MODEL=ROOT/SPEC['args']['resume']
V=m.traci.vehicle
S=m.traci.simulation

def setup():
    global env,agent
    torch.set_num_threads(1)
    a=SimpleNamespace(**SPEC['args'])
    cfg=m.prepare_scenario(a.stage,OUT/'maps'/str(os.getpid()),color_by_type=True)
    road,nav=m.load_network_metadata(cfg.parent/'network.net.xml',cfg.parent/'highway.rou.xml',a.nav_horizon)
    env=m.make_env(cfg,road,nav,a,*m.build_configs(a))
    env.eval_mode=True
    agent=m.PPO(obs_dim=47,act_dim=2,buffer=m.RolloutBuffer(),**(SPEC['hparams']|{'device':'cpu'}))
    agent.load(str(MODEL))
    agent.policy.eval()

def collisions():
    fields=('type','lane','pos','collider','victim','colliderType','victimType','colliderSpeed','victimSpeed')
    return [{k:getattr(c,k) for k in fields} for c in S.getCollisions() if 'ego' in (c.collider,c.victim)]

def snapshot(obs,action,partner):
    e=env.ego_id
    lid=V.getLaneID(e)
    data=dict(step=env.step_count,edge=V.getRoadID(e),lane=lid,pos=V.getLanePosition(e),
        remaining=m.traci.lane.getLength(lid)-V.getLanePosition(e),speed=V.getSpeed(e),
        accel=float(action[0]),cmd=int(action[1]),obs=obs.tolist(),leader=V.getLeader(e,120),
        occluders=len(env._occ_boxes))
    if partner and partner in V.getIDList():
        ex,ey=V.getPosition(e); px,py=V.getPosition(partner)
        env._occ_sensor_mask=True
        try: hidden=env._is_occluded(partner)
        finally: env._occ_sensor_mask=False
        data.update(foe_lane=V.getLaneID(partner),foe_pos=V.getLanePosition(partner),foe_speed=V.getSpeed(partner),
                    foe_distance=math.hypot(ex-px,ey-py),geometric_occlusion=hidden)
    return data

def trial(task):
    route,seed,kind,expect=task
    folder=OUT/kind
    dst=folder/f'{route}_{seed}.json'
    if dst.exists():return json.loads(dst.read_bytes())['outcome']
    row=dict(route=route,seed=seed)
    try: obs,_=env.reset(seed=seed,options={'route':route})
    except RuntimeError as exc:
        row.update(outcome='insertion_failure',error=str(exc))
    else:
        row.update(start_lane=V.getLaneID(env.ego_id))
        speed_sum=0.
        while True:
            action=agent.predict(obs,deterministic=True)
            detail=expect and env.step_count>=expect['steps']-26
            if detail:
                events=expect.get('contacts',[])
                partner=next((v for c in events for v in (c['collider'],c['victim']) if v!='ego'),None)
                frame=snapshot(obs,action,partner)
            obs,_,done,cut,info=env.step(action)
            speed_sum+=info.get('speed') or 0.
            if detail:
                frame.update(applied=bool(info['lane_change_applied']),lane_after=info['lane_after'])
                row.setdefault('history',[]).append(frame)
            if done or cut:
                out='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'other'
                row.update(outcome=out,steps=env.step_count,mean_speed=speed_sum/env.step_count,
                           contacts=collisions() if out=='collision' else [])
                break
    temp=dst.with_suffix('.tmp')
    temp.write_text(json.dumps(row,ensure_ascii=False),encoding='utf8');temp.replace(dst)
    return row['outcome']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('replicate','fresh','replay'): (OUT/name).mkdir(exist_ok=True)
    rng=random.Random(20261123)
    seeds=iter(rng.sample(range(2000000,900000000),1600))
    tasks=[(r,40001+ep*100+i,'replicate',None) for ep in range(5) for i,r in enumerate(m.ROUTES)]
    fresh=[(r,next(seeds),'fresh',None) for ep in range(100) for r in m.ROUTES]
    (OUT/'plan.json').write_text(json.dumps(dict(commit='ac2f9bc7a63594fe4a1b45df83f0e8cd09314f73',model=str(MODEL.relative_to(ROOT)),
        sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),sumo=S.getVersion(),torch=str(torch.__version__),reference=SPEC,
        protocol='best departure; deterministic; disjoint random traffic seeds; CPU libsumo, 6 workers',fresh=[t[:2] for t in fresh]),indent=2),encoding='utf8')
    start=time.monotonic()
    with ProcessPoolExecutor(6,initializer=setup) as pool:
        for batch in [tasks,fresh]:
            for i,_ in enumerate(pool.map(trial,batch),1):
                if i%40==0:print(batch[0][2],i,round(time.monotonic()-start,1),flush=True)
        failures=[json.loads(f.read_bytes()) for f in sorted((OUT/'fresh').glob('*.json'))]
        replay=[(x['route'],x['seed'],'replay',x) for x in failures if x['outcome']!='arrival' and 'steps' in x]
        list(pool.map(trial,replay))
    print('completed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
