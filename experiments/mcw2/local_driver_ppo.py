"""Local mcw2 driver DAgger -> on-policy PPO, using the original branch's network, observations and rewards."""
import json,os,sys,time,random,math,hashlib
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from . import local_experiment_x as x
m=x.m
torch=x.audit.torch
D=x.ROOT/'results'/'local_mcw2_driver_ppo'
D.mkdir(exist_ok=True)
R=x.ROOT/'results'/'mcw_runs'
SOURCE=R/'mcw2_occlusion_ppo'/'best_model.pt'
BC=R/'mcw2_agenda2_dagger_v2'/'best_model.pt'
FRESH=D/'dagger'/'best_model.pt'
parents={'saved':BC,'fresh':FRESH}
sha=x.digest
def read(p):return json.loads(p.read_bytes())
def wait(p):
    while not p.exists():time.sleep(1)
def setup(model,training=False,seed=11):
    global agent,env,hp
    torch.set_num_threads(1)
    a=SimpleNamespace(**(x.SPEC['args']|dict(seed=seed)))
    cfg=m.prepare_scenario(a.stage,D/'maps'/str(os.getpid()),color_by_type=a.color_by_type)
    road,nav=m.load_network_metadata(cfg.parent/'network.net.xml',cfg.parent/'highway.rou.xml',a.nav_horizon)
    env=m.make_env(cfg,road,nav,a,*m.build_configs(a));env.eval_mode=not training
    hp=x.SPEC['hparams']|dict(seed=seed,device='cpu')
    agent=m.PPO(obs_dim=47,act_dim=2,buffer=m.RolloutBuffer(),**hp)
    agent.load(str(model));agent.policy.train(training)
    assert a.agenda2 and a.occlusion and a.obs_noise and a.route_conn and not env.lc_safety_enabled
def dagger():
    folder=FRESH.parent;folder.mkdir(exist_ok=True)
    setup(SOURCE,True,17)
    cfg=read(R/'mcw2_agenda2_dagger_v2'/'manifest.json')['args']
    try:
        m.distill_safe_policy(agent,env,hp,folder,passes=cfg['distill_passes'],lr=cfg['distill_lr'],epochs=cfg['distill_epochs'],eval_episodes=1)
        x.write_json(D/'fresh_ready.json',dict(mentor=SOURCE.relative_to(x.ROOT).as_posix(),mentor_sha256=sha(SOURCE),bc_sha256=sha(FRESH),passes=cfg['distill_passes'],lr=cfg['distill_lr'],epochs=cfg['distill_epochs']))
    finally:env.close()
def model(name):
    if name=='mentor':return SOURCE
    return parents[name] if name in parents else D/(name+'.pt')
def train(name):
    if name=='fresh':wait(D/'fresh_ready.json')
    parent=model(name);seed=32031 if name=='saved' else 32033
    setup(parent,True,seed);rng=random.Random(seed)
    start=1310000000 if name=='saved' else 1320000000
    seeds=iter(rng.sample(range(start,start+1000000),3000))
    orig=env.reset;episodes=[];schedule=[]
    lr=2.5e-5 if name=='saved' else 8e-6
    for group in agent.optimizer.param_groups:group['lr']=lr
    def reset():
        if not schedule:schedule.extend(m.ROUTES);rng.shuffle(schedule)
        rt,sd=schedule.pop(),next(seeds);episodes.append([rt,sd])
        return orig(seed=sd,options={'route':rt})
    env.reset=reset
    try:
        obs,_=reset();ep=[0.,0]
        for i in range(1,13):
            obs,val,_=agent.collect_rollout(env,obs,ep_stats=ep)
            adv,ret=agent.buffer.compute_gae(val,agent.gamma,agent.gae_lambda)
            stat=agent.update(adv,ret)
            assert all(math.isfinite(v) for v in stat.values())
            if i in (2,6,12):
                ck=f'{name}_{i}';agent.save(str(model(ck)))
                x.write_json(D/(ck+'_train.json'),dict(parent_sha256=sha(parent),sha256=sha(model(ck)),seed=seed,lr=lr,steps=agent.num_timesteps,episodes=episodes,metrics=stat))
                print(ck,agent.num_timesteps,flush=True)
    finally:env.close()
def tasks(stage):
    lo,n={'screen':(1330000000,5),'dev':(1340000000,25),'final':(1350000000,100)}[stage]
    seeds=iter(random.Random(lo).sample(range(lo,lo+1000000),16*n))
    return [(r,next(seeds)) for _ in range(n) for r in m.ROUTES]
def drive(task):
    rt,sd=task;obs,_=env.reset(seed=sd,options={'route':rt})
    assert len(obs)==47 and x.audit.V.getSpeedMode('ego')==0 and x.audit.V.getLaneChangeMode('ego')==0
    while True:
        action=agent.predict(obs,deterministic=True);obs,_,dn,cut,info=env.step(action)
        if dn or cut:
            out='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'other'
            return dict(route=rt,seed=sd,outcome=out,steps=env.step_count)
def evaluate(name,stage):
    if name=='fresh':wait(D/'fresh_ready.json')
    elif '_' in name:wait(D/(name+'_train.json'))
    dest=D/(name+'_'+stage+'.json');p=model(name);seq=tasks(stage)
    if dest.exists():
        d=read(dest);assert d['sha256']==sha(p);return d['counts']
    workers=4 if stage=='screen' else 6
    with ProcessPoolExecutor(workers,initializer=setup,initargs=(p,)) as pool:
        rows=[]
        for row in pool.map(drive,seq):
            rows.append(row)
            if len(rows)%320==0:print(name,stage,len(rows),flush=True)
    assert [(r['route'],r['seed']) for r in rows]==seq
    counts=Counter(r['outcome'] for r in rows)
    x.write_json(dest,dict(model=p.relative_to(x.ROOT).as_posix(),sha256=sha(p),counts=counts,rows=rows))
    print(name,stage,counts,flush=True);return counts
def select(name):
    names=[name+'_'+str(i) for i in (2,6,12)]
    scores={ck:evaluate(ck,'screen') for ck in names}
    best=max(scores,key=lambda k:(scores[k].get('arrival',0),-scores[k].get('collision',0)))
    x.write_json(D/(name+'_screen_best.json'),dict(name=best,counts=scores[best]))
    c=evaluate(best,'dev');x.write_json(D/(name+'_dev_best.json'),dict(name=best,counts=c))
def decide():
    for family in parents:wait(D/(family+'_dev_best.json'))
    cands=[read(D/(k+'_dev_best.json')) for k in parents]
    winner=max(cands,key=lambda z:(z['counts'].get('arrival',0),-z['counts'].get('collision',0)))
    x.write_json(D/'freeze.json',winner|dict(sha256=sha(model(winner['name']))))
    evaluate(winner['name'],'final')
if __name__=='__main__':
    assert sha(BC)=='f2126af64a9052b03493b7fb1f889669bc053ee41f0f9ff0598047fc69408186' and sha(SOURCE)==x.BASE_HASH
    mode=sys.argv[1]
    if mode=='dagger':dagger()
    elif mode=='train':train(sys.argv[2])
    elif mode=='select':select(sys.argv[2])
    elif mode=='eval':evaluate(*sys.argv[2:])
    elif mode=='final':decide()
