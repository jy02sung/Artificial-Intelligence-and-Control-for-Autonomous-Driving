"""Native SUMO demonstrations -> randomly initialized hybrid BC -> on-policy PPO."""
import os
os.environ["IC_LIBSUMO"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace,MethodType
import random, json, sys, hashlib, time
import numpy as np
import torch
import torch.nn.functional as F
from . import local_experiment_x as x
from env.sumo_env import SumoHighwayEnv
from utils.networks import HybridActorCritic
m=x.m; a=x.audit; v=a.V; sim=a.S
D=x.ROOT/"results"/"local_native_bc_ppo"
D.mkdir(exist_ok=True); (D/"demos").mkdir(exist_ok=True)
def tasks(kind,n):
    # Separated ranges above all earlier seeds, below int32.
    lo={"demo":1100000000,"ppo":1120000000,"dev":1140000000,"final":1160000000,"screen":1180000000,"safety_holdout":1240000000}[kind]
    ids=iter(random.Random(lo).sample(range(lo,lo+10000000),len(m.ROUTES)*n))
    return [(r,next(ids)) for _ in range(n) for r in m.ROUTES]
def kinematics(raw):
    """Short-range gap, closing speed and time headway, from noisy vehicle observations only."""
    z=np.asarray(raw);extra=[]
    speed=z[...,0]*27.78
    for k in range(5):
        for sg,gv,vv in [(1,1,2),(-1,3,4)]:
            gap=sg*z[...,gv+4*k]*120
            rel=z[...,vv+4*k]*27.78
            closing=np.maximum(0,-sg*rel)
            moving=speed if sg==1 else np.maximum(0,speed+rel)
            extra.extend([1/(1+gap/5),1/(1+gap/(2+closing)),1/(1+gap/(2+moving))])
    return np.concatenate([z,np.stack(extra,axis=-1)],axis=-1).astype(np.float32,copy=False)
class LocalSpeedActor(HybridActorCritic):
    """Independent learned acceleration head using local traffic and road measurements."""
    def __init__(self,dim):
        super().__init__(dim,(256,256))
        self.mu_head=torch.nn.Sequential(torch.nn.Linear(40,96),torch.nn.Tanh(),torch.nn.Linear(96,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
        self.speed_fields=[0,*range(5,17),21,22,23,24,25,31,32,34,36,*range(dim-24,dim-6)]
        assert len(self.speed_fields)==40
    def forward(self,obs):
        hidden=self.trunk(obs)
        mu=torch.tanh(self.mu_head(obs[...,self.speed_fields]))
        return mu,torch.exp(self.log_std).expand_as(mu),self.lane_head(hidden),self.v_head(hidden).squeeze(-1)
def setup(model=None,train=False,kin=False,split=False):
    global env,agent,dim,edges
    torch.set_num_threads(1)
    args=SimpleNamespace(**(x.SPEC["args"]|{"seed":8549,"safety":False,"route_conn":False,"prep_reward":False}))
    cfg=m.prepare_scenario(args.stage,D/"maps"/str(os.getpid()),color_by_type=True)
    road,nav=m.load_network_metadata(cfg.parent/"network.net.xml",cfg.parent/"highway.rou.xml",args.nav_horizon)
    env=m.make_env(cfg,road,nav,args,*m.build_configs(args))
    env.eval_mode=not train
    edges=sorted(e for e in nav.counts if not e.startswith(":")); idx={e:i for i,e in enumerate(edges)}
    split=split or (model is not None and Path(model).name.startswith("split"))
    kin=kin or split or (model is not None and Path(model).name.startswith("kin"))
    rawdim=37+3*len(edges);dim=rawdim+(30 if kin else 0)
    native_obs=env._get_obs
    env._lane_connectivity=MethodType(SumoHighwayEnv._lane_connectivity,env)
    outgoing={}
    for e,nxt in nav.links:
        if e in idx and nxt in idx:outgoing.setdefault(e,set()).add(nxt)
    def observe():
        raw=native_obs()
        obs=np.zeros(rawdim,dtype=np.float32)
        if env.ego_id not in v.getIDList():return np.zeros(dim,dtype=np.float32)
        ego=env.ego_id;road_id=v.getRoadID(ego);lane=v.getLaneID(ego)
        path=env.routes[env.episode_route];i=v.getRouteIndex(ego)
        left=m.traci.lane.getLength(lane)-v.getLanePosition(ego)
        obs[:31]=raw[:31]
        obs[31:37]=[min(left/120,1),min(left/400,1),v.getLaneIndex(ego)/5,min(1,m.traci.edge.getLaneNumber(road_id)/6),float(road_id.startswith(":")),min(m.traci.lane.getMaxSpeed(lane)/35,1)]
        if 0<=i<len(path):
            obs[37+idx[path[i]]]=1
            if i+1<len(path):obs[37+len(edges)+idx[path[i+1]]]=1
            distance=left
            for j in range(i,len(path)-1):
                if distance>400:break
                if len(outgoing.get(path[j],()))>1:
                    obs[37+2*len(edges)+idx[path[j+1]]]=1;break
                distance+=nav.lengths[path[j+1]]
        return kinematics(obs) if kin else obs
    env._get_obs=observe
    hp=x.SPEC["hparams"]|dict(seed=8549,device="cpu",lr=1e-5,n_epochs=5)
    agent=m.PPO(obs_dim=dim,act_dim=2,buffer=m.RolloutBuffer(),**hp)
    if split:
        agent.policy=LocalSpeedActor(dim)
        agent.optimizer=torch.optim.Adam(agent.policy.parameters(),lr=hp["lr"])
    if model:agent.load(str(model))
    agent.policy.train(train)
def demo(task):
    rt,sd=task
    file=D/"demos"/f"{rt}_{sd}.npz"
    if file.exists():
        with np.load(file) as d:return str(d["outcome"])
    obs,_=env.reset(seed=sd,options={"route":rt});ego=env.ego_id
    v.setSpeedMode(ego,31);v.setLaneChangeMode(ego,1621);v.setSpeed(ego,-1)
    obslist=[];actions=[];outcome="timeout"
    for step in range(3000):
        edge=v.getRoadID(ego);lane=v.getLaneIndex(ego);speed=v.getSpeed(ego)
        m.traci.simulationStep();env.step_count+=1
        if ego in sim.getCollidingVehiclesIDList():outcome="collision";break
        if ego in sim.getStartingTeleportIDList():outcome="teleport";break
        if ego in sim.getArrivedIDList():outcome="arrival";break
        if ego not in v.getIDList():outcome="other";break
        delta=(v.getSpeed(ego)-speed)/env.step_length
        cmd=v.getLaneIndex(ego)-lane if v.getRoadID(ego)==edge else 99
        obslist.append(obs)
        actions.append((np.clip(delta/(env.max_accel if delta>=0 else env.max_decel),-1,1),cmd))
        obs=env._get_obs()
    tmp=file.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,obs=np.asarray(obslist,np.float32),actions=np.asarray(actions,np.float32),outcome=outcome)
    tmp.replace(file);return outcome
def collect(n):
    with ProcessPoolExecutor(6,initializer=setup) as pool:
        counts=Counter()
        for i,out in enumerate(pool.map(demo,tasks("demo",n)),1):
            counts[out]+=1
            if i%96==0:print("demo",i,counts,flush=True)
    x.write_json(D/"demo_summary.json",dict(outcomes=counts,tasks=tasks("demo",n)))
def drive(task):
    r,s=task;obs,_=env.reset(seed=s,options={"route":r})
    assert v.getSpeedMode(env.ego_id)==0 and v.getLaneChangeMode(env.ego_id)==0 and not env.lc_safety_enabled
    while True:
        act=agent.predict(obs,deterministic=True)
        obs,_,done,cut,info=env.step(act)
        if done or cut:
            outcome="collision" if info["collided"] else "arrival" if info["arrived"] else "timeout" if cut else "other"
            return {"route":r,"seed":s,"outcome":outcome,"steps":env.step_count}
def evaluate(model,stage):
    file=D/(model+"_"+stage+".json")
    if file.exists():
        data=json.loads(file.read_bytes());assert data["sha256"]==x.digest(D/(model+".pt"));return Counter(r["outcome"] for r in data["rows"])
    work=tasks(stage,{"dev":25,"final":100,"safety_holdout":100,"screen":5}[stage])
    with ProcessPoolExecutor(6,initializer=setup,initargs=(D/(model+".pt"),)) as pool:
        rows=[]
        for i,row in enumerate(pool.map(drive,work),1):
            rows.append(row)
            if i%160==0:print(model,stage,i,flush=True)
    assert [(r["route"],r["seed"]) for r in rows]==work
    x.write_json(file,dict(sha256=x.digest(D/(model+".pt")),rows=rows))
    c=Counter(r["outcome"] for r in rows);print(model,stage,c,flush=True);return c
if __name__=="__main__":
    if sys.argv[1]=="collect":collect(int(sys.argv[2]))
    else:evaluate(sys.argv[1],sys.argv[2])
