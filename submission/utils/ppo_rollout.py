from pathlib import Path
from collections import Counter
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
import xml.etree.ElementTree as ET
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
os.environ.setdefault("IC_LIBSUMO", "1")
import numpy as np
import torch
from utils import ppo_critic as prev
from utils import fork_geometry as roads
from utils.storage import save
from utils.storage import read
import utils.imitation as final
import utils.kl_ppo as old
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_pure_ppo_overnight"
LATERAL={"P4","P5"}
MARK=50000


def agent(seed=11, rollout=2048, minibatch=64):
    a=old.PPO(50,2,hidden_sizes=(256,256),activation="relu",seed=seed,device="cpu",
              gamma=.995,gae_lambda=.95,n_steps=rollout,n_epochs=10,minibatch_size=minibatch,
              lr=.0001,lane_keep_bias=1.,lane_entropy_coef=.002,entropy_coef=0.)
    prev.separate(a)
    return a


def digest(policy):
    h=hashlib.sha256()
    for k,v in sorted(policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def config(arm,step):
    if arm in {"P3","P4","P5"} and step<160000:
        if step<80000:
            return (5000,True) if arm=="P3" else (3000,False)
        return 8000,False
    return 15000,False


def lateral_transform(env):
    # The expert wait metric and raw initial-state fingerprint use the original evaluator.
    sim=env.sim
    original=sim.signs,sim.road_flags
    roads.configure_roads(sim,Path(sim.cfg_path).parent/"network.net.xml")
    signs,flags=sim.signs,sim.road_flags
    sim.signs,sim.road_flags=original
    assert set(signs)==set(sim.signs)
    def transform(x):
        edge,_,path,_,fork,*_=env.snapshot
        x=x.copy()
        x[34]=signs[path[fork],path[fork+1]] if fork is not None else 0.
        x[40:50]=flags.get(edge,sim.no_links)
        return x
    return transform


class Map:
    def __init__(self,env):
        self.env=env
        self.demand=prev.Demand(env)
        net=ET.parse(Path(env.sim.cfg_path).parent/"network.net.xml")
        self.links=defaultdict(list)
        for link in net.findall("connection"):
            self.links[link.get("from"),link.get("to")].append((int(link.get("fromLane")),int(link.get("toLane"))))
        self.short={}
        self.active=None
        for r in range(1,17):
            path=env.routes[f"route_{r:02d}"]
            forks=[i for i in range(len(path)-1) if (path[i],path[i+1]) in env.signs]
            if forks:
                suffix=path[forks[-1]:]
                ready=sorted(env.links[suffix[0]][suffix[1]])
                other=sorted(set(range(env.counts[suffix[0]]))-set(ready))
                self.short[r]=(suffix,ready,other)
        self.flexible=[r for r,(_,_,other) in self.short.items() if other]

    def set_stage(self,stage):
        if stage==self.active:
            return
        vph,short=stage
        root=deepcopy(self.demand.high)
        flows=root.findall("flow")
        sizes=np.array([int(f.get("number")) for f in flows])*vph/15000.
        numbers=np.floor(sizes).astype(int)
        numbers[np.argsort(-(sizes-numbers),kind="stable")[:vph-int(numbers.sum())]]+=1
        assert numbers.sum()==vph
        for flow,n in zip(flows,numbers):
            flow.set("number",str(n))
        if short:
            for r,(edges,_,_) in self.short.items():
                key=f"short_{r:02d}"
                root.insert(list(root).index(flows[0]),ET.Element("route",id=key,edges=" ".join(edges)))
                self.env.sim.routes[key]=edges
        ET.ElementTree(root).write(self.demand.path,encoding="utf-8",xml_declaration=True)
        assert sum(int(f.get("number")) for f in ET.parse(self.demand.path).findall("flow"))==vph
        self.active=stage

    def departure(self,r,block):
        if self.active[1] and r in self.short:
            _,ready,other=self.short[r]
            # Two forced-ready routes, thirteen flexible. Over 26 repeats per
            # route exactly 195 of 390 fork starts are ready and 195 need a change.
            assert len(self.short)==15 and len(self.flexible)==13
            use_ready=not other or (block+2*self.flexible.index(r))%26<11
            lanes=ready if use_ready else other
            return f"short_{r:02d}",lanes[block%len(lanes)],"ready" if use_ready else "change"
        key=f"route_{r:02d}"
        return key,min(block%2,self.env.counts[self.env.routes[key][0]]-1),"full"

    @lru_cache(maxsize=512)
    def ready(self,path,index,fork):
        allowed=set(self.env.links[path[fork]][path[fork+1]])
        for i in range(fork-1,index-1,-1):
            allowed={a for a,b in self.links[path[i],path[i+1]] if b in allowed}
        return allowed


class Episodes:
    clear_stop_enabled = True

    def __init__(self,env,arm,folder,*,training_seed=11):
        self.training_seed=training_seed
        self.env,self.arm,self.folder=env,arm,folder
        self.map=Map(env)
        self.transform=lateral_transform(env) if arm in LATERAL else lambda x:x
        self.exp=None
        self.steps=self.count=0
        old.driving.ARRIVAL_REWARD,old.driving.PASS_REWARD=10.,2.5
        old.driving.COLLISION_REWARD,old.driving.TIMEOUT_REWARD=-12.,-8.
        env.progress_cap,env.off_route_penalty=6.,-8.
        assert env.physical_forks and env.occlusion and not env.lc_safety_enabled and not env.prefill_enabled

    def observe(self,x):
        assert len(x)==52 and not x[50:].any()
        return self.transform(x[:50]) if self.env.snapshot is not None else x[:50]

    def reset(self):
        block,i=divmod(self.count,16)
        self.route=int(np.random.default_rng(self.training_seed+block).permutation(16)[i])+1
        self.seed=205000000+(self.training_seed-11)*1000000+self.count
        self.count+=1
        self.map.set_stage(config(self.arm,self.steps))
        key,lane,self.start_type=self.map.departure(self.route,block)
        self.env.ego_depart_lane=str(lane)
        self.start_lane=lane
        x,info=self.env.reset(seed=self.seed,options={"route":key})
        assert self.env.snapshot[0]==self.env.routes[key][0] and self.env.snapshot[1]==lane
        self.obs=self.observe(x)
        self.actions=[]
        self.idle=0
        self.segment=None
        self.seq=[]
        self.initial_ready={}
        self.prepaid=set()
        self.exp_sum=self.prep_sum=self.total=0.
        return self.obs,info

    def drive(self,action):
        self.actions.append(np.asarray(action).copy())
        x,r,d,c,info=self.env.step(action)
        return self.observe(x),r,d,c,info

    def visible(self,obs,snap):
        return snap[4] if snap and obs[35]>=0 else None

    def is_ready(self,snap):
        _,lane,path,idx,fork,*_=snap
        return lane in self.map.ready(tuple(path),idx,fork)

    def step(self,action):
        nxt,reward,done,cut,info=self.drive(action)
        self.steps+=1
        clear=self.clear_stop_enabled and info.get("speed") is not None and info["speed"]<2 and (info["gap"] is None or info["gap"]>25)
        self.idle=self.idle+1 if clear else 0
        if clear:
            reward-=.02
        if self.idle>=50 and not(done or cut):
            reward-=9.
            info["outcome"]="clear_stop"
            done=True
        self.total+=reward
        self.obs=nxt
        if done or cut:
            old.append(self.folder/"episodes.jsonl",dict(step=self.steps,seed=self.seed,route=self.route,lane=self.start_lane,
                       start_type=self.start_type,vph=self.map.active[0],outcome=info["outcome"],r_exp=self.exp_sum,
                       prep_adjustment=self.prep_sum,episode_return=self.total))
        return nxt,reward,done,cut,info

    def state(self):
        return {k:getattr(self,k) for k in ("count","steps","obs","actions","idle","segment","seq","initial_ready",
                                             "prepaid","exp_sum","prep_sum","total")}

    def restore(self,s):
        self.count=s["count"]-1
        self.steps=s["steps"]-len(s["actions"])
        self.reset()
        for action in s["actions"]:
            x,_,d,c,_=self.drive(action)
            assert not(d or c)
            self.obs=x
        assert np.array_equal(self.obs,s["obs"]), "Active drive replay differs from checkpoint"
        for k,v in s.items():
            setattr(self,k,v)


def folder(arm):
    return ROOT/arm


def train(arm,end,*,directory=None,make_agent=agent,make_episodes=Episodes,expected_hash=None,
          training_seed=11,update=None):
    torch.set_num_threads(1)
    f=directory or folder(arm)
    f.mkdir(exist_ok=True,parents=True)
    a=make_agent()
    initial=digest(a.policy)
    assert initial==(expected_hash or read(ROOT/"protocol.json")["initial_sha256"])
    save(f/"initial.json",dict(sha256=initial,seed=training_seed))
    resume=f/"resume.pt"
    state=torch.load(resume,weights_only=False) if resume.exists() else None
    logs=("episodes.jsonl","updates.jsonl")
    with final.simulator() as env:
        ep=make_episodes(env,arm,f,training_seed=training_seed)
        stats=[0.,0]
        if state:
            assert state["initial"]==initial and state["arm"]==arm
            a.policy.load_state_dict(state["model"])
            a.optimizer.load_state_dict(state["optimizer"])
            a.num_timesteps=state["step"]
            ep.restore(state["episodes"])
            if ep.exp:
                ep.exp.restore(state["experience"])
            stats=state["stats"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["np_rng"])
        else:
            ep.reset()
        for name in logs:
            with (f/name).open("ab") as out:
                offset=state["offsets"][name] if state else 0
                assert out.tell()>=offset
                out.truncate(offset)
        obs=ep.obs
        while a.num_timesteps<end:
            limit=min(end,((a.num_timesteps//MARK)+1)*MARK)
            if arm in {"P3","P4","P5"}:
                limit=min([limit]+[n for n in (80000,160000) if a.num_timesteps<n])
            size=min(a.n_steps,limit-a.num_timesteps)
            obs,last,_=a.collect_rollout(ep,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=a.buffer.compute_gae(last,a.gamma,a.gae_lambda)
            metrics=update(a,ep,adv,ret) if update else a.update(adv,ret)
            diag=ep.exp.update() if ep.exp else None
            n=a.num_timesteps
            old.append(f/"updates.jsonl",dict(step=n,size=size,metrics=metrics,predictor=diag))
            if config(arm,n)!=ep.map.active:
                # The last trajectory bootstrapped from its own observation in compute_gae.
                obs,_=ep.reset()
                stats=[0.,0]
            if n%MARK==0 or n==end:
                state=dict(arm=arm,step=n,initial=initial,model=a.policy.state_dict(),optimizer=a.optimizer.state_dict(),
                           episodes=ep.state(),experience=ep.exp.state() if ep.exp else None,stats=stats,
                           torch_rng=torch.get_rng_state(),np_rng=np.random.get_state(),
                           offsets={name:(f/name).stat().st_size for name in logs})
                final.atomic_torch(state,resume)
        n=a.num_timesteps
        assert n==end
        state=dict(policy={k:v for k,v in a.policy.state_dict().items() if not k.startswith("value_trunk.")},
                   activation="relu",step=n,obs_dim=50,hidden_sizes=[256,256])
        if getattr(a.policy,"value_trunk",None) is not None:
            state["value_trunk"]=a.policy.value_trunk.state_dict()
        final.atomic_torch(state,f/f"policy_{n}.pt")
    return arm,end
