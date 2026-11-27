from pathlib import Path
import math
from functools import lru_cache
import xml.etree.ElementTree as ET

import utils.lane_change_cost as C
import utils.ppo_finetune as L
import utils.ppo_rollout as P
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_m5ft_targeted"
STEPS = {22:75000, 33:125000}
FOCUS = (10,11,13,14)


def source(s):
    return L.folder("M5-FT", s)/f"policy_{STEPS[s]}.pt"


def folder(a,s):
    return ROOT/a/f"s{s}"


@lru_cache(maxsize=256)
def order(s,block):
    return np.random.default_rng(9013+s+block).permutation(64)


def start(arm,s,count):
    block,i=divmod(count,64)
    slot=int(order(s,block)[i])
    group,r=divmod(slot,16)
    route=r+1 if group<3 or arm=="A" else FOCUS[r%4]
    lane=(group+block)%2 if group<3 or arm=="A" else (r//4+block)%2
    return route,lane,arm=="C" and group==3 and route in (13,14),1500000000+s*100000+count


def load(s):
    model=P.agent(s)
    cp=torch.load(source(s),weights_only=True,map_location="cpu")
    assert cp["step"] == STEPS[s]
    weights=dict(cp["policy"])
    weights.update({"value_trunk."+k:v for k,v in cp["value_trunk"].items()})
    model.policy.load_state_dict(weights,strict=True)
    assert not model.optimizer.state
    return model


class Episodes(L.Episodes):
    def __init__(self,env,arm,path,*,training_seed,schedule):
        super().__init__(env,"M5-FT",path,training_seed=training_seed)
        self.schedule=schedule
        self.strong=arm=="D"
        self.count=0
        self.map.set_stage((15000,False))
        self.length={}
        for route in (13,14):
            full=env.routes[f"route_{route:02d}"]
            suffix=full[full.index("170059995"):]
            assert suffix[:3]==["170059995","170059994","1063527748"]
            key=f"target_{route}"
            env.sim.routes[key]=suffix
            self.length[route]=sum(env.navigation.lengths[x] for x in full)+sum(env.navigation.link_lengths.get(pair,0) for pair in zip(full,full[1:]))
            doc=ET.parse(self.map.demand.path)
            flows=doc.getroot().findall("flow")
            doc.getroot().insert(list(doc.getroot()).index(flows[0]),ET.Element("route",id=key,edges=" ".join(suffix)))
            assert sum(int(f.get("number")) for f in flows)==15000
            doc.write(self.map.demand.path,encoding="utf8",xml_declaration=True)

    def reset(self):
        self.route,lane,short,self.seed=start(self.schedule,self.training_seed,self.count)
        self.count+=1
        env=self.env
        key=f"target_{self.route}" if short else f"route_{self.route:02d}"
        self.start_lane=min(lane,env.counts[env.routes[key][0]]-1)
        env.ego_depart_lane=str(self.start_lane)
        raw,info=C.start_trace(lambda:env.reset(seed=self.seed,options={"route":key}),self.xml)
        env.progress_cap=6.*(env.total-env.start)/self.length[self.route] if short else 6.
        assert env.snapshot[:2]==(env.routes[key][0],self.start_lane)
        assert env.high==env.start and not env.sim.paid_forks
        if short:
            assert 0<env.progress_cap<6 and math.isclose(env.progress_cap/(env.total-env.start),6/self.length[self.route])
        self.start_type="upstream" if short else "full"
        self.obs=self.observe(raw)
        self.actions=[]
        self.idle=0
        self.segment=None
        self.seq=[]
        self.initial_ready={}
        self.prepaid=set()
        self.exp_sum=self.prep_sum=self.total=0.
        return self.obs,info

    def drive(self,action):
        before=self.stats["risk_moves"]
        obs,reward,done,cut,info=super().drive(action)
        if self.strong:
            reward-=.75*(self.stats["risk_moves"]-before)
        return obs,reward,done,cut,info

    def step(self,action):
        edge=self.env.snapshot[0]
        key=f"{self.route}/{self.start_type}"
        self.stats["steps/"+key]+=1
        if not self.actions:
            self.stats["starts/"+key]+=1
        if edge!=getattr(self,"prior_edge",None) or not self.actions:
            self.stats["visits/"+key+"/"+edge]+=1
        self.prior_edge=edge
        return super().step(action)

    def restore(self,state):
        super().restore(state)
        self.stats.clear()
        self.prior_edge=None


def train(job, *, directory=None, marks=(12500,25000), collision_reward=None, gae_lambda=None):
    arm,s,schedule=job
    ref=load(s).policy.eval().requires_grad_(False)
    dest=directory or folder(arm,s)
    def make_agent():
        agent=load(s)
        if gae_lambda is not None:
            assert agent.gamma==.995 and agent.gae_lambda==.95
            agent.gae_lambda=gae_lambda
        return agent
    def build(env,name,path,*,training_seed):
        episodes = Episodes(env,arm,path,training_seed=training_seed,schedule=schedule)
        if collision_reward is not None:
            assert L.old.driving.COLLISION_REWARD == -12.
            L.old.driving.COLLISION_REWARD = collision_reward
        return episodes
    def update(agent,ep,adv,ret):
        for g in agent.optimizer.param_groups:
            g["lr"]=1e-5
        agent.lane_entropy_coef=.0005
        m=C.verified_update(agent,ep,adv,ret,ref)
        if gae_lambda is not None:
            assert agent.gae_lambda==gae_lambda
            m["gae_lambda"]=agent.gae_lambda
        m.update(ep.stats)
        ep.stats.clear()
        return m
    for n in marks:
        if not (dest/f"policy_{n}.pt").exists():
            P.train("M5-FT",n,directory=dest,make_agent=make_agent,make_episodes=build,training_seed=s,
                    expected_hash=P.digest(ref),update=update)
