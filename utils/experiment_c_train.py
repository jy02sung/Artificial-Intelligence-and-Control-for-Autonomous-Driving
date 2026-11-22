"""Train experiment C with atomic checkpoints and reproducible active-episode replay."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import json
import random
import os
import shutil
import uuid
import numpy as np
from utils.experiment_b import agent_for,digest,m,save,torch
from utils.experiment_c_env import OUT,ARMS,SEEDS,STEPS,make_c

class ScheduledEpisodes:
    def __init__(self,env,seed,folder,n=0):
        self.env,self.seed,self.folder,self.n=env,seed,folder,n
        self.actions=[]
    def reset(self):
        block,idx=divmod(self.n,16)
        order=list(m.ROUTES)
        random.Random(self.seed*100000+block).shuffle(order)
        route=order[idx]
        sim=self.seed*1000000+self.n
        self.env.ego_depart_lane=str(min(block%2,self.env.counts[self.env.routes[route][0]]-1))
        obs,info=self.env.reset(seed=sim,options={"route":route})
        from utils.experiment_b import traci
        record=[self.n,route,sim,self.env.snapshot[1],traci.vehicle.getLanePosition(self.env.ego_id),self.env.snapshot[7]]
        with self.folder.open("a") as f:print(json.dumps(record),file=f)
        self.actions=[]
        self.n+=1
        return obs,info
    def step(self,action):
        self.actions.append(np.array(action,copy=True))
        return self.env.step(action)

def atomic_torch(obj,path):
    temp=path.with_suffix(".tmp")
    torch.save(obj,temp)
    temp.replace(path)

def train(job, *, out=OUT, make_env=make_c):
    arm,seed=job
    torch.set_num_threads(1)
    folder=out/f"{arm}_{seed}"
    design=json.loads((out/"design.json").read_text())
    agent=agent_for(seed)
    assert digest(agent)==design["checks"][str(seed)][0]
    resume=folder/"resume.pt"
    state=torch.load(resume,weights_only=False) if resume.exists() else None
    if state:
        agent.policy.load_state_dict(state["policy"])
        agent.optimizer.load_state_dict(state["optimizer"])
        agent.num_timesteps=state["step"]
        if not (folder/f"policy_{state['step']}.pt").exists():
            atomic_torch(state["policy"],folder/f"policy_{state['step']}.pt")
    if agent.num_timesteps==STEPS[-1]:return
    env=make_env("high_real_ratio",arm,seed,folder/"train_map")
    log=folder/("episodes.jsonl" if not (folder/"episodes.jsonl").exists() else f"episodes_resume_{agent.num_timesteps}_{uuid.uuid4().hex[:6]}.jsonl")
    try:
        episodes=ScheduledEpisodes(env,seed,log,state["next_episode"]-1 if state else 0)
        obs,_=episodes.reset()
        stats=[0.,0]
        if state:
            for action in state["actions"]:
                obs,_,done,cut,_=episodes.step(action)
                assert not (done or cut)
            assert np.array_equal(obs,state["obs"])
            torch.set_rng_state(state["rng"])
            np.random.set_state(state["np_rng"])
            stats=state["stats"]
        for target in STEPS:
            if target<=agent.num_timesteps:continue
            while agent.num_timesteps<target:
                n=min(2048,target-agent.num_timesteps)
                obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=n,timeout_is_failure=arm!="C0")
                adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
                agent.update(adv,ret)
                save(folder/"progress.json",dict(steps=agent.num_timesteps,episodes=episodes.n))
            atomic_torch(dict(policy=agent.policy.state_dict(),optimizer=agent.optimizer.state_dict(),step=target,
                obs=obs,actions=episodes.actions,next_episode=episodes.n,stats=stats,rng=torch.get_rng_state(),
                np_rng=np.random.get_state()),resume)
            atomic_torch(agent.policy.state_dict(),folder/f"policy_{target}.pt")
    finally:env.close()

if __name__=="__main__":
    jobs=[(arm,seed) for seed in SEEDS for arm in ARMS]
    with ProcessPoolExecutor(6) as pool:list(pool.map(train,jobs))
