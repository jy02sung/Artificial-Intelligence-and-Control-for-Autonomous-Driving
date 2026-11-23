"""Experiment Y: risk and preparation reward changes in training only."""
import random
import sys
from collections import Counter
import numpy as np
from . import local_iterate as it
from .local_iterate import x, D, a

def main():
    x.OUT=D
    x.initialize(a.MODEL,True)
    agent,env=x.agent,x.env
    # Use the same native environment for evaluation, with no wrapper.
    opt=agent.optimizer
    for group in opt.param_groups:group["lr"]=8e-6
    native_reset=env.reset
    native_step=env.step
    rng=random.Random(55733)
    seeds=iter(rng.sample(range(941000000,942000000),8000))
    schedule=[]
    starts=[]
    prev=None
    credited=set()
    def reset():
        nonlocal prev
        if not schedule:
            schedule.extend(list(x.m.ROUTES)+x.HARD)
            rng.shuffle(schedule)
        route=schedule.pop()
        seed=next(seeds)
        starts.append([route,seed])
        prev,info=native_reset(seed=seed,options={"route":route})
        credited.clear()
        return prev,info
    def step(action):
        nonlocal prev
        cmd=env._quantize_lane_change(float(action[1]))
        adjust=0.
        if cmd:
            k=2-cmd
            dist_front=float(prev[1+4*k])*120
            dist_back=-float(prev[3+4*k])*120
            closing_front=max(0.,-float(prev[2+4*k])*env.vmax)
            closing_back=max(0.,float(prev[4+4*k])*env.vmax)
            risk=max(0.,1.-dist_front/(3.+closing_front),1.-dist_back/(3.+closing_back))
            valid=prev[21+k]>=0
            adjust=-1.5*risk if valid else 0.
            nav=env.navigation_context()
            lane=a.V.getLaneIndex(env.ego_id)
            edge=a.V.getRoadID(env.ego_id)
        obs,reward,done,cut,info=native_step(action)
        if cmd and not done and not cut and risk<0.01 and valid and nav["prepare_targets"] and nav["distance"] is not None:
            level=min(abs(lane-t) for t in nav["prepare_targets"])
            actual=a.V.getLaneIndex(env.ego_id)
            key=(nav["decision_edge"],level)
            if a.V.getRoadID(env.ego_id)==edge and min(abs(actual-t) for t in nav["prepare_targets"])<level and key not in credited:
                adjust+=0.8*min(1.,nav["distance"]/100.)
                credited.add(key)
        prev=obs
        return obs,reward+adjust,done,cut,info
    env.reset,env.step=reset,step
    obs,_=env.reset()
    ep=[0.,0]
    try:
        for i in range(1,13):
            obs,val,_=agent.collect_rollout(env,obs,ep_stats=ep)
            adv,ret=agent.buffer.compute_gae(val,agent.gamma,agent.gae_lambda)
            stats=agent.update(adv,ret)
            assert all(np.isfinite(v) for v in stats.values())
            if i in (4,12):
                agent.save(str(D/f"Y{i}.pt"))
            print("Y",i,len(starts),flush=True)
        x.write_json(D/"Y_training.json",dict(episodes=starts,steps=agent.num_timesteps,lr=8e-6))
    finally:env.close()
if __name__=="__main__":main()
