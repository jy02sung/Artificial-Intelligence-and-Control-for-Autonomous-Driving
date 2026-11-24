"""Local scratch BC/DAgger -> PPO experiment. Run as a module from the worktree root."""
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import local_driver_ppo as base

m = base.m
ROOT = base.x.ROOT
OUT = ROOT / "results" / "scratch_bc_dagger_ppo_20261124"
SEED = 241124
WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))
MODELS = [f"ppo_{n}" for n in (4, 8, 16)]
OUT.mkdir(exist_ok=True)


def save(name, data):
    base.x.write_json(OUT / name, data)


def weights(name):
    return OUT / (name + ".pt")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init(model=None, training=False):
    global env, agent
    torch.set_num_threads(1)
    args = SimpleNamespace(**(base.x.SPEC["args"] | {"seed": SEED}))
    cfg = m.prepare_scenario(args.stage, OUT / "maps" / str(os.getpid()), color_by_type=args.color_by_type)
    road, nav = m.load_network_metadata(cfg.parent/"network.net.xml", cfg.parent/"highway.rou.xml", args.nav_horizon)
    env = m.make_env(cfg, road, nav, args, *m.build_configs(args))
    env.eval_mode = not training
    hp = base.x.SPEC["hparams"] | {"seed": SEED, "device":"cpu", "lr":2.5e-5}
    agent = m.PPO(obs_dim=47, act_dim=2, buffer=m.RolloutBuffer(), **hp)
    if model:
        agent.load(str(weights(model)))
    agent.policy.train(training)
    assert not env.lc_safety_enabled and args.occlusion and args.obs_noise


def expert(obs):
    # IDM longitudinal acceleration, with route-aware lane choice. No learned model.
    v = float(obs[0]) * env.vmax
    reach = obs[21:26]
    target_speed = 21.0
    best_index = min(range(5), key=lambda i: (-round(float(reach[i]), 3), abs(i-2)))
    wished = int(np.sign(2-best_index)) if reach[best_index] > reach[2]+0.006 else 0
    lane_cmd = 0
    lead = env._neighbor(0, leader=True)
    if wished:
        distance = max(0., float(reach[2])*env.visibility)
        target_speed = min(target_speed, max(4., math.sqrt(2.8*distance)))
        edge = m.traci.vehicle.getRoadID(env.ego_id)
        if not edge.startswith(":") and edge not in env.solid and env._lane_change_is_safe(wished):
            lane_cmd = wished
            other = env._neighbor(wished, leader=True)
            if other and (lead is None or other[0]<lead[0]):
                lead = other
    interaction = 0.
    if lead:
        gap, speed = lead
        desired = 2.5 + max(0., 1.05*v + v*(v-speed)/(2*math.sqrt(2.2*3.2)))
        interaction = (desired/max(0.2,gap))**2
    acc = 2.2*(1-(v/target_speed)**4-interaction)
    return np.array([np.clip(acc/(env.max_accel if acc>=0 else env.max_decel),-1,1), lane_cmd],np.float32)


def cases(low, per_route):
    rng = random.Random(low)
    seeds = iter(rng.sample(range(low,low+1000000),16*per_route))
    return [(r,next(seeds)) for _ in range(per_route) for r in m.ROUTES]


def outcome(info,cut):
    return "collision" if info["collided"] else "arrival" if info["arrived"] else "timeout" if cut else "other"


def episode(task):
    route, seed, beta, archive = task
    obs, _ = env.reset(seed=seed, options={"route":route})
    assert m.traci.vehicle.getSpeedMode(env.ego_id)==0 and m.traci.vehicle.getLaneChangeMode(env.ego_id)==0
    rng = random.Random(seed)
    states, labels = [], []
    while True:
        if beta >= 0:
            label = expert(obs)
            if archive:
                states.append(obs.copy())
                labels.append(label)
            action = label if rng.random()<beta else agent.predict(obs)
        else:
            action = agent.predict(obs)
        obs, _, done, cut, info = env.step(action)
        if done or cut:
            break
    if archive:
        folder = OUT/archive
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f"{route}_{seed}.npz", obs=states, actions=labels)
    return dict(route=route,seed=seed,steps=env.step_count,outcome=outcome(info,cut))


def roll(name, model, schedule, beta=-1, archive=None, training=False):
    with ProcessPoolExecutor(WORKERS, initializer=init, initargs=(model,training)) as pool:
        rows = list(pool.map(episode, [(r,s,beta,archive) for r,s in schedule],chunksize=2))
    assert [(r["route"],r["seed"]) for r in rows]==schedule
    summary = dict(model=model,sha256=digest(weights(model)) if model else None,rows=rows,
                   counts=dict(Counter(r["outcome"] for r in rows)), steps=sum(r["steps"] for r in rows))
    save(name+".json",summary)
    print(name,summary["counts"],summary["steps"],flush=True)
    return summary


def fit(name, parent, archives):
    init(parent,True)
    try:
        inputs, targets = [], []
        for archive in archives:
            for path in sorted((OUT/archive).glob("*.npz")):
                with np.load(path) as data:
                    inputs.append(data["obs"]); targets.append(data["actions"])
        x=torch.from_numpy(np.concatenate(inputs)); actions=torch.from_numpy(np.concatenate(targets))
        lane=(1-actions[:,1]).long()
        changed=torch.where(lane!=1)[0]
        assert len(changed)>0
        policy=agent.policy
        optimizer=torch.optim.Adam(policy.parameters(),lr=5e-4 if len(archives)==1 else 2e-4)
        gen=torch.Generator().manual_seed(SEED+len(archives))
        history=[]
        for epoch in range(12):
            sums=np.zeros(2); batches=0
            for raw in torch.randperm(len(x),generator=gen).split(384):
                extra=changed[torch.randint(len(changed),(64,),generator=gen)]
                idx=torch.cat([raw,extra])
                mu,_,logits,_=policy(x[idx])
                motor=F.smooth_l1_loss(mu[:len(raw),0],actions[raw,0],beta=.15)
                lateral=F.cross_entropy(logits,lane[idx])
                optimizer.zero_grad();(motor+lateral).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
                sums+=[motor.item(),lateral.item()];batches+=1
            history.append(dict(epoch=epoch+1,accel_loss=sums[0]/batches,lane_loss=sums[1]/batches))
        agent.save(str(weights(name)))
        save(name+"_fit.json",dict(parent=parent,parent_sha256=digest(weights(parent)),sha256=digest(weights(name)),
                                   labels=len(x),lane_changes=len(changed),epochs=history,archives=archives))
        print(name,"samples",len(x),"lateral examples",len(changed),flush=True)
    finally:
        env.close()


def ppo(parent):
    init(parent,True)
    rng=random.Random(SEED+3);torch.manual_seed(SEED+3);np.random.seed(SEED+3)
    schedule=cases(198000000,40)
    blocks=[schedule[i:i+16] for i in range(0,len(schedule),16)]
    for block in blocks:rng.shuffle(block)
    scheduled=iter([row for block in blocks for row in block])
    visited=[]
    def reset():
        rt,sd=next(scheduled);visited.append([rt,sd])
        return env.reset(seed=sd,options={"route":rt})[0]
    obs=reset()
    try:
        with (OUT/"ppo_updates.csv").open("w",newline="") as f:
            writer=None
            for update in range(1,17):
                agent.buffer.clear()
                for _ in range(2048):
                    action,lp,v=agent.act(obs)
                    new,reward,term,cut,_=env.step(action)
                    # A driving timeout ends the episode; do not pass returns across reset.
                    agent.buffer.add(obs,action,lp,reward,v,term or cut)
                    obs=reset() if term or cut else new
                adv,ret=agent.buffer.compute_gae(agent.get_value(obs),agent.gamma,agent.gae_lambda)
                metrics=agent.update(adv,ret)
                assert all(math.isfinite(v) for v in metrics.values())
                row=dict(update=update,steps=update*2048,**metrics)
                if writer is None:writer=csv.DictWriter(f,fieldnames=row);writer.writeheader()
                writer.writerow(row);f.flush()
                if update in (4,8,16):
                    name=f"ppo_{update}";agent.save(str(weights(name)))
                    save(name+"_train.json",dict(parent=parent,parent_sha256=digest(weights(parent)),sha256=digest(weights(name)),episodes=visited,**row))
                    print(name,row,flush=True)
    finally:
        env.close()


def paired(before, after):
    rng=np.random.default_rng(SEED)
    diff={}
    for category in ("arrival","collision","timeout"):
        x=np.array([int(b["outcome"]==category)-int(a["outcome"]==category) for a,b in zip(before["rows"],after["rows"])])
        grouped=x.reshape(-1,16).T
        bootstrap=np.zeros(10000)
        for g in grouped:
            bootstrap+=g[rng.integers(len(g),size=(10000,len(g)))].mean(1)/16
        diff[category]=dict(gain=int((x==1).sum()),loss=int((x==-1).sum()),delta_pp=100*x.mean(),ci95_pp=(100*np.quantile(bootstrap,[.025,.975])).tolist())
    return diff


def run():
    screen=cases(199000000,5);hold=cases(200000000,100)
    plan=OUT/"protocol.json"
    if not plan.exists():
        save(plan.name,dict(seed=SEED,source="mcw2_experiments a5ec0c4",student_initialization="torch random, no checkpoint",teacher="hand-coded IDM plus route reach and gap rules",
                            dagger_mixture=[1,.5,0], imitation_episodes=[96,64,64],ppo_budget=32768,ppo_screen_candidates=MODELS,
                            selection="max arrivals then min collisions then earliest step on screen",screen=screen,holdout=hold))
    init()
    try:
        if weights("initial").exists():
            for k,v in torch.load(weights("initial"),weights_only=True).items():assert torch.equal(v,agent.policy.state_dict()[k])
        else:agent.save(str(weights("initial")))
    finally:env.close()
    archives=[]
    parent="initial"
    for i,(name,n,beta) in enumerate([("bc",6,1.),("dagger1",4,.5),("dagger2",4,0.)]):
        archive=name+"_samples";archives.append(archive)
        if not (OUT/(name+"_collect.json")).exists():
            roll(name+"_collect",parent,cases(194000000+i*1000000,n),beta,archive)
        if not weights(name).exists():fit(name,parent,archives)
        parent=name
    if not weights("ppo_16").exists():ppo(parent)
    def evaluate(name, stage, seeds):
        path=OUT/f"{name}_{stage}.json"
        if path.exists():
            data=base.read(path);assert data["sha256"]==digest(weights(name));return data
        return roll(f"{name}_{stage}",name,seeds)
    scores={name:evaluate(name,"screen",screen)["counts"] for name in [parent]+MODELS}
    winner=max(MODELS,key=lambda n:(scores[n].get("arrival",0),-scores[n].get("collision",0)))
    save("freeze.json",dict(baseline=parent,ppo=winner,baseline_sha=digest(weights(parent)),ppo_sha=digest(weights(winner)),scores=scores))
    before=evaluate(parent,"holdout",hold);after=evaluate(winner,"holdout",hold)
    result=dict(before=before["counts"],after=after["counts"],ppo=winner,differences=paired(before,after))
    save("paired.json",result)
    print(result,flush=True)


if __name__=="__main__":
    run()
