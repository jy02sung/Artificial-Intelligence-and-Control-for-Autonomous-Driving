"""Train the initial policy by supervised BC, then update by PPO with new rollouts."""
import random,json,sys,os
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from . import local_bc_ppo as e
from .local_bc_ppo import D,x
def bc(fraction,updates=4500,recovery=False,kin=False,split=False,hard=False,lateral=False):
    e.setup(kin=kin,split=split)
    pi=e.agent.policy;pi.train()
    files=sorted((D/"demos").glob("*.npz"))
    assert len(files)>=768
    if recovery:
        extra=sorted((D/"recovery_demos").glob("*.npz"))
        assert len(extra)==768
        files+=extra
    if lateral:
        extra=sorted((D/"lateral_demos").glob("*.npz"));assert len(extra)==640
        files+=extra
    rng=random.Random(724202)
    rng.shuffle(files)
    val=set(files[:64])
    sets=[([],[]),([],[])]
    quality=Counter()
    for p in files:
        with np.load(p) as cache:
            outcome=str(cache["outcome"]);quality[outcome]+=1
            if outcome!="arrival":continue
            obs,act=cache["obs"],cache["actions"]
            dest=sets[int(p in val)];dest[0].append(e.kinematics(obs) if kin else obs);dest[1].append(act)
    def unpack(bucket):return [torch.from_numpy(np.concatenate(z)) for z in bucket]
    states,acts=unpack(sets[0]); sv,av=unpack(sets[1])
    positive=(acts[:,1].abs()==1).nonzero().flatten()
    ordinal=(acts[:,1]==0).nonzero().flatten()
    name=("kinlat" if lateral else "kinhard" if hard else "split" if split else "kin" if kin else "recover" if recovery else "bc")+str(round(fraction*100))
    slow=(states[:,0]<0.42).nonzero().flatten()
    opt=torch.optim.Adam(pi.parameters(),lr=4e-4)
    gen=torch.Generator().manual_seed(7712)
    def select(ix,n):return ix[torch.randint(len(ix),(n,),generator=gen)]
    npos=round(384*fraction)
    if hard:
        # Learn from actual SUMO labels in close-quarters scenes; every target is still the recorded expert action.
        near_left=(states[:,22]>=0)&((states[:,5]<.075)|(-states[:,7]<.075))
        near_right=(states[:,24]>=0)&((states[:,13]<.075)|(-states[:,15]<.075))
        close_keep=((acts[:,1]==0)&(near_left|near_right)).nonzero().flatten()
        close_front=((states[:,9]<.1)&(states[:,10]<.02)).nonzero().flatten()
    log=[]
    for i in range(1,updates+1):
        if i==int(.72*updates):
            for g in opt.param_groups:g["lr"]=1e-4
        choices=torch.cat([select(positive,npos),select(ordinal,384-npos)])
        if hard:choices=torch.cat([choices[:288],select(close_keep,96)])
        velocity=torch.randint(len(acts),(256,),generator=gen)
        if recovery:
            velocity=torch.cat([velocity[:160],select(slow,96)])
        if hard:velocity=torch.cat([velocity,select(close_front,64)])
        ids=torch.cat([choices,velocity])
        mu,_,logits,_=pi(states[ids]);target=(1-acts[choices,1]).long()
        lane=F.cross_entropy(logits[:len(choices)],target)
        at=acts[velocity,0];weight=1.+(at<-.15).float()*(0.35 if recovery else 2.)
        acc=(F.smooth_l1_loss(mu[len(choices):,0],at,reduction="none",beta=.18)*weight).mean()
        loss=lane+2.*acc
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),1.);opt.step()
        if i%500==0:print(name,i,round(lane.item(),3),round(acc.item(),3),flush=True)
    with torch.no_grad():
        preds=torch.cat([pi(z)[2].argmax(-1) for z in sv.split(2048)])
        valid=av[:,1].abs()<=1; target=(1-av[:,1]).long()
        score={str(k):[int(((preds==k)&(target==k)&valid).sum()),int(((target==k)&valid).sum())] for k in range(3)}
        pi.log_std.fill_(-1.4) # BC fits deterministic means; initialize exploration for PPO.
    e.agent.save(str(D/(name+".pt")))
    x.write_json(D/(name+"_fit.json"),dict(initialization="seed 8549, random; no inherited weights",teacher="native SUMO only, speedMode=31 laneChangeMode=1621",
        demo_outcomes=quality,train_rows=len(states),validation_rows=len(sv),positive_training_examples=len(positive),
        fraction=fraction,updates=updates,validation_by_true_lane=score,validation_episodes=[p.stem for p in sorted(val)]))
    e.env.close()
def ppo(parent,n,seed=9023,lr=1e-5,safe=False):
    e.setup(D/(parent+".pt"),True);agent,env=e.agent,e.env
    focus=os.environ.get("BC_PPO_FOCUS")=="1"
    if focus:
        # Fit the value baseline without letting its regression gradients erase the BC representation.
        agent.policy.v_head.register_forward_pre_hook(lambda mod,inputs:(inputs[0].detach(),))
        env.rw=dict(env.rw,collision_penalty=45.,arrival_bonus=16.)
    torch.manual_seed(seed);np.random.seed(seed)
    for g in agent.optimizer.param_groups:g["lr"]=lr
    rng=random.Random(seed)
    idseq=iter(rng.sample(range(1120000000,1130000000),12000))
    sched=[];episodes=[];reset=env.reset;nst=env.step;last=None
    def training_reset():
        nonlocal last
        if not sched:sched.extend(e.m.ROUTES);rng.shuffle(sched)
        route,sd=sched.pop(),next(idseq);episodes.append([route,sd])
        last,meta=reset(seed=sd,options={"route":route});return last,meta
    env.reset=training_reset
    if safe:
        def training_step(act):
            nonlocal last
            cmd=env._quantize_lane_change(float(act[1]));k=2-cmd
            risk=0.
            if cmd and last[21+k]>=0:
                front=120*last[1+4*k];rear=-120*last[3+4*k]
                risk=max(0.,1.-front/(4.+max(0.,-env.vmax*last[2+4*k])),1.-rear/(4.+max(0.,env.vmax*last[4+4*k])))
            last,r,done,cut,info=nst(act)
            return last,r-(2.0 if focus else 0.8)*risk,done,cut,info
        env.step=training_step
    obs,_=env.reset();stats=[0.,0];history=[]
    for i in range(1,n+1):
        obs,value,_=agent.collect_rollout(env,obs,ep_stats=stats)
        adv,ret=agent.buffer.compute_gae(value,agent.gamma,agent.gae_lambda)
        metrics=agent.update(adv,ret)
        assert all(np.isfinite(v) for v in metrics.values())
        history.append(metrics)
        if i in {4,12,24,n}:
            name=f"{parent}_p{seed}_{i}"
            agent.save(str(D/(name+".pt")))
            x.write_json(D/(name+"_train.json"),dict(steps=agent.num_timesteps,seed=seed,episodes=episodes,updates=history,lr=lr,safe_reward=safe,focus=focus,parent_sha256=x.digest(D/(parent+".pt"))))
        print(parent,"PPO",i,len(episodes),flush=True)
    env.close()
if __name__=="__main__":
    if sys.argv[1] in ("bc","recover","kin","split","kinhard","kinlat"):
        recovery=sys.argv[1]!="bc"
        bc(float(sys.argv[2]),8000 if sys.argv[1]=="kinlat" else 6500 if recovery else 4500,recovery,sys.argv[1] in ("kin","split","kinhard","kinlat"),sys.argv[1]=="split",sys.argv[1]=="kinhard",sys.argv[1]=="kinlat")
    else:ppo(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),bool(int(sys.argv[6])))
