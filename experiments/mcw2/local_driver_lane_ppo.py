"""On-policy PPO of the native lane output after mcw2 DAgger, preserving its acceleration and features."""
import random,math,sys
import numpy as np
from . import local_driver_ppo as d
D=d.D
def train(name='lane'):
    direct=name!='lane'
    parent={'lane':d.FRESH,'direct':d.BC,'latest':d.SOURCE}[name]
    if not direct:d.wait(D/'fresh_ready.json')
    seed={'lane':43583,'direct':43587,'latest':43589}[name];d.setup(parent,True,seed)
    env,agent=d.env,d.agent; pi=agent.policy
    for p in pi.parameters():p.requires_grad_(False)
    for head in (pi.lane_head,pi.v_head):
        for p in head.parameters():p.requires_grad_(True)
    actor_lr=0.0004 if direct else 8e-5
    agent.optimizer=d.torch.optim.Adam([{'params':pi.lane_head.parameters(),'lr':actor_lr},{'params':pi.v_head.parameters(),'lr':2e-4}])
    agent.clip_eps=.12;agent.n_epochs=5;agent.value_coef=.05
    actor_start={k:v.clone() for k,v in pi.state_dict().items()}
    reset0,step0=env.reset,env.step
    lo={'lane':1370000000,'direct':1380000000,'latest':1390000000}[name]
    rng=random.Random(seed);seeds=iter(rng.sample(range(lo,lo+1000000),2000));schedule=[];episodes=[];prev=None
    def reset():
        nonlocal prev
        if not schedule:
            schedule.extend(d.m.ROUTES+['route_02','route_05','route_11','route_13','route_14']);rng.shuffle(schedule)
        rt,sd=schedule.pop(),next(seeds);episodes.append([rt,sd]);prev,meta=reset0(seed=sd,options={'route':rt});return prev,meta
    def step(action):
        nonlocal prev
        cmd=env._quantize_lane_change(float(action[1]));k=2-cmd;cost=0.
        if cmd and prev[21+k]>=0:
            f=max(0.,120.*prev[1+4*k]);r=max(0.,-120.*prev[3+4*k])
            front_need=4.+max(0.,-env.vmax*prev[2+4*k])
            rear_need=4.+max(0.,env.vmax*prev[4+4*k])
            cost=max(0.,1.-f/front_need,1.-r/rear_need)
        before=prev
        prev,reward,done,cut,info=step0(action)
        if direct:
            # Reward what the simulator actually did. Only a completed, safe lateral move towards
            # an open path earns preparation credit. No behaviour labels or action gates.
            available=before[21+k]>=0
            moved=cmd and info.get('lane_after') is not None and info['lane_after']!=info.get('lane_before')
            gain=max(0.,float(before[21+k]-before[23])) if moved and available else 0.
            safety=6.*cost if available and cmd else 0.
            reward=.3*reward+2.0*gain*(1.-cost)-safety-16.*int(info['collided'])+7.*int(info['arrived'])
        else:reward=reward-1.7*cost-12.*int(info['collided'])
        return prev,reward,done,cut,info
    env.reset=reset;env.step=step
    try:
        obs,_=reset();ep=[0.,0]
        for i in range(1,17 if direct else 9):
            obs,last,_=agent.collect_rollout(env,obs,ep_stats=ep)
            adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda);stats=agent.update(adv,ret)
            assert all(math.isfinite(v) for v in stats.values())
            if i in ((2,8,16) if direct else (2,8)):
                nm=f'{name}_{i}'
                for k,v in pi.state_dict().items():
                    if not k.startswith(('lane_head.','v_head.')):assert d.torch.equal(v,actor_start[k])
                delta=(pi.lane_head.weight-actor_start['lane_head.weight']).detach().abs().max().item();assert delta>0.
                agent.save(str(d.model(nm)))
                d.x.write_json(D/(nm+'_train.json'),dict(parent_sha256=d.sha(parent),parent=parent.relative_to(d.x.ROOT).as_posix(),sha256=d.sha(d.model(nm)),seed=seed,steps=agent.num_timesteps,episodes=episodes,max_lane_weight_change=delta,actor_lr=actor_lr,critic_lr=2e-4,clip=.12,epochs=5,reward='0.3 x branch reward, safe progress into a better-connected lane, short-gap and collision costs, arrival bonus' if direct else 'branch reward minus 1.7 * short gap risk at a requested lane change and 12 for ego collision'))
                print(nm,agent.num_timesteps,delta,flush=True)
    finally:env.close()
def select(name='lane'):
    ck=[f'{name}_{i}' for i in ((2,8) if name=='lane' else (2,8,16))];scores={k:d.evaluate(k,'screen') for k in ck}
    win=max(ck,key=lambda k:(scores[k].get('arrival',0),-scores[k].get('collision',0)))
    d.x.write_json(D/(name+'_screen_best.json'),dict(name=win,counts=scores[win]))
    c=d.evaluate(win,'dev');d.x.write_json(D/(name+'_dev_best.json'),dict(name=win,counts=c))
if __name__=='__main__':
    (train if sys.argv[1]=='train' else select)(*sys.argv[2:])
