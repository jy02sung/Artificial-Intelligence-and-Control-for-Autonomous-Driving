"""On-policy clipped PPO with an immediate lane-risk advantage and KL to initialization."""
import sys,random,copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from . import local_driver_ppo as d
D=d.D
def train(name, updates=None):
    merge=name=='merge'
    seed={'gap':53171,'gaplatest':53173,'merge':53219}[name];parent=d.model('gap_4') if merge else d.BC if name=='gap' else d.SOURCE
    d.setup(parent,True,seed);pi=d.agent.policy;env=d.env;anchor=copy.deepcopy(pi).eval()
    rng=random.Random(seed);torch.manual_seed(seed);gen=np.random.default_rng(seed)
    lo={'gap':1450000000,'gaplatest':1460000000,'merge':1470000000}[name]
    seeds=iter(rng.sample(range(lo,lo+1000000),4000));sched=[];episodes=[]
    def reset():
        if not sched:
            sched.extend(d.m.ROUTES+(['route_02','route_05','route_13','route_14']*5 if merge else ['route_02','route_05','route_11','route_13','route_14']*2));rng.shuffle(sched)
        route,sd=sched.pop(),next(seeds);episodes.append((route,sd));return env.reset(seed=sd,options={'route':route})[0]
    opt=torch.optim.Adam([dict(params=pi.trunk.parameters(),lr=8e-5),dict(params=pi.lane_head.parameters(),lr=2.5e-4),dict(params=pi.mu_head.parameters(),lr=2e-5),dict(params=pi.v_head.parameters(),lr=3e-4)])
    def distro(logits):
        return Categorical(probs=0.96*(logits/1.5).softmax(-1)+0.04/3)
    def local_rewards(obs):
        # Potential rewards for the finite lane actions. None of these quantities enter inference.
        out=torch.zeros((len(obs),3))
        current=obs[:,23].clamp(0,1)
        for j,k in ((0,1),(2,3)):
            real=obs[:,21+k]>=0
            front=120*obs[:,1+4*k];rear=-120*obs[:,3+4*k]
            closing_front=(-27.78*obs[:,2+4*k]).clamp(min=0);closing_rear=(27.78*obs[:,4+4*k]).clamp(min=0)
            risk=torch.maximum(1-front/(5+closing_front),1-rear/(5+closing_rear)).clamp(0,1)
            delta=(obs[:,21+k].clamp(0,1)-current).clamp(-1,1)
            out[:,j]=torch.where(real, 3.5*delta*(1-risk)-7*risk-.04, -0.6)
            if merge:
                out[:,j]=torch.where(real,9.*delta*(1-risk)-6.5*risk-.06,-2.)
        return out
    obs=reset(); total=0;logs=[]
    try:
        for it in range(1, (updates or (12 if merge else 20)) + 1):
            buf=[]
            with torch.no_grad():
                for step in range(2048):
                    obs_t=torch.from_numpy(obs).unsqueeze(0)
                    mu,_,logits,v=pi(obs_t);lane=distro(logits);j=lane.sample()
                    motor=Normal(mu[:,0],.18);a=motor.sample();lp=motor.log_prob(a)+lane.log_prob(j)
                    act=np.array([float(a),1-int(j)],dtype=np.float32)
                    new,reward,done,cut,info=env.step(act);end=done or cut
                    buf.append((obs,act[0],int(j),float(lp),lane.probs[0].numpy(),float(v),.12*reward+5.*info['arrived']-10.*info['collided'],end))
                    obs=reset() if end else new
                tail=float(pi(torch.from_numpy(obs).unsqueeze(0))[3])
            states=torch.from_numpy(np.stack([z[0] for z in buf]));acc=torch.tensor([z[1] for z in buf]);act=torch.tensor([z[2] for z in buf]);oldlp=torch.tensor([z[3] for z in buf]);oldp=torch.from_numpy(np.stack([z[4] for z in buf]));values=np.array([z[5] for z in buf],np.float32)
            gae=np.zeros(len(buf),np.float32);g=0.
            for k in range(len(buf)-1,-1,-1):
                g=buf[k][6]+.99*(1-buf[k][7])*tail-values[k]+.99*.93*(1-buf[k][7])*g;gae[k]=g;tail=values[k]
            vt=torch.from_numpy(gae+values)
            with torch.no_grad():
                risk=local_rewards(states);local=(risk.gather(1,act[:,None]).squeeze(1)-(oldp*risk).sum(1))
                # Separate immediate safety credit from the longer-term driving return.
                adv=local+0.4*torch.from_numpy(gae);adv=(adv-adv.mean())/adv.std().clamp(min=.1)
                a0,_,lg0,_=anchor(states);base=lg0.log_softmax(-1)
            for epoch in range(8):
                for idx in np.array_split(gen.permutation(len(buf)),16):
                    mu,_,lg,_=pi(states[idx]);pr=distro(lg)
                    # Value regression may not update the shared policy features.
                    value=pi.v_head(pi.trunk(states[idx]).detach()).squeeze(-1)
                    lp=Normal(mu[:,0],.18).log_prob(acc[idx])+pr.log_prob(act[idx])
                    ratio=(lp-oldlp[idx]).exp()
                    surrogate=torch.minimum(ratio*adv[idx],ratio.clamp(.84,1.16)*adv[idx])
                    kl=F.kl_div(lg.log_softmax(-1),base[idx],reduction='batchmean',log_target=True)
                    loss=-surrogate.mean()+.05*F.smooth_l1_loss(value,vt[idx])+.035*kl+1.2*F.mse_loss(mu,a0[idx])
                    if merge:
                        # Sum the same clipped PPO objective over the three possible lane actions
                        # for the analytic immediate reward. Long-term returns still come from actual rollouts.
                        local_adv=risk[idx]-(oldp[idx]*risk[idx]).sum(-1,keepdim=True)
                        r=pr.probs/oldp[idx]
                        exact=(oldp[idx]*torch.minimum(r*local_adv,r.clamp(.84,1.16)*local_adv)).sum(-1)
                        need=(states[idx,23]<.98)&(torch.maximum(states[idx,22],states[idx,24])>states[idx,23]+.03)
                        old_change=oldp[idx,0]+oldp[idx,2]
                        weight=1.+24.*need.float()+3.*(old_change>.25).float()
                        loss=loss-.9*(weight*exact).sum()/weight.sum()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),1.);opt.step()
                    assert torch.isfinite(loss)
            total+=len(buf)
            if it in ((2,6,12) if merge else (4,10,20)):
                nm=f'{name}_{it}';d.agent.save(str(d.model(nm)))
                parent_path=parent.relative_to(d.x.ROOT) if parent.is_relative_to(d.x.ROOT) else parent
                d.x.write_json(D/(nm+'_train.json'),dict(parent=parent_path.as_posix(),parent_sha256=d.sha(parent),sha256=d.sha(d.model(nm)),seed=seed,steps=total,episodes=episodes,clip=.16,rollouts=it,epochs=8,local_reward='short gap / closing speed risk plus road connectivity difference',temperature=1.5,fixed_accel_std=.18,anchor_kl=.035,loss=float(loss.detach())))
                print(nm,total,flush=True)
    finally:env.close()
def select(name):
    ck=[f'{name}_{i}' for i in ((2,6,12) if name=='merge' else (4,10,20))];scores={k:d.evaluate(k,'screen') for k in ck}
    win=max(ck,key=lambda k:(scores[k].get('arrival',0),-scores[k].get('collision',0)))
    d.x.write_json(D/(name+'_screen_best.json'),dict(name=win,counts=scores[win]))
    c=d.evaluate(win,'dev');d.x.write_json(D/(name+'_dev_best.json'),dict(name=win,counts=c))
if __name__=='__main__':
    (train if sys.argv[1]=='train' else select)(sys.argv[2])
