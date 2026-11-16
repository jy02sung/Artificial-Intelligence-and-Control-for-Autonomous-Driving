"""On-policy PPO with separate supervised keep/safe batches and KL guard."""
import copy
import numpy as np
import torch
from torch.distributions import Normal,Categorical,kl_divergence
from algorithms.ppo import PPO
from utils.timeout_rollout import TimeoutBoundaryMixin

def distribution_kl(reference,current):
    rm,rs,rl,_=reference;cm,cs,cl,_=current
    return (kl_divergence(Normal(rm,rs),Normal(cm,cs)).sum(-1)+kl_divergence(Categorical(logits=rl),Categorical(logits=cl))).mean()

class KeepPPO(TimeoutBoundaryMixin,PPO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        actor=[p for n,p in self.policy.named_parameters() if not n.startswith('v_head.')]
        critic=list(self.policy.v_head.parameters())
        self.optimizer=torch.optim.Adam([{'params':actor,'lr':3e-5},{'params':critic,'lr':3e-4}])
        self.actor_parameters=actor
        self.target_kl=.015;self.keep_weight=.05;self.safe_weight=0.
        self.reference=None;self.keep=None;self.safe=None
        self.aux_rng=np.random.default_rng(11011)
    def set_supervision(self,reference,keep,safe=None):
        self.reference=copy.deepcopy(reference).eval()
        for p in self.reference.parameters():p.requires_grad_(False)
        self.keep=keep;self.safe=safe
    def supervised_batch(self,data):
        cats=np.unique(data['category']);ids=[]
        for _ in range(self.minibatch_size):
            group=cats[self.aux_rng.integers(len(cats))]
            choices=np.flatnonzero(data['category']==group)
            ids.append(choices[self.aux_rng.integers(len(choices))])
        return torch.tensor(data['obs'][ids],device=self.device),torch.tensor(data['actions'][ids],device=self.device)
    def update(self,advantages,returns):
        obs,act,oldlp=self.buffer.to_tensors(self.device)
        adv=torch.as_tensor(advantages,device=self.device)
        ret=torch.as_tensor(returns,device=self.device)
        if self.normalize_advantage:adv=(adv-adv.mean())/(adv.std()+1e-8)
        stats={k:0. for k in ('policy_loss','value_loss','keep_loss','safe_loss','accel_entropy','lane_entropy','clip_frac','approx_kl','keep_grad_norm','actor_grad_norm','safe_grad_norm')}
        batches=0;early=False;peak=0.
        with torch.no_grad(): before=self.policy(obs);before=tuple(x.detach().clone() for x in before)
        for epoch in range(self.n_epochs):
            ids=np.random.permutation(len(obs))
            for start in range(0,len(ids),self.minibatch_size):
                ix=ids[start:start+self.minibatch_size]
                lp,_,value=self.policy.evaluate_actions(obs[ix],act[ix]);logratio=lp-oldlp[ix];ratio=logratio.exp()
                # Check the whole current rollout, not a possibly noisy tiny minibatch.
                with torch.no_grad():
                    full_lp,_,_=self.policy.evaluate_actions(obs,act)
                    delta=full_lp-oldlp
                    kl=((delta.exp()-1)-delta).mean().item();peak=max(peak,kl)
                if kl>self.target_kl:early=True;break
                actor=-torch.minimum(ratio*adv[ix],ratio.clamp(1-self.clip_eps,1+self.clip_eps)*adv[ix]).mean()
                vloss=((value-ret[ix])**2).mean()
                mu,sd,logits,_=self.policy(obs[ix]);he=Normal(mu,sd).entropy().sum(-1).mean();hl=Categorical(logits=logits).entropy().mean()
                ko,_=self.supervised_batch(self.keep)
                with torch.no_grad():teacher=self.reference(ko)
                keep=distribution_kl(teacher,self.policy(ko))
                safe=torch.zeros((),device=self.device)
                if self.safe_weight and self.safe is not None and len(self.safe['obs']):
                    so,sa=self.supervised_batch(self.safe);sm,_,sl,_=self.policy(so)
                    safe=((sm[:,0]-sa[:,0])**2).mean()+torch.nn.functional.cross_entropy(sl,(1-sa[:,1].round()).long())
                if batches==0:
                    for key,term in [('actor_grad_norm',actor),('keep_grad_norm',self.keep_weight*keep),('safe_grad_norm',self.safe_weight*safe)]:
                        if term.requires_grad:
                            grad=torch.autograd.grad(term,self.actor_parameters,retain_graph=True,allow_unused=True)
                            stats[key]=float(torch.sqrt(sum((g.detach()**2).sum() for g in grad if g is not None)))
                loss=actor+self.value_coef*vloss+self.keep_weight*keep+self.safe_weight*safe-self.entropy_coef*(he+hl)-self.lane_entropy_coef*hl
                self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm);self.optimizer.step()
                vals=dict(policy_loss=actor.item(),value_loss=vloss.item(),keep_loss=keep.item(),safe_loss=safe.item(),accel_entropy=he.item(),lane_entropy=hl.item(),clip_frac=((ratio-1).abs()>self.clip_eps).float().mean().item(),approx_kl=kl)
                for k,v in vals.items():stats[k]+=v
                batches+=1
            if early:break
        for k in stats:
            if not k.endswith('grad_norm'):stats[k]/=max(batches,1)
        with torch.no_grad():
            after=self.policy(obs);err=after[-1]-ret
            final_lp,_,_=self.policy.evaluate_actions(obs,act);delta=final_lp-oldlp
            final_kl=((delta.exp()-1)-delta).mean().item()
            stats.update(value_mse=err.square().mean().item(),return_variance=ret.var().item(),explained_variance=1-err.var().item()/ret.var().item() if ret.var().item()>1e-8 else None,
                accel_mean_shift=(after[0]-before[0]).abs().mean().item(),rollout_kl_final=final_kl,rollout_kl_max=max(peak,final_kl),early_stop=early,optimizer_steps=batches,std=self.policy.current_std)
        return stats
