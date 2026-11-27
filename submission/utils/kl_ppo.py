from pathlib import Path
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
import time
os.environ.setdefault("IC_LIBSUMO", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from algorithms.ppo import PPO
from utils import imitation as final
from utils import driving_env as driving
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_relu_ppo_comparison"
ORIG = {"BC": final.checkpoint(0), "R5": final.checkpoint(5)}
HASH = {"BC":"61f77baac57252c3adccea5af9e2855f135c0237a14f8458ea6fc9d7ab495e7e",
        "R5":"fb3a775fa4d4ccb56ab1c414b9e05af1a99ebf0623c992e096b063f2967728cd"}
SOURCE = {"A":"BC", "B":"R5"}
RATE = {"A":2e-5,"B":1e-5,"C":1e-4}
BETA = {"A":.5,"B":1.,"C":0.}
PANELS = {"screen":(95380000,8),"hold64":(81260100,4),
          "designated":(91000000,16),"additional":(104320000,16)}
WARM = 20000
MARK = 8192
from utils.storage import read
from utils.storage import save


def check_sources():
    hashes = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in ORIG.items()}
    assert hashes == HASH, hashes
    return hashes


def episode_bases(arm, phase):
    return 200000000 + "ABC".index(arm)*2000000 + (1000000 if phase=="ppo" else 0)


def append(path, record):
    with path.open("a",encoding="utf8") as f:
        print(json.dumps(record,allow_nan=False),file=f)


class TrainingEpisodes:
    def __init__(self, env, arm, phase, on_step, *, root=ROOT, clear_penalty=-7.):
        self.env,self.arm,self.phase,self.on_step=env,arm,phase,on_step
        self.base=episode_bases(arm,phase)
        self.count=self.steps=0
        self.folder=root/arm
        self.clear_penalty=clear_penalty
        self.outcomes=Counter()
        self.first_rewards=Counter()
        self.first_outcomes=Counter()
        env.progress_cap=6.
        env.off_route_penalty=-8.
        driving.ARRIVAL_REWARD,driving.PASS_REWARD=10.,2.5
        driving.COLLISION_REWARD,driving.TIMEOUT_REWARD=-10.,-8.
        assert env.physical_forks and env.occlusion and not env.lc_safety_enabled
        assert env.max_steps==1000 and env.step_length==.2 and len(env.routes)==16

    def reset(self):
        block,index=divmod(self.count,16)
        # Matched route order and departure lanes, distinct SUMO seeds across conditions.
        order=np.random.default_rng(11+block).permutation(16)+1
        self.route=int(order[index])
        path=f"route_{self.route:02d}"
        self.env.ego_depart_lane=str(min(block%2,self.env.counts[self.env.routes[path][0]]-1))
        self.seed=self.base+self.count
        assert self.count<250000
        self.count+=1
        obs,info=self.env.reset(seed=self.seed,options={"route":path})
        self.idle=0
        self.episode_return=0.
        assert obs.shape==(52,) and not obs[50:].any()
        return obs[:50],info

    def step(self,action):
        env=self.env
        obs,reward,done,cut,info=env.step(action)
        self.steps+=1
        clear=(info.get("speed") is not None and info["speed"]<2. and
               (info["gap"] is None or info["gap"]>25))
        self.idle=self.idle+1 if clear else 0
        # Physical terminal events take precedence. Charge per-step idle when observed.
        if clear:
            reward-=.02
            env.sums["idle_clear"]-=.02
        if self.idle>=50 and not (done or cut):
            reward+=self.clear_penalty
            env.sums["clear_stop"]+=self.clear_penalty
            done=True
            info["outcome"]="clear_stop"
        self.episode_return+=reward
        if self.arm=="C" and self.steps<=50000:
            parts={"arrival":10.*info["arrived"],"fork":2.5*info["fork_passed"],
                   "progress":info["progress_bonus"],"collision":-10.*info["collided"],
                   "off_route":-8.*info["off_route"],"timeout_stall":info["timeout_reward"],
                   "idle_clear":-.02*clear,"clear_stop":self.clear_penalty*(info["outcome"]=="clear_stop")}
            assert np.isclose(sum(parts.values()),reward)
            self.first_rewards.update(parts)
            if done or cut:
                self.first_outcomes[info["outcome"]]+=1
            if self.steps==50000:
                save(self.folder/"early_50k.json",dict(steps=self.steps,outcomes=dict(self.first_outcomes),
                     rewards=dict(self.first_rewards),return_sum=sum(self.first_rewards.values())))
        if done or cut:
            self.outcomes[info["outcome"]]+=1
            append(self.folder/f"{self.phase}_episodes.jsonl",
                   dict(step=self.steps,seed=self.seed,route=self.route,outcome=info["outcome"],
                        episode_return=self.episode_return))
        self.on_step(self.steps)
        return obs[:50],reward,done,cut,info


def store_checkpoint(agent,arm,step):
    final.atomic_torch(dict(policy=agent.policy.state_dict(),activation="relu",
                            obs_dim=50,hidden_sizes=[256,256],step=step),ROOT/arm/f"policy_{step}.pt")


def warm_critic(agent, reference, arm):
    for name,param in agent.policy.named_parameters():
        param.requires_grad_(name.startswith("v_head."))
    optimizer=torch.optim.Adam(agent.policy.v_head.parameters(),lr=2e-5)
    with final.simulator() as env:
        episodes=TrainingEpisodes(env,arm,"warm",lambda _:None)
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<WARM:
            size=min(2048,WARM-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            _,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            states,*_=agent.buffer.to_tensors(agent.device)
            with torch.no_grad():
                features=agent.policy.trunk(states)
            target=torch.as_tensor(ret)
            def mse(ids):
                return (agent.policy.v_head(features[ids]).flatten()-target[ids]).square().mean()
            before=mse(slice(None)).item()
            for _ in range(10):
                for ids in torch.randperm(size).split(64):
                    loss=mse(ids)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy.v_head.parameters(),.5)
                    optimizer.step()
            append(ROOT/arm/"warm_updates.jsonl",dict(step=agent.num_timesteps,mse_before=before,
                                                       mse_after=mse(slice(None)).item()))
    for k,v in agent.policy.state_dict().items():
        if not k.startswith("v_head."):
            assert torch.equal(v.view(torch.uint8),reference.state_dict()[k].view(torch.uint8)),k
    assert not torch.equal(agent.policy.v_head.weight,reference.v_head.weight)
    agent.policy.requires_grad_(True)
    save(ROOT/arm/"warm_check.json",dict(steps=WARM,actor_bits_identical=True,critic_changed=True))
    agent.num_timesteps=0


def train(arm, *, fixed_budget=False):
    torch.set_num_threads(1)
    check_sources()
    folder=ROOT/arm
    folder.mkdir(parents=True,exist_ok=False)
    agent=PPO(50,2,activation="relu",hidden_sizes=(256,256),seed=11,device="cpu",lr=RATE[arm],
              gamma=.995,gae_lambda=.95,n_steps=2048,n_epochs=10,minibatch_size=64,
              entropy_coef=0.,lane_entropy_coef=.002 if arm=="C" else 0.,lane_keep_bias=1.)
    reference=None
    if arm in SOURCE:
        agent.policy.load_state_dict(final.network(ORIG[SOURCE[arm]]).state_dict())
        reference=deepcopy(agent.policy).eval().requires_grad_(False)
        warm_critic(agent,reference,arm)
    target=200000 if arm=="C" else 32768
    def snapshot_inside_rollout(step):
        if arm=="C" and step in (50000,100000,150000):
            # Policy in force at this exact interaction, between full 2,048-step updates.
            store_checkpoint(agent,arm,step)
    with final.simulator() as env:
        episodes=TrainingEpisodes(env,arm,"ppo",snapshot_inside_rollout)
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<target:
            env.sums.clear()
            size=min(2048,target-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            metrics=agent.update(adv,ret,reference_policy=reference,reference_coef=BETA[arm])
            step=agent.num_timesteps
            append(folder/"updates.jsonl",dict(step=step,metrics=metrics,rewards=dict(env.sums),
                                              completed=dict(episodes.outcomes)))
            if (arm in SOURCE and step%MARK==0) or step==target:
                store_checkpoint(agent,arm,step)
            if arm in SOURCE and step==32768 and not fixed_budget:
                decision=folder/"extension.json"
                while not decision.exists():
                    time.sleep(2)
                if read(decision)["extend"]:
                    target=65536
    save(folder/"done.json",dict(step=target,source_hashes=check_sources()))
