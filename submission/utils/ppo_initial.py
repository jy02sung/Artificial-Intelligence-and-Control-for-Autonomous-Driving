from pathlib import Path
from collections import Counter
from utils import ppo_rollout as p
import numpy as np
import utils.kl_ppo as old
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_pure_ppo_q_sweep"
END=400000


def folder(arm,seed):
    return ROOT/f"{arm}_s{seed}"


class Episodes(p.Episodes):
    def __init__(self,env,arm,f,*,training_seed):
        super().__init__(env,arm,f,training_seed=training_seed)
        self.transform=p.lateral_transform(env)
        self.visits=Counter()
        assert arm == "Q1"

    def step(self,action):
        self.visits[self.route]+=1
        return super().step(action)


def update(agent,ep,adv,ret):
    n=agent.num_timesteps
    stable=ep.arm!="Q1"
    if stable:
        progress=np.clip((n-agent.n_steps)/(END-agent.n_steps),0.,1.)
        lr=float(.0001+(.00001-.0001)*progress)
        for group in agent.optimizer.param_groups:
            group["lr"]=lr
    metrics=agent.update(adv,ret,target_kl=.015 if stable else None)
    counts=[ep.visits[i] for i in range(1,17)]
    assert sum(counts)==len(adv)
    metrics.update(lr=agent.optimizer.param_groups[0]["lr"],route_steps=counts)
    ep.visits.clear()
    return metrics
