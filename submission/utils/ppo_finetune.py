from pathlib import Path
from utils import lane_change_cost as c
import torch
import utils.critic_warmup as anti
import utils.ppo_rollout as p
import utils.ppo_initial as q
import utils.kl_ppo as old
from utils.storage import read
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_ppo_clean_ladder"


def folder(a,s):
    return ROOT/f"{a}_s{s}"


def marks(a):
    if a=="M5-FT":
        return range(25000,150001,25000)
    return range(50000,250001 if a in ("M0","M1") else 300001,50000)


def fresh(a,s):
    if a in ("M0","M1"):
        return old.PPO(50,2,hidden_sizes=(256,256),activation="relu",seed=s,device="cpu",lr=.0001,
                       gamma=.99 if a=="M0" else .995,gae_lambda=.95,n_steps=2048,n_epochs=10,
                       minibatch_size=64,entropy_coef=0.,lane_entropy_coef=.002,lane_keep_bias=1.)
    agent=p.agent(s)
    if a=="M5-FT":
        agent.policy.load_state_dict(torch.load(folder(a,s)/"warm.pt",weights_only=True))
    return agent


class Episodes(c.Episodes):
    def __init__(self,env,arm,path,*,training_seed,warm=False):
        super().__init__(env,arm,path,training_seed=training_seed,warm=warm)
        self.costed=arm in ("M4","M5","M5-FT")
        self.risk=arm in ("M5","M5-FT")
        self.clear_stop_enabled=arm!="M0"
        if self.costed:
            self.transform=p.lateral_transform(env)
        if arm=="M5-FT":
            self.count=200000 if warm else 100000
        if arm=="M0":
            old.driving.ARRIVAL_REWARD,old.driving.PASS_REWARD=5.,1.
            old.driving.COLLISION_REWARD,old.driving.TIMEOUT_REWARD=-5.,-5.
            env.progress_cap,env.off_route_penalty=2.,-5.
        assert old.driving.COLLISION_REWARD==(-5. if arm=="M0" else -12.)


def train(a,s, *, end=None):
    torch.set_num_threads(1)
    f=folder(a,s)
    f.mkdir(exist_ok=True,parents=True)
    if a=="M5-FT" and not (f/"warm.pt").exists():
        anti.warmup(s,arm=a,dest=f,episode_type=Episodes)
    ref=anti.load_peak(s).policy.eval().requires_grad_(False) if a=="M5-FT" else None
    expected=(read(f/"warm_check.json")["sha256"] if ref is not None else
              read(ROOT/"protocol.json")["initial"][str(s)]["shared" if a in ("M0","M1") else "split"])
    def update(agent,e,adv,ret):
        if ref is not None:
            for group in agent.optimizer.param_groups:
                group["lr"]=3e-5-2e-5*max(0,agent.num_timesteps-agent.n_steps)/(150000-agent.n_steps)
            agent.lane_entropy_coef=.0005
        m=c.verified_update(agent,e,adv,ret,ref)
        m.update(e.stats)
        e.stats.clear()
        return m
    for n in marks(a):
        if end is not None and n>end:
            break
        if not (f/f"policy_{n}.pt").exists():
            p.train(a,n,directory=f,make_agent=lambda:fresh(a,s),make_episodes=Episodes,
                    expected_hash=expected,training_seed=s,update=update)
