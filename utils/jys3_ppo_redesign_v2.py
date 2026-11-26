"""Matched three-arm PPO redesign. Run python -m utils.jys3_ppo_redesign_v2."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import subprocess
import time
import xml.etree.ElementTree as ET

from utils import jys3_relu_ppo_comparison as old
from utils import jys3_diagnose_ppo as diagnosis

import torch

final, read, save = old.final, old.read, old.save
ROOT = old.ROOT.parent / "jys3_ppo_redesign_v2"


def path_for(arm, step):
    return ROOT / arm / f"policy_{step}.pt"


def data_path(arm, step, panel):
    return ROOT / "evaluation" / f"{arm}_{step}_{panel}.json"


def rank(arm, step):
    t = final.tally(read(data_path(arm, step, "screen")))
    return t["arrival"], -t["collision"], -t["stall"]-t["timeout"], -t["off_route"], -step


def extend(arm, arrivals, upward):
    if arm == "A":
        return arrivals >= 107 and upward
    return arrivals >= 125 or upward


def preflight():
    protocol = old.preflight()
    protocol.update(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    critic="independent clone of policy trunk", kl_keep_change=[.9, .1],
                    eval_workers_max=4, C_train_vph=[[0,50000,3000],[50000,200000,15000]],
                    C_clear_stop=-10, seed=11, warm_steps_AB=20000,
                    upward="fourth screen exceeds both first and third",
                    checkpoint_selection="max arrivals, min collisions, min stall+timeout, min deviations, earliest")
    return protocol


class Demand:
    """Scale only the sixteen flow counts in the simulator's private route file."""
    def __init__(self, env):
        cfg = Path(env.sim.cfg_path)
        route_file = ET.parse(cfg).find("input/route-files").attrib["value"]
        self.path = cfg.parent / route_file
        self.original = self.path.read_bytes()
        self.high = ET.fromstring(self.original)
        flows = self.high.findall("flow")
        assert len(flows)==16 and sum(int(f.get("number")) for f in flows)==15000
        assert all(float(f.get("end"))-float(f.get("begin"))==3600 for f in flows)

    def set(self, vph):
        if vph == 15000:
            self.path.write_bytes(self.original)
        else:
            assert vph == 3000
            scaled = deepcopy(self.high)
            for flow in scaled.findall("flow"):
                n = int(flow.attrib["number"])
                assert n % 5 == 0
                flow.set("number", str(n//5))
            ET.ElementTree(scaled).write(self.path, encoding="utf-8", xml_declaration=True)
        flows = ET.parse(self.path).findall("flow")
        assert sum(int(f.get("number")) for f in flows)==vph


class Curriculum(old.TrainingEpisodes):
    def __init__(self, env, agent, callback):
        super().__init__(env,"C","ppo",callback,root=ROOT,clear_penalty=-10.)
        self.agent = agent
        self.demand = Demand(env)
        self.demand.set(3000)

    def reset(self):
        self.discounted = 0.
        self.length = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, cut, info = super().step(action)
        self.discounted += self.agent.gamma**self.length * reward
        self.length += 1
        if done or cut:
            old.append(self.folder/"returns.jsonl", dict(step=self.steps, seed=self.seed,
                       vph=3000 if self.steps<=50000 else 15000, outcome=info["outcome"],
                       discounted_return=self.discounted))
        if self.steps==50000:
            # Cut the low-density trajectory for GAE only. Bootstrap the final observation,
            # then mask the transition so GAE never crosses into the new traffic episode.
            boundary = dict(seed=self.seed, length=self.length, physical_terminal=bool(done or cut))
            if not (done or cut):
                boundary["bootstrap"] = self.agent.gamma*self.agent.get_value(obs)
                reward += boundary["bootstrap"]
                cut = True
            self.demand.set(15000)
            save(self.folder/"traffic_switch.json", boundary)
        return obs,reward,done,cut,info


def is_value(name):
    return name.startswith(("v_head.", "value_trunk."))


def separate(agent):
    policy = agent.policy
    policy.value_trunk = deepcopy(policy.trunk)
    agent.optimizer.add_param_group({"params":policy.value_trunk.parameters()})
    # A value-only backward pass must produce no gradients on any actor parameter.
    policy(torch.ones(1,50))[-1].sum().backward()
    for name, param in policy.named_parameters():
        assert (param.grad is not None) == is_value(name), name
    agent.optimizer.zero_grad(set_to_none=True)


def snapshot(agent,arm,step):
    model = agent.policy
    compatible = {k:v for k,v in model.state_dict().items() if not k.startswith("value_trunk.")}
    path = path_for(arm,step)
    final.atomic_torch(dict(policy=compatible, value_trunk=model.value_trunk.state_dict(),
                           obs_dim=50,activation="relu",hidden_sizes=[256,256],step=step),path)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        loaded = final.network(path)
        probe = torch.linspace(-1,1,150).reshape(3,50)
        for a,b in zip(loaded(probe)[:3],model(probe)[:3]):
            assert torch.equal(a,b)


def warm(agent, ref, arm):
    net=agent.policy
    params=[]
    for name,param in net.named_parameters():
        param.requires_grad_(is_value(name))
        if is_value(name):
            params.append(param)
    optimizer=torch.optim.Adam(params,lr=2e-5)
    with final.p.roll.baseline.simulator() as env:
        episodes=old.TrainingEpisodes(env,arm,"warm",lambda _:None,root=ROOT)
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<old.WARM:
            n=min(2048,old.WARM-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=n,timeout_is_failure=True)
            _,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            states=agent.buffer.to_tensors(agent.device)[0]
            target=torch.as_tensor(ret)
            def mse(ids):
                value=net.v_head(net.value_trunk(states[ids])).flatten()
                return (value-target[ids]).square().mean()
            before=mse(slice(None)).item()
            for _ in range(10):
                for ids in torch.randperm(n).split(64):
                    optimizer.zero_grad()
                    mse(ids).backward()
                    torch.nn.utils.clip_grad_norm_(params,.5)
                    optimizer.step()
            old.append(ROOT/arm/"warm_updates.jsonl",dict(step=agent.num_timesteps,mse_before=before,
                                                         mse_after=mse(slice(None)).item()))
    for name,value in net.state_dict().items():
        if not is_value(name):
            assert torch.equal(value.view(torch.uint8),ref.state_dict()[name].view(torch.uint8)),name
    assert not torch.equal(net.value_trunk[0].weight,net.trunk[0].weight)
    net.requires_grad_(True)
    save(ROOT/arm/"warm_check.json",dict(steps=agent.num_timesteps,actor_bits_identical=True,
                                        critic_trunk_changed=True,value_gradient_isolated=True))
    agent.num_timesteps=0


def train(arm):
    torch.set_num_threads(1)
    old.check_sources()
    folder=ROOT/arm
    folder.mkdir(parents=True,exist_ok=False)
    agent=old.PPO(50,2,activation="relu",hidden_sizes=(256,256),seed=11,device="cpu",lr=old.RATE[arm],
                  gamma=.995,gae_lambda=.95,n_steps=2048,n_epochs=10,minibatch_size=64,
                  entropy_coef=0.,lane_entropy_coef=.002 if arm=="C" else 0.,lane_keep_bias=1.)
    ref=None
    if arm in old.SOURCE:
        agent.policy.load_state_dict(final.network(old.ORIG[old.SOURCE[arm]]).state_dict())
        ref=deepcopy(agent.policy).eval().requires_grad_(False)
    separate(agent)
    if ref is not None:
        warm(agent,ref,arm)
    def on_step(step):
        if arm=="C" and step in (50000,100000,150000):
            snapshot(agent,arm,step)
    target=200000 if arm=="C" else 32768
    with final.p.roll.baseline.simulator() as env:
        episodes=(Curriculum(env,agent,on_step) if arm=="C" else
                  old.TrainingEpisodes(env,arm,"ppo",on_step,root=ROOT))
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<target:
            env.sums.clear()
            size=min(2048,target-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            metrics=agent.update(adv,ret,reference_policy=ref,reference_coef=old.BETA[arm],
                                 stratified_reference=ref is not None)
            step=agent.num_timesteps
            old.append(folder/"updates.jsonl",dict(step=step,metrics=metrics,rewards=dict(env.sums),
                                                  completed=dict(episodes.outcomes)))
            if (arm in old.SOURCE and step%old.MARK==0) or step==target:
                snapshot(agent,arm,step)
            if arm in old.SOURCE and step==32768:
                decision=folder/"extension.json"
                while not decision.exists():
                    time.sleep(1)
                if read(decision)["extend"]:
                    target=65536
    save(folder/"done.json",dict(step=target,source_hashes=old.check_sources()))


def evaluate(task):
    arm,step,panel,rep=task
    base,reps=old.PANELS[panel]
    assert rep<reps
    cases=[(route,rep,base) for route in final.ROUTES]
    return final.run_episodes((str(ROOT/"evaluation"),cases,True,path_for(arm,step),False,False),
                              record_commands=arm=="C",record_terminal=arm=="C")


def report(choices):
    from utils.jys3_ppo_redesign_report import generate
    generate(choices)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",choices=old.RATE)
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    parser.add_argument("--report",action="store_true")
    args=parser.parse_args()
    if args.train:
        train(args.train)
    elif args.report:
        report(read(ROOT/"selection.json"))
    else:
        old.run(args.workers,sys.modules[__name__])

