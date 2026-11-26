"""Paired high-traffic Q sweep. Run as a module; the same command resumes durable checkpoints."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import subprocess

from utils import jys3_pure_ppo_overnight as p

np,torch,old,final,read,save=p.np,p.torch,p.old,p.final,p.read,p.save
ROOT=p.ROOT.parent/"jys3_pure_ppo_q_sweep"
REPORT=final.DOCS/"jys3-pure-ppo-q-sweep.md"
ARMS=tuple(f"Q{i}" for i in range(1,6))
SEEDS=(11,22,33)
END=400000
MARKS=tuple(p.marks(END))
HOLDS=tuple(k for k in old.PANELS if k!="screen")


def folder(arm,seed):
    return ROOT/f"{arm}_s{seed}"


def panel(arm,seed,n,name):
    return folder(arm,seed)/f"eval_{n}_{name}.json"


class GatedExperience(p.Experience):
    def __init__(self):
        super().__init__()
        self.accuracy=None
        self.applications=0

    def shaping(self,previous,current):
        if self.accuracy is None or self.accuracy<.70:
            return 0.
        self.applications+=1
        a,b=self.prob(np.stack([previous,current]))
        return float(np.clip(.3*(.995*b-a),-.1,.1))

    def update(self):
        info=super().update()
        val=info["online_validation"]
        if val:
            self.accuracy=val["correct"]/val["n"]
        info.update(next_gate_open=self.accuracy is not None and self.accuracy>=.70,
                    latest_accuracy=self.accuracy,applied_steps=self.applications)
        return info

    def state(self):
        return dict(**super().state(),accuracy=self.accuracy,applications=self.applications)

    def restore(self,s):
        super().restore(s)
        self.accuracy,self.applications=s["accuracy"],s["applications"]


class Episodes(p.Episodes):
    def __init__(self,env,arm,f,*,training_seed):
        super().__init__(env,arm,f,training_seed=training_seed)
        self.transform=p.lateral_transform(env)
        self.visits=Counter()
        if arm=="Q4":
            old.driving.COLLISION_REWARD=-16.
            env.off_route_penalty=-10.
        if arm=="Q5":
            self.exp=GatedExperience()

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


def train(arm,seed,n):
    rollout,batch=(4096,128) if arm=="Q3" else (2048,64)
    initial=read(ROOT/"protocol.json")["initial_sha256"][str(seed)]
    return p.train(arm,n,directory=folder(arm,seed),make_agent=partial(p.agent,seed,rollout,batch),
                   make_episodes=Episodes,expected_hash=initial,training_seed=seed,update=update)


def eval_part(arm,seed,n,name,rep):
    if arm!="R5":
        p.eval_part(arm,n,name,rep,directory=folder(arm,seed),lateral=True)
        return
    base,_=old.PANELS[name]
    out=folder(arm,seed)/"parts"/f"{n}_{name}_{rep}.json"
    if out.is_file():return
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=final.run_episodes((str(folder(arm,seed)),[(r,rep,base) for r in final.ROUTES],True,
                             old.ORIG["R5"],False,False),record_commands=True)
    archive=[r for r in read(old.data_path("R5",0,name)) if r["rep"]==rep]
    old.paired(archive,rows)
    assert [(r["outcome"],r["steps"]) for r in archive]==[(r["outcome"],r["steps"]) for r in rows]
    save(out,rows)


def phase(pool,training,evaluations):
    futures={}; shards=Counter()
    def enqueue(key):
        arm,seed,n,name=key
        if panel(*key).is_file():return
        for rep in range(old.PANELS[name][1]):
            futures[pool.submit(eval_part,*key,rep)]=("eval",key)
            shards[key]+=1
    for arm,seed,n in training:
        if (folder(arm,seed)/f"policy_{n}.pt").exists():
            enqueue((arm,seed,n,"screen"))
        else:
            futures[pool.submit(train,arm,seed,n)]=("train",(arm,seed,n))
    for key in evaluations:enqueue(key)
    while futures:
        fut=next(as_completed(futures))
        mode,key=futures.pop(fut)
        fut.result()
        if mode=="train":
            enqueue((*key,"screen"))
        else:
            shards[key]-=1
            if not shards[key]:
                arm,seed,n,name=key
                p.combine(arm,n,name,directory=folder(arm,seed))
                report()
                print(key,read(panel(*key))["metrics"],flush=True)


def ranking(arm,seed,n):
    m=read(panel(arm,seed,n,"screen"))["metrics"]
    return m["arrival"],-m["collision"],m["required_arrived"],-n


def best(arm,seed):
    return max(MARKS,key=lambda n:ranking(arm,seed,n))


def preflight():
    hashes=old.check_sources()
    inits={s:p.digest(p.agent(s).policy) for s in SEEDS}
    assert len(set(inits.values()))==3 and inits[11]==read(p.ROOT/"protocol.json")["initial_sha256"]
    reserved={n for name in old.PANELS for n in final.planned(*old.PANELS[name])}
    for s in SEEDS:
        base=205000000+(s-11)*1000000
        assert not any(base<=n<=base+END for n in reserved)
    with final.p.roll.baseline.simulator() as env:
        cfg=Path(env.sim.cfg_path)
        p.roads.configure_roads(env.sim,cfg.parent/"network.net.xml")
        forks,counts=p.roads.audit_map(env,cfg.parent.parent)
        assert len(forks)==7 and counts["wrong"]==0 and env.physical_forks and not env.lc_safety_enabled
        p.prev.Demand(env)
    return dict(initial_sha256=inits,protected_sha256=hashes,geometry=forks,route_signs=counts,
                baseline_commit="8724960eb98a288ce3938ed21e504272882ff20e",
                code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                seeds=list(SEEDS),steps=END,interval=p.MARK,workers=4,panels=old.PANELS,
                gamma=.995,gae_lambda=.95,lr=[.0001,.00001],target_kl=.015,
                gate="Latest nonempty pre-training validation rollout. Disable again below 0.70.",
                arm_changes=dict(Q1="lateral signs and map slots",Q2="Q1 plus linear LR and minibatch KL stop",
                                 Q3="Q2 plus 4096/128",Q4="Q2 plus -16 collision/-10 deviation",
                                 Q5="Q2 plus accuracy-gated PBRS alpha 0.3 clip 0.1"))


def report():
    from utils.jys3_pure_ppo_q_report import generate
    generate()


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/"protocol.json").exists():
        save(ROOT/"protocol.json",preflight())
    old.check_sources()
    report()
    with ProcessPoolExecutor(4) as pool:
        for n in MARKS:
            phase(pool,[(a,11,n) for a in ARMS],[])
        choices={a:best(a,11) for a in ARMS}
        winner=max(ARMS,key=lambda a:ranking(a,11,choices[a]))
        save(ROOT/"winner.json",dict(winner=winner,stage1=choices))
        for i,n in enumerate(MARKS):
            frozen=[(ARMS[i],11,choices[ARMS[i]],k) for k in HOLDS] if i<len(ARMS) else []
            if i==len(ARMS):frozen=[("R5",0,0,k) for k in HOLDS]
            phase(pool,[(winner,s,n) for s in SEEDS[1:]],frozen)
        phase(pool,[],[(winner,s,best(winner,s),k) for s in SEEDS[1:] for k in HOLDS])
    report()
    s=read(ROOT/"summary.json")
    assert s["screens"]==56 and len(s["q_holdouts"])==7 and s["three_seed"]
    save(ROOT/"finished.json",dict(winner=winner,protected_sha256=old.check_sources(),screens=56,holdouts=7))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",action="store_true")
    if parser.parse_args().report:report()
    else:main()
