"""Factorial terminal reward comparison and conditional fork-reward follow-up."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import jys3_pure_f as exp
from utils import jys3_speed_trial as speed

ROOT=exp.ROOT/"terminal_ratio"
REF=speed.ROOT
SEEDS=(11,22,33)
MARKS=speed.MARKS
PAIR=speed.HOLD
FRESH=41110000
ARMS={"F1":(5,-15,.25),"arrival10":(10,-15,.25),"collision5":(5,-5,.25),"both":(10,-5,.25)}
GATE="At 65536, more completions than control pooled and in at least two training seeds; rank by pooled completions, then fewer collisions."

def params(arm):
    if arm.endswith("_fork1"):
        return (*ARMS[arm.removesuffix("_fork1")][:2],1.)
    return ARMS[arm]

class Env(speed.SpeedEnv):
    def __init__(self,arm,folder,seed):
        super().__init__("F1",folder,seed)
        assert self.progress_enabled and not self.nav_enabled and self.sim.max_steps==1000
        assert exp.j.TIMEOUT_REWARD==-5 and exp.j.STALL_SECONDS==30

def setup(arm):
    exp.j.ARRIVAL_REWARD,exp.j.COLLISION_REWARD,exp.j.PASS_REWARD=params(arm)
    exp.ROOT=ROOT
    exp.FactorEnv=Env

def train(job):
    arm,seed=job
    setup(arm)
    exp.train((arm,seed,MARKS))
    gethash=lambda p:json.loads((p/"preflight.json").read_bytes())["initial_hash"]
    assert gethash(ROOT/f"{arm}_{seed}")==gethash(REF/f"F1_{seed}")

def evaluate(job):
    arm,seed,n,rep,base=job
    setup(arm)
    if arm=="F1":
        # Reuse only its weights for unseen cases; save the new shard under this trial.
        import shutil
        dest=ROOT/f"F1_{seed}"
        dest.mkdir(exist_ok=True)
        # Baseline weights are prepared once by the parent before workers start.
        assert (dest/f"policy_{n}.pt").exists()
    exp.evaluate(job)

def stage(arms):
    with ProcessPoolExecutor(4) as tp,ProcessPoolExecutor(6) as ep:
        ts={tp.submit(train,(a,s)):(a,s) for s in SEEDS for a in arms}
        es=[]
        for f in as_completed(ts):
            f.result()
            a,s=ts[f]
            es.extend(ep.submit(evaluate,(a,s,n,r,PAIR)) for n in (32768,65536) for r in range(4))
        for f in as_completed(es):f.result()

def rows(arm,s,n,b):
    root=REF if arm=="F1" and b==PAIR else ROOT
    return [r for rep in range(4) for r in json.loads((root/f"{arm}_{s}"/f"eval_{b}_{n}_{rep}.json").read_bytes())]

def comparison(arms,n,b):
    result={}
    for a in arms:
        result[a]={}
        for s in SEEDS:
            old=rows("F1",s,n,b);new=rows(a,s,n,b)
            keys=lambda x:[(r["route"],r["case"],r["traffic_seed"],r["start"]) for r in x]
            assert len(old)==len(new)==64 and keys(old)==keys(new)
            c=Counter(r["outcome"] for r in new)
            transitions=Counter(p["outcome"]+"->"+q["outcome"] for p,q in zip(old,new))
            sixteen=[r for r in new if r["route"]==16 and r["start"][1]==0]
            result[a][s]=dict(counts=dict(c),transitions=dict(transitions),exit13=sum(r["route"]==13 and "169728835" in r["forks"] for r in new),main16=sum("86198725" in r["forks"] for r in sixteen),left16=sum(r["first_left"] is not None for r in sixteen),pass_then_crash=sum(bool(r["forks"]) and r["outcome"]=="collision" for r in new))
    return result

def main():
    import shutil
    ROOT.mkdir(exist_ok=False)
    exp.save(ROOT/"protocol.json",dict(settings=ARMS,marks=MARKS,seeds=SEEDS,paired_base=PAIR,holdout=FRESH,gate=GATE,timeout=-5,cap=1000,stall_seconds=30))
    for s in SEEDS:
        dest=ROOT/f"F1_{s}";dest.mkdir()
        shutil.copy2(REF/dest.name/"policy_65536.pt",dest/"policy_65536.pt")
    arms=list(ARMS)
    stage(arms[1:])
    summaries={str(n):comparison(arms,n,PAIR) for n in (32768,65536)}
    end=summaries["65536"]
    count=lambda a,s,k:end[a][s]["counts"].get(k,0)
    total=lambda a,k:sum(count(a,s,k) for s in SEEDS)
    gains=[a for a in arms[1:] if sum(count(a,s,"arrival")>count("F1",s,"arrival") for s in SEEDS)>=2 and total(a,"arrival")>total("F1","arrival")]
    winner=max(gains,key=lambda a:(total(a,"arrival"),-total(a,"collision"))) if gains else None
    exp.save(ROOT/"selection.json",dict(winner=winner,gate=GATE,paired=summaries))
    if winner:
        extra=winner+"_fork1";stage([extra]);arms.append(extra)
        for n in (32768,65536):summaries[str(n)].update(comparison([extra],n,PAIR))
    with ProcessPoolExecutor(8) as ep:
        list(ep.map(evaluate,[(a,s,65536,r,FRESH) for s in SEEDS for a in arms for r in range(4)]))
    exp.save(ROOT/"summary.json",dict(winner=winner,paired=summaries,holdout=comparison(arms,65536,FRESH)))

if __name__=="__main__":main()
