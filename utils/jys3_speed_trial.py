"""Speed reward trial: shared longitudinal progress, separate target and overspeed costs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import math
from utils import jys3_pure_f as original
from utils.jys3_pure_f_report import aggregate

BASE_ENV=original.FactorEnv
ROOT=original.ROOT/"speed_trial"
ARMS=("F1","F4","F5")
SEEDS=(11,22,33)
MARKS=(*original.STEPS,65536)
SCREEN=34110000
HOLD=39110000

def speed_terms(speed,limit,leader):
    target=limit
    if leader is not None:
        gap, lead_speed=leader
        # Solve gap - buffer = v*tau + (v^2 - v_lead^2)/(2*b).
        b,tau=2.,1.5
        target=min(limit,max(0.,math.sqrt((b*tau)**2+lead_speed**2+2*b*max(0.,gap-2.))-b*tau))
    excess=max(0.,speed/limit-1.)
    return target,-.02*min(.5,excess)

class SpeedEnv(BASE_ENV):
    def __init__(self,arm,folder,seed):
        super().__init__("F1",folder,seed)
        self.arm=arm
        self.limits={}

    def step(self,action):
        obs,reward,done,cut,info=super().step(action)
        if done or self.snapshot is None:return obs,reward,done,cut,info
        veh=original.j.traci.vehicle
        lid=veh.getLaneID(self.ego_id)
        if lid not in self.limits:self.limits[lid]=original.j.traci.lane.getMaxSpeed(lid)
        limit=self.limits[lid]
        speed=self.snapshot[7]
        front=veh.getLeader(self.ego_id,120.)
        leader=(front[1],veh.getSpeed(front[0])) if front and front[0] else None
        safe,over_cost=speed_terms(speed,limit,leader)
        target=safe if self.arm=="F5" else limit
        # Tolerate a 10%-of-limit band, scale up to 0.002 per 0.2 s.
        match_cost=-.002*min(1.,max(0.,abs(speed-target)/limit-.1))
        if self.arm=="F1":match_cost=over_cost=0.
        info["speed_metrics"]=dict(measured=1,overspeed=int(speed>limit),overspeed10=int(speed>1.1*limit),
                                   following=int(safe<.8*limit),following_close_to_target=int(safe<.8*limit and abs(speed-safe)<=.1*limit),
                                   clear_low_speed=int(leader is None and speed<.5),speed_sum=speed,target_error=abs(speed-target),over_cost=over_cost,match_cost=match_cost)
        return obs,reward+over_cost+match_cost,done,cut,info

def setup():
    original.FactorEnv=SpeedEnv
    original.ROOT=ROOT

def train(job):
    setup()
    original.train(job)
    arm,seed,_=job
    p=ROOT/f"{arm}_{seed}"
    if arm=="F1" and seed==11:
        ref=ROOT.parent/"F1_11"/"policy_32768.pt"
        a,b=[original.j.torch.load(x,weights_only=True)["policy"] for x in [ref,p/ref.name]]
        assert all(original.j.torch.equal(a[k],b[k]) for k in a)

def evaluate(job):
    setup()
    original.evaluate(job)

def summarize():
    summary={}
    for base,marks in [(SCREEN,original.STEPS[1:]),(HOLD,(32768,65536))]:
        ref=None
        for seed in SEEDS:
            for arm in ARMS:
                p=ROOT/f"{arm}_{seed}"
                for n in marks:
                    rows=[r for rep in range(4) for r in json.loads((p/f"eval_{base}_{n}_{rep}.json").read_bytes())]
                    keys=[(r["case"],r["route"],r["traffic_seed"],r["start"]) for r in rows]
                    if ref is None:ref=keys
                    assert ref==keys and len(rows)==64
                    totals=Counter()
                    for r in rows:totals.update(r["speed_metrics"])
                    v=aggregate(rows)
                    v["speeding_pct"]=100*totals["overspeed"]/totals["measured"]
                    v["following_match_pct"]=round(100*totals["following_close_to_target"]/totals["following"],2) if totals["following"] else None
                    v["speed_kmh"]=3.6*totals["speed_sum"]/totals["measured"]
                    summary[f"{base}_{seed}_{arm}_{n}"]=v
    for seed in SEEDS:
        assert len({json.loads((ROOT/f"{arm}_{seed}"/"preflight.json").read_bytes())["initial_hash"] for arm in ARMS})==1
    original.save(ROOT/"summary.json",summary)

def main():
    ROOT.mkdir(exist_ok=False)
    assert math.isclose(speed_terms(22,20,None)[1]*50,-.1)
    assert math.isclose(speed_terms(24,20,None)[1]*50,-.2)
    assert speed_terms(0,20,(2,0))[0]==0 and speed_terms(0,20,None)[0]==20
    original.save(ROOT/"protocol.json",dict(arms=ARMS,seeds=SEEDS,marks=MARKS,screen=SCREEN,hold=HOLD,
        dt=.2,progress=2.,over="-.02*min(.5,max(0,v/L-1))",match="-.002*min(1,max(0,abs(v-target)/L-.1))",standstill_gap=2,tau=1.5,braking=2.))
    with ProcessPoolExecutor(4) as tp, ProcessPoolExecutor(6) as ep:
        jobs={tp.submit(train,(a,s,MARKS)):(a,s) for s in SEEDS for a in ARMS}
        evals=[]
        for f in as_completed(jobs):
            f.result()
            a,s=jobs[f]
            for base,marks in [(SCREEN,original.STEPS[1:]),(HOLD,(32768,65536))]:
                for n in marks:
                    evals.extend(ep.submit(evaluate,(a,s,n,r,base)) for r in range(4))
        for f in as_completed(evals):f.result()
    summarize()

if __name__=="__main__":main()
