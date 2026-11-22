"""Matched SUMO interventions on every failed C3 final route-16 lane-0 holdout."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import hashlib,itertools,json,os
from utils.experiment_b import agent_for,save,torch
from utils.experiment_b_eval import evaluate_episode,initialize
from utils.experiment_c_env import OUT as C,SEEDS,make_c

OUT=C.parent/"experiment_d_v1"
RULE="Include D2/D3 if sustained produces more correct first-fork passages than single without more episode collisions over matched reachable interventions."
def probe(job):
    seed,sim,threshold,original=job
    policy=agent_for(seed)
    policy.load(str(C/f"C3_{seed}"/"policy_1000000.pt"))
    env=make_c("high_real_ratio","C3",seed,OUT/"probe_maps"/str(os.getpid()),audit=True)
    rows=[]
    try:
        for mode in ("original","single","sustained"):
            state={}; issued=0
            class Intervene:
                def predict(self,obs,deterministic=True):
                    nonlocal issued
                    action=policy.predict(obs,deterministic=True).copy()
                    edge,lane,_,_,_,_,sides,speed,d=env.snapshot
                    if not state and edge=="86198728" and d<=threshold:
                        with torch.no_grad():
                            probs=policy.policy(torch.as_tensor(obs)[None,:])[2].softmax(-1)[0].tolist()
                        state.update(step=env.step_count,edge=edge,lane=lane,distance=d,speed=speed,left_gaps=sides[1],
                            obs_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),probabilities=dict(zip(map(str,policy.policy.LANE_CMDS),probs)))
                    if state and edge==state["edge"] and lane==state["lane"] and state["lane"]==0:
                        elapsed=env.step_count-state["step"]
                        if mode=="single" and elapsed==0 or mode=="sustained" and elapsed<5 and issued==elapsed:
                            action[env.lane_change_index]=1
                            issued+=1
                    return action
            r=evaluate_episode(env,Intervene(),"route_16",sim,0)
            if mode=="original":
                full=dict(r,drops=env.c_drop,actions16=env.c_commands,reward_events=env.c_journal,reward_totals=env.c_totals)
                assert json.loads(json.dumps(full))==original
            if rows:assert state==rows[0]["state"]
            rows.append(dict(seed=seed,sim=sim,threshold=threshold,mode=mode,state=state,issued=issued,
                passed=r["first"]["actual"]==r["first"]["expected"],changed_to_main=r["first"]["returns"]>0,outcome=r["outcome"]))
        return rows
    finally:env.close()
def run():
    OUT.mkdir(exist_ok=True)
    jobs=[]
    for seed in SEEDS:
        f=C/f"C3_{seed}"/"hold_1000000_high_real_ratio.json"
        for r in json.loads(f.read_text()):
            if r["route"]=="route_16" and r["lane_case"]==0 and r["first"]["actual"]!=r["first"]["expected"]:
                jobs.extend((seed,r["sim_seed"],d,r) for d in (300,100))
    save(OUT/"probe_plan.json",dict(rule=RULE,cases=len(jobs),persistent_steps=5,reset_on_success=True))
    with ProcessPoolExecutor(6,initializer=initialize) as pool:rows=[r for batch in pool.map(probe,jobs) for r in batch]
    save(OUT/"counterfactual.json",rows)
    counts={}
    for mode in ("original","single","sustained"):
        g=[r for r in rows if r["mode"]==mode and r["state"] and r["state"]["lane"]==0]
        counts[mode]=dict(n=len(g),passed=sum(r["passed"] for r in g),moved=sum(r["changed_to_main"] for r in g),outcomes=Counter(r["outcome"] for r in g))
    a,b=counts["single"],counts["sustained"]
    extended=b["passed"]>a["passed"] and b["outcomes"]["collision"]<=a["outcomes"]["collision"]
    save(OUT/"decision.json",dict(counts=counts,arms=["D0","D1","D2","D3"] if extended else ["D0","D1"],rule=RULE))
if __name__=="__main__":run()

