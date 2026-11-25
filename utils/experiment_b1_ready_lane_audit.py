"""Replay observed ready-lane departures and measure credit on the actual transition."""
import json
from concurrent.futures import ProcessPoolExecutor
from utils.experiment_b1_1m import OUT, STAGES, agent_for, environment, initialize
from utils.experiment_b_eval import evaluate_episode

def replay(job):
    seed,stage,r,k=job
    env=environment(stage,"B1",seed,OUT/f"B1_{seed}"/f"audit_{stage}_{r['route']}_{r['sim_seed']}")
    ag=agent_for(seed)
    ag.load(str(OUT/f"B1_{seed}"/"policy_1000000.pt"))
    fork=r["forks"][k]
    transitions=[]
    original=env.step
    def step(action):
        snap=env.snapshot
        before=len(env.credits.paid)
        result=original(action)
        after=env.snapshot
        if snap[0]==fork["edge"] and snap[3]==int(k) and after and after[0]==snap[0] and after[1]!=snap[1]:
            transitions.append(dict(distance=snap[8],before=snap[1],after=after[1],reward=round(.3*(len(env.credits.paid)-before),2),sides=snap[6]))
        return result
    env.step=step
    try:
        new=evaluate_episode(env,ag,r["route"],r["sim_seed"],r["lane_case"])
        assert json.loads(json.dumps(new))==r
    finally:env.close()
    return dict(seed=seed,stage=stage,route=r["route"],sim=r["sim_seed"],edge=fork["edge"],targets=fork["targets"],actual=fork["actual"],expected=fork["expected"],outcome=r["outcome"],transitions=transitions)

if __name__=="__main__":
    jobs=[]
    for stage in STAGES:
        for f in sorted(OUT.glob("*/hold_1000000_"+stage+".json")):
            for r in json.loads(f.read_text()):
                for k,x in r["forks"].items():
                    if x.get("initially_ready") and x["wrong_entries"]:
                        jobs.append((int(f.parent.name.split("_")[1]),stage,r,k))
    with ProcessPoolExecutor(4,initializer=initialize) as pool:
        rows=list(pool.map(replay,jobs))
    (OUT/"ready_lane_replay.json").write_text(json.dumps(rows,indent=2))
    for row in rows:
        print(row['stage'],row['route'],row['outcome'],row['actual']==row['expected'],[(x['before'],x['after'],round(x['distance']),x['reward']) for x in row['transitions']])
