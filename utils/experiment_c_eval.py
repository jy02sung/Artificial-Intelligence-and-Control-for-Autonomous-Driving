"""Independent, fixed-weight experiment C selection and held-out evaluation."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import time
from utils.experiment_b import agent_for,digest,save,m,STAGES
from utils.experiment_b_eval import evaluate_episode,initialize,score
from utils.experiment_c_env import OUT,ARMS,SEEDS,STEPS,SELECT,HOLD,make_c

def evaluate(job):
    arm,seed,step,stage,part=job
    directory=OUT/f"{arm}_{seed}"
    output=directory/f"{part}_{step}_{stage}.json"
    if output.exists():return
    agent=agent_for(seed)
    agent.load(str(directory/f"policy_{step}.pt"))
    agent.policy.eval()
    fingerprint=digest(agent)
    # A process owns its config directory; no two jobs can rewrite the same XML.
    import os
    env=make_c(stage,arm,seed,OUT/"eval_maps"/str(os.getpid()),audit=True)
    rows=[]
    try:
        for base,lane,route in itertools.product(HOLD if part=="hold" else SELECT,(0,1),m.ROUTES):
            r=evaluate_episode(env,agent,route,base+lane,lane)
            r.update(drops=env.c_drop,actions16=env.c_commands,reward_events=env.c_journal,reward_totals=env.c_totals)
            rows.append(r)
        pairs=json.loads((OUT/"design.json").read_text())["pairs"]
        score(rows,pairs)
        assert digest(agent)==fingerprint and agent.num_timesteps==0
        save(output,rows)
    finally:env.close()

def run():
    pairs=json.loads((OUT/"design.json").read_text())["pairs"]
    jobs=list(itertools.product(ARMS,SEEDS,STEPS,STAGES,["select"]))
    with ProcessPoolExecutor(4,initializer=initialize) as pool:
        while jobs:
            ready=[j for j in jobs if (OUT/f"{j[0]}_{j[1]}"/f"policy_{j[2]}.pt").exists()]
            if not ready:
                time.sleep(20)
                continue
            ready.sort(key=lambda j:j[2])
            list(pool.map(evaluate,ready))
            finished=set(ready)
            jobs=[j for j in jobs if j not in finished]
        jobs=[]
        for arm,seed in itertools.product(ARMS,SEEDS):
            d=OUT/f"{arm}_{seed}"
            scores={s:score(json.loads((d/f"select_{s}_high_real_ratio.json").read_text()),pairs) for s in STEPS}
            best=max(STEPS,key=lambda s:(scores[s][0],*scores[s][2:],-s))
            save(d/"selection.json",dict(selected=best,scores=scores))
            jobs.extend(itertools.product([arm],[seed],sorted({best,STEPS[-1]}),STAGES,["hold"]))
        list(pool.map(evaluate,jobs))
if __name__=="__main__":run()
