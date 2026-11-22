"""D training and matched four-start evaluation, independently restartable."""
from concurrent.futures import ProcessPoolExecutor
import itertools,json,os,time,sys
from utils.experiment_b import agent_for,digest,m,save,STAGES
from utils.experiment_b_eval import evaluate_episode,initialize
from utils.experiment_c_env import SEEDS,STEPS
from utils.experiment_c_train import train
from utils.experiment_d_check import SELECT,HOLD
from utils.experiment_d_env import OUT,environment

def training(job):
    return train(job,out=OUT,make_env=environment)

def score(rows):
    by={(r["route"],r["sim_seed"],r["lane_case"]):r for r in rows}
    assert len(by)==len(rows)
    ok=lambda r:r["first"]["actual"]==r["first"]["expected"]
    sims=sorted({r["sim_seed"] for r in rows})
    both=0
    for s in sims:
        for lane in (0,1):
            assert by["route_13",s,lane]["start"]==by["route_16",s,lane]["start"]
        both+=ok(by["route_13",s,1]) and ok(by["route_16",s,0])
    return [both,-sum(r["outcome"]=="collision" for r in rows),sum(r["outcome"]=="arrival" for r in rows)]

def evaluate(job, *, publish=save):
    arm,seed,step,stage,part=job
    d=OUT/f"{arm}_{seed}"
    out=d/f"{part}_{step}_{stage}.json"
    if out.exists():return
    cases=itertools.product(HOLD if part=="hold" else SELECT,(0,1),m.ROUTES)
    rows=collect_cases(job,cases)
    score(rows)
    publish(out,rows)

def collect_cases(job,cases):
    arm,seed,step,stage,part=job
    d=OUT/f"{arm}_{seed}"
    agent=agent_for(seed);agent.load(str(d/f"policy_{step}.pt"));agent.policy.eval()
    h=digest(agent)
    env=environment(stage,arm,seed,OUT/"eval_maps"/str(os.getpid()),audit=True)
    rows=[]
    try:
        for sim,lane,route in cases:
            r=evaluate_episode(env,agent,route,sim,lane)
            r.update(drops=env.c_drop,actions16=env.c_commands,reward_events=env.c_journal,reward_totals=env.c_totals,cycles=env.d_cycles)
            rows.append(r)
        assert digest(agent)==h and agent.num_timesteps==0
        return rows
    finally:env.close()

def main(mode):
    arms=json.loads((OUT/"decision.json").read_text())["arms"]
    if mode=="train":
        with ProcessPoolExecutor(6) as pool:list(pool.map(training,itertools.product(arms,SEEDS)))
        return
    assert mode=="eval"
    jobs=list(itertools.product(arms,SEEDS,STEPS,STAGES,["select"]))
    with ProcessPoolExecutor(4,initializer=initialize) as pool:
        while jobs:
            ready=[j for j in jobs if (OUT/f"{j[0]}_{j[1]}"/f"policy_{j[2]}.pt").exists()]
            if not ready:
                time.sleep(15)
                continue
            ready.sort(key=lambda j:j[2])
            list(pool.map(evaluate,ready))
            jobs=[j for j in jobs if j not in set(ready)]
        hold=[]
        for arm,seed in itertools.product(arms,SEEDS):
            d=OUT/f"{arm}_{seed}"
            scores={s:score(json.loads((d/f"select_{s}_high_real_ratio.json").read_text())) for s in STEPS}
            best=max(STEPS,key=lambda s:(*scores[s],-s))
            save(d/"selection.json",dict(selected=best,scores=scores))
            hold.extend(itertools.product([arm],[seed],sorted({best,STEPS[-1]}),STAGES,["hold"]))
        list(pool.map(evaluate,hold))

if __name__=="__main__":main(sys.argv[1])
