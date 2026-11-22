"""Measured CPU concurrency and resumable fixed-policy evaluation."""
import itertools,json,os,time,uuid
from concurrent.futures import ProcessPoolExecutor,as_completed
from utils.experiment_d_run import OUT,STAGES,STEPS,SEEDS,SELECT,HOLD,agent_for,collect_cases,environment,evaluate_episode,initialize,m,save,score

def path(job):
    a,s,k,t,p=job
    return OUT/f"{a}_{s}"/f"{p}_{k}_{t}.json"

def warm(args):
    token,count=args
    agent_for(11)
    d=OUT/"speed_trial"/token
    d.mkdir(parents=True,exist_ok=True)
    (d/str(os.getpid())).touch()
    end=time.monotonic()+180
    while len(list(d.iterdir()))<count:
        if time.monotonic()>end:raise TimeoutError("worker startup")
        time.sleep(.1)

def bench(_):
    agent=agent_for(11)
    agent.load(str(OUT/"D1_11"/"policy_1000000.pt"))
    agent.policy.eval()
    steps=[]
    for stage in STAGES:
        env=environment(stage,"D1",11,OUT/"speed_maps"/str(os.getpid()),audit=True)
        try:
            env.max_steps=500  # isolated throughput fixture only
            steps.append(evaluate_episode(env,agent,"route_16",850000,0)["steps"])
        finally:env.close()
    return steps

def choose():
    f=OUT/"worker_benchmark.json"
    if f.exists():return json.loads(f.read_text())["chosen"]
    times={}
    with ProcessPoolExecutor(12,initializer=initialize) as pool:
        list(pool.map(warm,[(uuid.uuid4().hex,12)]*12))
        for n in (4,6,8,10,12,12,10,8,6,4):
            begin=time.monotonic()
            rows=list(pool.map(bench,range(n)))
            dur=time.monotonic()-begin
            assert all(r==rows[0] for r in rows)
            times.setdefault(n,[]).append(dict(steps=sum(map(sum,rows)),seconds=dur))
            print(n,round(sum(map(sum,rows))/dur),flush=True)
    rates={n:sum(r["steps"] for r in rs)/sum(r["seconds"] for r in rs) for n,rs in times.items()}
    chosen=max(rates,key=rates.get)
    save(f,dict(runs=times,steps_per_second=rates,chosen=chosen))
    return chosen

def parts(job):
    return list(itertools.product(HOLD if job[-1]=="hold" else SELECT,(0,1)))

def shard(job,sim,lane):
    f=path(job)
    return OUT/"eval_chunks"/f.parent.name/f.stem/f"{sim}_{lane}.json"

def chunk(args):
    job,sim,lane=args
    f=shard(job,sim,lane)
    if not f.exists():
        f.parent.mkdir(exist_ok=True,parents=True)
        save(f,collect_cases(job,[(sim,lane,r) for r in m.ROUTES]))
    return job

def merge(job):
    f=path(job)
    if f.exists():return True
    files=[shard(job,s,l) for s,l in parts(job)]
    if not all(x.exists() for x in files):return False
    rs=[r for x in files for r in json.loads(x.read_text())]
    assert [(r["sim_seed"],r["lane_case"],r["route"]) for r in rs]==[(s,l,r) for s,l in parts(job) for r in m.ROUTES]
    score(rs)
    save(f,rs)
    return True

def process(pool,jobs):
    tasks=[]
    for job in jobs:
        if merge(job):continue
        tasks.extend(pool.submit(chunk,(job,s,l)) for s,l in parts(job) if not shard(job,s,l).exists())
    for future in as_completed(tasks):merge(future.result())

def main():
    arms=json.loads((OUT/"decision.json").read_text())["arms"]
    assert all(path((a,s,1000000,"","")).with_name("policy_1000000.pt").exists() for a,s in itertools.product(arms,SEEDS))
    job=("D0",11,0,"high_real_ratio","select")
    ref=json.loads(path(job).read_text())[0]
    test=collect_cases(job,[(ref["sim_seed"],ref["lane_case"],ref["route"])])[0]
    assert json.loads(json.dumps(test))==ref
    with ProcessPoolExecutor(choose(),initializer=initialize) as pool:
        high=list(itertools.product(arms,SEEDS,STEPS,["high_real_ratio"],["select"]))
        final=list(itertools.product(arms,SEEDS,[1000000],STAGES,["hold"]))
        process(pool,high+final)
        selected=[]
        for a,s in itertools.product(arms,SEEDS):
            scores={k:score(json.loads(path((a,s,k,"high_real_ratio","select")).read_text())) for k in STEPS}
            best=max(STEPS,key=lambda k:(*scores[k],-k))
            save(OUT/f"{a}_{s}"/"selection.json",dict(selected=best,scores=scores))
            selected.extend(itertools.product([a],[s],[best],STAGES,["hold"]))
        rest=list(itertools.product(arms,SEEDS,STEPS,[t for t in STAGES if t!="high_real_ratio"],["select"]))
        process(pool,selected+rest)

if __name__=="__main__":
    initialize()
    main()
