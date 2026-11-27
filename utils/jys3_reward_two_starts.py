"""Two-machine reward comparison: selected M5-FT continuation or Q1/M3 from scratch."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from utils import jys3_m5ft_ablation as ft

L,P,C,A = ft.L,ft.P,ft.C,ft.A
ROOT = L.ROOT.parent / "jys3_reward_two_starts"
REWARDS = ("base","fork","progress")
PEAK = {22:75000,33:125000}


def marks(track):
    return (25000,50000,75000) if track=="ft" else tuple(range(50000,300001,50000))


def out(track,reward,s):
    return ROOT/track/reward/f"s{s}"


def start_model(s):
    agent=P.agent(s)
    src=ft.original(s)/f"policy_{PEAK[s]}.pt"
    ck=L.torch.load(src,map_location="cpu",weights_only=True)
    assert ck["step"]==PEAK[s]
    weights=dict(ck["policy"])
    weights.update({"value_trunk."+k:v for k,v in ck["value_trunk"].items()})
    agent.policy.load_state_dict(weights,strict=True)
    return agent


def train(job):
    track,reward,s,n=job
    dest=out(track,reward,s)
    if (dest/f"policy_{n}.pt").is_file():
        return
    def new():
        return start_model(s) if track=="ft" else P.agent(s)
    def episodes(env,arm,path,*,training_seed):
        if track=="ft":
            ep=L.Episodes(env,"M5-FT",path,training_seed=s)
            ep.count=300000  # separate from prior warm-up and fine-tuning episode streams
        else:
            ep=L.q.Episodes(env,"Q1",path,training_seed=s)
        assert env.progress_cap==6. and L.old.driving.PASS_REWARD==2.5
        if reward=="fork":
            L.old.driving.PASS_REWARD=3.5
        if reward=="progress":
            env.progress_cap=8.
        return ep
    reference=A.load_peak(s).policy.eval().requires_grad_(False) if track=="ft" else None
    def update(agent,ep,adv,ret):
        if track=="scratch":
            return L.q.update(agent,ep,adv,ret)
        for group in agent.optimizer.param_groups:
            group["lr"]=1e-5
        agent.lane_entropy_coef=.0005
        m=C.verified_update(agent,ep,adv,ret,reference)
        m.update(ep.stats)
        ep.stats.clear()
        return m
    expected=P.digest(new().policy) if track=="ft" else L.read(L.q.ROOT/"protocol.json")["initial_sha256"][str(s)]
    P.train("M5-FT" if track=="ft" else "Q1",n,directory=dest,make_agent=new,
            make_episodes=episodes,expected_hash=expected,training_seed=s,update=update)


def evaluation(job):
    track,reward,s,n,panel,rep=job
    directory=out(track,reward,s)
    source=(L.q.folder("Q1",s) if track=="scratch" and reward=="base" else directory)
    C.evaluate("Q1",s,n,panel,rep,directory=directory,checkpoint=source/f"policy_{n}.pt",lateral=True)


def panel_path(track,r,s,n,h):
    if track=="scratch" and r=="base" and h=="screen":
        return L.q.panel("Q1",s,n,h)
    return out(track,r,s)/f"eval_{n}_{h}.json"


def evaluate(pool,keys):
    pending=[key for key in dict.fromkeys(keys) if not panel_path(*key).is_file()]
    jobs=[(*key,i) for key in pending for i in range(L.old.PANELS[key[-1]][1])]
    list(pool.map(evaluation,jobs))
    for track,r,s,n,h in pending:
        rows=[row for i in range(L.old.PANELS[h][1]) for row in
              L.read(out(track,r,s)/"parts"/f"{n}_{h}_{i}.json")]
        assert [x["seed"] for x in rows]==L.final.planned(*L.old.PANELS[h])
        L.old.paired(L.read(L.q.panel("Q1",s,A.PEAK[s],h))["rows"],rows)
        m={k:sum(x["outcome"]==k for x in rows) for k in A.OUT}
        m["required_arrived"]=sum(x["outcome"]=="arrival" and P.groups()[x["route"],x["lane"]]=="needs_change" for x in rows)
        L.save(panel_path(track,r,s,n,h),dict(metrics=m,rows=rows))


def best(track,r,s):
    def rank(n):
        m=L.read(panel_path(track,r,s,n,"screen"))["metrics"]
        return m["arrival"],-m["collision"],m["required_arrived"],-n
    return max(marks(track),key=rank)


def results(track):
    rows=[]
    table=[]
    for mode in ("fixed","selected"):
        for s in ft.SEEDS:
            baseline=None
            for r in REWARDS:
                n=marks(track)[-1] if mode=="fixed" else best(track,r,s)
                panels=[L.read(panel_path(track,r,s,n,h))["rows"] for h in L.HOLDS]
                if r=="base":
                    baseline=panels
                outcomes=C.summary(panels)
                pairs=[L.old.paired(a,b) for a,b in zip(baseline,panels)]
                shifts={k:sum(d[k] for d in pairs) for k in pairs[0]}
                rows.append(dict(mode=mode,seed=s,reward=r,step=n,**outcomes,**shifts))
                table.append([mode,s,r,n,*[outcomes[k] for k in (*A.OUT,"required","cutin")],
                              shifts["lost"],shifts["gained"]])
    L.save(ROOT/track/"comparison.json",rows)
    md=["# "+track+" 출발 보상 비교","","같은 출발점의 base와 비교한 주행 건수. 시드당 576회, 변경 필요 출발 208회."]
    md+=L.md_table(["비교","시드","보상","스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주","컷인","기존 완주 상실","새 완주"],table)
    tmp=ROOT/track/"report.tmp"
    tmp.write_text("\n".join(md)+"\n",encoding="utf8")
    tmp.replace(ROOT/track/"report.md")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("track",choices=("ft","scratch"))
    ap.add_argument("--workers",type=int,choices=range(1,5),default=4)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    t=args.track
    L.torch.set_num_threads(1)
    artifacts={}
    for s in ft.SEEDS:
        if t=="scratch":
            assert P.digest(P.agent(s).policy)==L.read(L.q.ROOT/"protocol.json")["initial_sha256"][str(s)]
            files=[L.q.folder("Q1",s)/f"policy_{n}.pt" for n in {best(t,"base",s),marks(t)[-1]}]
        else:
            start_model(s)
            files=[ft.original(s)/f"policy_{PEAK[s]}.pt",A.checkpoint(s)]
        for f in files:
            artifacts[f.relative_to(L.ROOT.parent).as_posix()]=ft.sha(f)
    protocol=dict(track=t,checkpoints=artifacts,seeds=ft.SEEDS,steps=marks(t),
                  fork=[2.5,3.5,2.5],progress=[6,6,8],arms=REWARDS,
                  baseline="archived Q1" if t=="scratch" else "matched continuation",panels=L.old.PANELS)
    if args.check:
        print(json.dumps(protocol,indent=2))
        return
    path=ROOT/t/"protocol.json"
    if path.exists():
        assert L.read(path)==json.loads(json.dumps(protocol))
    L.save(path,protocol)
    with ProcessPoolExecutor(args.workers) as pool:
        for n in marks(t):
            tasks=[(t,r,s,n) for r in REWARDS for s in ft.SEEDS]
            list(pool.map(train,[x for x in tasks if not(t=="scratch" and x[1]=="base")]))
            evaluate(pool,[(*x,"screen") for x in tasks])
            print("screen",n,flush=True)
        evaluate(pool,[(t,r,s,n,h) for r in REWARDS for s in ft.SEEDS
                       for n in sorted({best(t,r,s),marks(t)[-1]}) for h in L.HOLDS])
    assert all(ft.sha(L.ROOT.parent/f)==digest for f,digest in artifacts.items())
    results(t)


if __name__=="__main__":
    main()
