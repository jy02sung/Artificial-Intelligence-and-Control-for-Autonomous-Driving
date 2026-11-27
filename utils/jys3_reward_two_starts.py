"""Matched fork rewards from the Q1/M3 random initialization. Assign one seed per PC."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from utils import jys3_m5ft_ablation as ft

L,P,C,A = ft.L,ft.P,ft.C,ft.A
ROOT = L.ROOT.parent / "jys3_matched_fork_two_starts"
REWARDS = ("base","fork1")
MARKS = tuple(range(50000,300001,50000))


def out(track,reward,s):
    return ROOT/track/reward/f"s{s}"


def train(job):
    track,reward,s,n=job
    assert track=="scratch" and reward=="fork1"
    dest=out(track,reward,s)
    if (dest/f"policy_{n}.pt").is_file():
        return
    def episodes(env,arm,path,*,training_seed):
        ep=L.q.Episodes(env,arm,path,training_seed=training_seed)
        assert env.progress_cap==6. and L.old.driving.PASS_REWARD==2.5
        L.old.driving.PASS_REWARD=1.
        return ep
    expected=L.read(L.q.ROOT/"protocol.json")["initial_sha256"][str(s)]
    P.train("Q1",n,directory=dest,make_agent=lambda:P.agent(s),
            make_episodes=episodes,expected_hash=expected,training_seed=s,update=L.q.update)


def evaluation(job):
    track,reward,s,n,panel,rep=job
    directory=out(track,reward,s)
    source=(L.q.folder("Q1",s) if reward=="base" else directory)
    C.evaluate("Q1",s,n,panel,rep,directory=directory,checkpoint=source/f"policy_{n}.pt",lateral=True,record_first_fork=True)


def panel_path(track,r,s,n,h):
    if r=="base" and h=="screen":
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
    return max(MARKS,key=rank)


def results(track,seeds):
    rows=[]
    table=[]
    effects=[]
    effect_table=[]
    for mode in ("fixed","selected"):
        for s in seeds:
            baseline=None
            for r in REWARDS:
                n=MARKS[-1] if mode=="fixed" else best(track,r,s)
                panels=[L.read(panel_path(track,r,s,n,h))["rows"] for h in L.HOLDS]
                outcomes=C.summary(panels)
                needed = [x for block in panels[1:] for x in block
                          if P.groups()[x["route"],x["lane"]]=="needs_change"]
                assert len(needed)==208 and all(x["first_fork_passed"] is not None for x in needed)
                outcomes["required_first_pass"] = sum(x["first_fork_passed"] for x in needed)
                if r=="base":
                    baseline=panels
                    high=outcomes
                else:
                    # Direction matches the published M0 fork experiment: +2.5 minus +1.
                    effect={k:high[k]-outcomes[k] for k in (*A.OUT,"required_first_pass","required","cutin")}
                    effects.append(dict(mode=mode,seed=s,counts=effect))
                    effect_table.append([mode,s,*[effect[k] for k in (*A.OUT,"required_first_pass","required","cutin")]])
                pairs=[L.old.paired(a,b) for a,b in zip(baseline,panels)]
                shifts={k:sum(d[k] for d in pairs) for k in pairs[0]}
                rows.append(dict(mode=mode,seed=s,reward=r,step=n,**outcomes,**shifts))
                table.append([mode,s,r,n,*[outcomes[k] for k in (*A.OUT,"required_first_pass","required","cutin")],
                              shifts["lost"],shifts["gained"]])
    dest = ROOT/track if len(seeds)>1 else ROOT/track/f"summary_s{seeds[0]}"
    L.save(dest/"comparison.json",rows)
    L.save(dest/"fork_effect.json",effects)
    md=["# "+track+" 출발 보상 비교","","같은 출발점의 base와 비교한 주행 건수. 시드당 576회, 변경 필요 출발 208회."]
    md+=L.md_table(["비교","시드","보상","스텝","완주","충돌","이탈","끝 정지","시간초과","필요 출발 첫 분기 통과","변경 필요 완주","컷인","기존 완주 상실","새 완주"],table)
    md+=["","분기 보상 +2.5에서 +1의 결과를 뺀 건수. 각 주행은 출발 관측 해시까지 짝을 맞췄다."]
    md+=L.md_table(["비교","시드","완주 차이","충돌 차이","이탈 차이","끝 정지 차이","시간초과 차이","필요 출발 첫 분기 통과 차이","변경 필요 완주 차이","컷인 차이"],effect_table)
    tmp=dest/"report.tmp"
    tmp.write_text("\n".join(md)+"\n",encoding="utf8")
    tmp.replace(dest/"report.md")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("track",choices=("scratch",))
    ap.add_argument("--workers",type=int,choices=range(1,5),default=4)
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--seed",type=int,choices=ft.SEEDS)
    ap.add_argument("--summarize",action="store_true")
    args=ap.parse_args()
    t=args.track
    seeds=(args.seed,) if args.seed is not None else ft.SEEDS
    L.torch.set_num_threads(1)
    artifacts={}
    # Common protocol also supports copying the two per-seed runs onto one machine.
    for s in ft.SEEDS:
        assert P.digest(P.agent(s).policy)==L.read(L.q.ROOT/"protocol.json")["initial_sha256"][str(s)]
        for n in {best(t,"base",s),MARKS[-1]}:
            f=L.q.folder("Q1",s)/f"policy_{n}.pt"
            artifacts[f.relative_to(L.ROOT.parent).as_posix()]=ft.sha(f)
    protocol=dict(track=t,checkpoints=artifacts,seeds=ft.SEEDS,steps=MARKS,
                  fork=[2.5,1.],progress=[6,6],arms=REWARDS,
                  baseline="archived Q1",panels=L.old.PANELS)
    if args.check:
        print(json.dumps(protocol,indent=2))
        return
    path=ROOT/t/"protocol.json"
    if path.exists():
        assert L.read(path)==json.loads(json.dumps(protocol))
    L.save(path,protocol)
    if args.summarize:
        results(t,seeds)
        return
    with ProcessPoolExecutor(args.workers) as pool:
        for n in MARKS:
            tasks=[(t,r,s,n) for r in REWARDS for s in seeds]
            list(pool.map(train,[x for x in tasks if x[1]!="base"]))
            evaluate(pool,[(*x,"screen") for x in tasks])
            print("screen",n,flush=True)
        evaluate(pool,[(t,r,s,n,h) for r in REWARDS for s in seeds
                       for n in sorted({best(t,r,s),MARKS[-1]}) for h in L.HOLDS])
    assert all(ft.sha(L.ROOT.parent/f)==digest for f,digest in artifacts.items())
    results(t,seeds)


if __name__=="__main__":
    main()
