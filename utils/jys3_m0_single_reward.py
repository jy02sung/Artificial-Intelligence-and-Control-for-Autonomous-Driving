"""Teammate-only reward controls. One condition per invocation; the main ladder is independent."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import jys3_ppo_clean_ladder as ladder

c,p,old=ladder.c,ladder.p,ladder.old
read,save=ladder.read,ladder.save
ROOT=ladder.ROOT.parent/"jys3_m0_single_reward"
BASE=dict(ARRIVAL_REWARD=5.,PASS_REWARD=1.,COLLISION_REWARD=-5.,TIMEOUT_REWARD=-5.)
CHANGES={"arrival":("ARRIVAL_REWARD",10.),"fork":("PASS_REWARD",2.5),"collision":("COLLISION_REWARD",-12.)}
NAMES={"arrival":"완주 보너스","fork":"올바른 분기 통과","collision":"실제 충돌"}
SEEDS=(11,22,33)
MARKS=tuple(ladder.marks("M0"))


def reward_config(arm):
    key,value=CHANGES[arm]
    values={**BASE,key:value}
    assert [k for k in BASE if BASE[k]!=values[k]]==[key]
    return values


def folder(arm,seed):
    return ROOT/arm/f"s{seed}"


class Episodes(ladder.Episodes):
    def __init__(self,env,arm,path,*,training_seed):
        super().__init__(env,"M0",path,training_seed=training_seed)
        assert not self.costed and not self.risk and not self.clear_stop_enabled
        assert env.progress_cap==2. and env.off_route_penalty==-5.
        assert all(getattr(old.driving,k)==v for k,v in BASE.items())
        key,value=CHANGES[arm]
        setattr(old.driving,key,value)


def train(arm,seed,n):
    out=folder(arm,seed)
    if (out/f"policy_{n}.pt").is_file():
        return
    sha=read(ladder.ROOT/"protocol.json")["initial"][str(seed)]["shared"]
    p.train(arm,n,directory=out,make_agent=lambda:ladder.fresh("M0",seed),make_episodes=Episodes,
            expected_hash=sha,training_seed=seed,update=c.verified_update)


def evaluate(arm,seed,n,name,rep):
    c.evaluate(arm,seed,n,name,rep,directory=folder(arm,seed),lateral=False)


def collect(pool,arm,checks,name):
    jobs={}
    for seed,n in checks:
        out=folder(arm,seed)
        if (out/f"eval_{n}_{name}.json").is_file():
            continue
        for rep in range(old.PANELS[name][1]):
            jobs[pool.submit(evaluate,arm,seed,n,name,rep)]=seed,n
    for job in as_completed(jobs):
        job.result()
    for seed,n in checks:
        f=folder(arm,seed)
        rows=[r for rep in range(old.PANELS[name][1]) for r in read(f/"parts"/f"{n}_{name}_{rep}.json")]
        assert [r["seed"] for r in rows]==ladder.final.planned(*old.PANELS[name])
        m=Counter(r["outcome"] for r in rows)
        metrics={k:m[k] for k in ladder.anti.OUT}
        metrics["required_arrived"]=sum(r["outcome"]=="arrival" and p.groups()[r["route"],r["lane"]]=="needs_change" for r in rows)
        save(f/f"eval_{n}_{name}.json",dict(metrics=metrics,rows=rows))


def pick(folder_path):
    def rank(n):
        m=read(folder_path/f"eval_{n}_screen.json")["metrics"]
        return m["arrival"],-m["collision"],m["required_arrived"],-n
    return max(MARKS,key=rank)


def baseline(seed,root):
    f=root/f"M0_s{seed}"
    if all((f/f"eval_{n}_screen.json").is_file() for n in MARKS):
        n=pick(f)
        if all((f/f"eval_{n}_{h}.json").is_file() for h in c.HOLDS):
            return n,[read(f/f"eval_{n}_{h}.json")["rows"] for h in c.HOLDS]
    return None


def summarize(arm,baseline_root):
    seeds={}
    comparisons=[]
    transition_rows=[]
    totals=[]
    for seed in SEEDS:
        f=folder(arm,seed)
        n=pick(f)
        blocks=[read(f/f"eval_{n}_{h}.json")["rows"] for h in c.HOLDS]
        result=c.summary(blocks)
        row=dict(step=n,result=result)
        seeds[seed]=row
        totals.append([seed,n,*[f"{result[k]} ({result[k]/576:.1%})" for k in ladder.anti.OUT],
                       result["required"],result["invalid"],result["moves"],result["cutin"]])
        reference=baseline(seed,baseline_root)
        if reference:
            n0,original=reference
            m0=c.summary(original)
            pairs=Counter()
            for a,b in zip(original,blocks):
                old.paired(a,b)
                pairs.update((x["outcome"],y["outcome"]) for x,y in zip(a,b))
            assert sum(pairs.values())==576
            row.update(m0_step=n0,m0=m0,transitions={x:{y:pairs[x,y] for y in ladder.anti.OUT} for x in ladder.anti.OUT})
            comparisons.append([seed,n0,n,*[f"{100*(result[k]-m0[k])/576:+.2f}" for k in ladder.anti.OUT]])
            transition_rows.extend([seed,x,*[pairs[x,y] for y in ladder.anti.OUT]] for x in ladder.anti.OUT)
    for k in ("result","m0"):
        if all(k in r for r in seeds.values()):
            mean={key:sum(r[k][key] for r in seeds.values())/3 for key in seeds[11][k]}
            totals.append(["3시드 평균" if k=="result" else "M0 3시드 평균","—",
                           *[f"{mean[t]:.1f} ({mean[t]/576:.1%})" for t in ladder.anti.OUT],
                           *[f"{mean[t]:.1f}" for t in ("required","invalid","moves","cutin")]])
    complete=len(comparisons)==3
    save(ROOT/arm/"comparison.json",dict(arm=arm,config=reward_config(arm),seeds=seeds,matched_all=complete))
    cols=["완주","충돌","이탈","끝 정지","일반 시간초과"]
    md=[f"# M0 {NAMES[arm]} 단독 대조","",f"초기 가중치와 교통 시드를 짝지어 시드별 576회를 평가했다. M0 짝비교 완료 {len(comparisons)}/3."]
    md+=ladder.md_table(["시드","선정 스텝",*cols,"변경 필요 완주 /208","무효 명령","실제 변경","컷인 충돌"],totals)
    md+=ladder.md_table(["시드","M0 선정","후보 선정",*[k+" 변화 (%p)" for k in cols]],comparisons)
    md+=ladder.md_table(["시드","M0의 종료",*cols],transition_rows)
    md+=["중간 128회는 선정용이며 세 홀드아웃의 종료를 합쳐 576회를 구성한다."]
    if not complete:
        md+=["M0 선정 패널과 홀드아웃 원자료가 들어오면 --summarize로 짝비교를 갱신한다."]
    target=ROOT/arm/"report.md"
    tmp=target.with_suffix(".tmp")
    tmp.write_text("\n".join(md)+"\n",encoding="utf8")
    tmp.replace(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=CHANGES)
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    parser.add_argument("--baseline-root",type=Path,default=ladder.ROOT)
    parser.add_argument("--summarize",action="store_true")
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    ladder.torch.set_num_threads(1)
    arm=args.arm
    protocol=read(ladder.ROOT/"protocol.json")
    for s in SEEDS:
        assert p.digest(ladder.fresh("M0",s).policy)==protocol["initial"][str(s)]["shared"]
    config=dict(arm=arm,rewards=reward_config(arm),gamma=.99,progress=2.,off_route=-5.,
                clear_stop=False,shared_critic=True,seed=SEEDS,steps=250000,marks=MARKS,panels=old.PANELS)
    if args.check:
        print(json.dumps(config))
        return
    if not args.summarize:
        save(ROOT/arm/"protocol.json",config)
        with ProcessPoolExecutor(args.workers) as pool:
            for n in MARKS:
                for job in as_completed([pool.submit(train,arm,s,n) for s in SEEDS]):
                    job.result()
                collect(pool,arm,[(s,n) for s in SEEDS],"screen")
            checks=[(s,pick(folder(arm,s))) for s in SEEDS]
            for h in c.HOLDS:
                collect(pool,arm,checks,h)
    summarize(arm,args.baseline_root)


if __name__=="__main__":
    main()
