"""Pure PPO ladder. python -m utils.jys3_ppo_clean_ladder starts or resumes."""
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from collections import Counter
from utils import jys3_ppo_causal_chain as c

p,q,anti,old,final,torch = c.p,c.q,c.anti,c.old,c.final,c.torch
read,save=c.read,c.save
ROOT=c.ROOT.parent/"jys3_ppo_clean_ladder"
REPORT=c.REPORT.parent/"jys3-ppo-clean-ladder.md"
SEEDS={a:((11,22,33) if a in ("M0","M1","M2","M3") else (22,33)) for a in ("M0","M1","M2","M3","M4","M5","M5-FT")}
HOLDS=c.HOLDS
DESIGN=[
    ("M0","M1","분기 정상 통과와 종점의 보상이 작고 먼 결과가 크게 할인된다","할인율·보상·앞빈 정지 종료를 함께 변경","완주율 상승","arrival"),
    ("M1","M2","가치 추정의 기울기가 정책의 공통 은닉층에도 전달된다","가치 추정용 은닉층 분리","같은 25만 예산에서 완주율 상승","arrival"),
    ("M2","M3","도로 굽힘과 진출 방향의 좌우 부호가 일부 분기에서 다르다","도로 방향 관측 교정(34, 40~49번)","변경 필요 출발 완주율 상승","required"),
    ("M3","M4","경계에서 좌우를 반복해도 비용이 없었다","경계·실선 무효 명령에만 −0.02","무효 명령 감소, 변경 필요 완주 유지","invalid"),
    ("M4","M5","좁은 틈에 실제 진입해 측면 충돌이 발생했다","위험한 실제 변경에만 −1.5","컷인 충돌과 전체 충돌 감소","cutin"),
    ("M5","M5-FT","초기부터 안전 보상을 주면 분기 탐색까지 위축될 수 있다","Q1 정점에서 Critic 워밍업·Self-KL·저학습률 미세조정","완주 증가, 충돌 감소","arrival")]


def folder(a,s):
    return ROOT/f"{a}_s{s}"


def source(a,s):
    if a=="M2":
        return p.folder("P0") if s==11 else c.dest("P0",s)
    if a=="M3":
        return q.folder("Q1",s)
    return folder(a,s)


def marks(a):
    if a=="M5-FT":
        return range(25000,150001,25000)
    return range(50000,250001 if a in ("M0","M1") else 300001,50000)


def panel(a,s,n,h):
    return folder(a,s)/f"eval_{n}_{h}.json"


def choose(a,s,cap=None):
    def rank(n):
        m=read(panel(a,s,n,"screen"))["metrics"]
        return m["arrival"],-m["collision"],m["required_arrived"],-n
    return max((n for n in marks(a) if cap is None or n<=cap),key=rank)


def fresh(a,s):
    if a in ("M0","M1"):
        return old.PPO(50,2,hidden_sizes=(256,256),activation="relu",seed=s,device="cpu",lr=.0001,
                       gamma=.99 if a=="M0" else .995,gae_lambda=.95,n_steps=2048,n_epochs=10,
                       minibatch_size=64,entropy_coef=0.,lane_entropy_coef=.002,lane_keep_bias=1.)
    agent=p.agent(s)
    if a=="M5-FT":
        agent.policy.load_state_dict(torch.load(folder(a,s)/"warm.pt",weights_only=True))
    return agent


class Episodes(c.Episodes):
    def __init__(self,env,arm,path,*,training_seed,warm=False):
        super().__init__(env,arm,path,training_seed=training_seed,warm=warm)
        self.costed=arm in ("M4","M5","M5-FT")
        self.risk=arm in ("M5","M5-FT")
        self.clear_stop_enabled=arm!="M0"
        if self.costed:
            self.transform=p.lateral_transform(env)
        if arm=="M5-FT":
            self.count=200000 if warm else 100000
        if arm=="M0":
            old.driving.ARRIVAL_REWARD,old.driving.PASS_REWARD=5.,1.
            old.driving.COLLISION_REWARD,old.driving.TIMEOUT_REWARD=-5.,-5.
            env.progress_cap,env.off_route_penalty=2.,-5.
        assert old.driving.COLLISION_REWARD==(-5. if arm=="M0" else -12.)


def train(a,s):
    torch.set_num_threads(1)
    f=folder(a,s)
    f.mkdir(exist_ok=True,parents=True)
    if a=="M5-FT" and not (f/"warm.pt").exists():
        anti.warmup(s,arm=a,dest=f,episode_type=Episodes)
    ref=anti.load_peak(s).policy.eval().requires_grad_(False) if a=="M5-FT" else None
    expected=(read(f/"warm_check.json")["sha256"] if ref is not None else
              read(ROOT/"protocol.json")["initial"][str(s)]["shared" if a in ("M0","M1") else "split"])
    def update(agent,e,adv,ret):
        if ref is not None:
            for group in agent.optimizer.param_groups:
                group["lr"]=3e-5-2e-5*max(0,agent.num_timesteps-agent.n_steps)/(150000-agent.n_steps)
            agent.lane_entropy_coef=.0005
        m=c.verified_update(agent,e,adv,ret,ref)
        m.update(e.stats)
        e.stats.clear()
        return m
    for n in marks(a):
        if not (f/f"policy_{n}.pt").exists():
            p.train(a,n,directory=f,make_agent=lambda:fresh(a,s),make_episodes=Episodes,
                    expected_hash=expected,training_seed=s,update=update)


def hashes():
    paths={old.ORIG["R5"]}
    for a in ("M2","M3"):
        for s in SEEDS[a]:
            paths.update(source(a,s).glob("*.pt"))
    return {f.relative_to(ROOT.parent).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)}


def preflight():
    ROOT.mkdir(exist_ok=True,parents=True)
    h=hashes()
    if (ROOT/"protocol.json").exists():
        assert h==read(ROOT/"protocol.json")["protected"]
        return
    init={}
    for s in SEEDS["M0"]:
        raw=fresh("M0",s).policy
        shared=fresh("M1",s).policy
        split=p.agent(s).policy
        assert p.digest(raw)==p.digest(shared)
        for k,v in raw.state_dict().items():
            assert torch.equal(v,split.state_dict()[k])
        assert p.digest(split)==read(source("M2",s)/"initial.json")["sha256"]==read(source("M3",s)/"initial.json")["sha256"]
        init[s]=dict(shared=p.digest(shared),split=p.digest(split))
        for a in ("M2","M3"):
            assert all((source(a,s)/f"policy_{n}.pt").is_file() for n in marks(a))
    assert h[old.ORIG["R5"].relative_to(ROOT.parent).as_posix()]==old.HASH["R5"]
    with final.p.roll.baseline.simulator() as env:
        assert len(old.CROSSINGS)==20 and env.physical_forks and not env.lc_safety_enabled and env.occlusion
        p.prev.Demand(env)
    save(ROOT/"protocol.json",dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         protected=h,initial=init,panels=old.PANELS,workers=4,seeds=SEEDS,design=DESIGN,
         training_seed_formula="205000000+(seed-11)*1000000+episode_index",gamma=[.99,.995],
         selection="arrival, -collision, required_arrived, -step",invalid=-.02,risky_move=-1.5,
         fine_tune=dict(warm=10000,beta_kl=.08,lr=[3e-5,1e-5],lane_entropy=.0005)))


def evaluate(a,s,n,h,rep):
    c.evaluate(a,s,n,h,rep,directory=folder(a,s),checkpoint=source(a,s)/f"policy_{n}.pt",
               lateral=a in ("M3","M4","M5","M5-FT"))


def combine(a,s,n,h):
    rows=[r for rep in range(old.PANELS[h][1]) for r in read(folder(a,s)/"parts"/f"{n}_{h}_{rep}.json")]
    assert [r["seed"] for r in rows]==final.planned(*old.PANELS[h])
    old.paired(read(q.panel("Q1",s,anti.PEAK.get(s,100000),h))["rows"],rows)
    if a in ("M2","M3"):
        archive=source(a,s)/f"eval_{n}_{h}.json"
        if archive.exists():
            other=read(archive)["rows"]
            old.paired(other,rows)
            assert all(all(x[k]==y[k] for k in ("outcome","steps","left_commands","right_commands")) for x,y in zip(other,rows))
    counts=Counter(r["outcome"] for r in rows)
    m={k:counts[k] for k in anti.OUT}
    m["required_arrived"]=sum(r["outcome"]=="arrival" and p.groups()[r["route"],r["lane"]]=="needs_change" for r in rows)
    save(panel(a,s,n,h),dict(metrics=m,rows=rows))


def screens(a,s):
    if a in ("M2","M3"):
        for n in marks(a):
            f=source(a,s)/f"eval_{n}_screen.json"
            data=read(f)
            assert len(data["rows"])==128 and [r["seed"] for r in data["rows"]]==final.planned(*old.PANELS["screen"])
            save(panel(a,s,n,"screen"),data)


def get_result(a,s,n):
    if not all(panel(a,s,n,h).exists() for h in HOLDS):
        return None
    return dict(step=n,**c.summary([read(panel(a,s,n,h))["rows"] for h in HOLDS]))


def stage(pool,runs):
    backlog=[(a,s) for a,s in runs if a not in ("M2","M3")]
    jobs={}
    for a,s in runs:
        screens(a,s)
    seen=set()
    parts={}
    finished=set()
    while len(finished)<len(runs):
        while backlog and len(jobs)<(3 if parts else 4):
            key=backlog.pop(0)
            jobs[pool.submit(train,*key)]=key
        for a,s in runs:
            if (a,s) in finished:
                continue
            keys=[(a,s,n,"screen") for n in marks(a) if (source(a,s)/f"policy_{n}.pt").is_file()]
            if (a,s) not in backlog and (a,s) not in jobs.values() and len(keys)==len(marks(a)) and all(panel(*k).is_file() for k in keys):
                picks={choose(a,s)}
                if a=="M2":
                    picks.add(choose(a,s,250000))
                keys.extend((a,s,n,h) for n in picks for h in HOLDS)
                if all(panel(*k).is_file() for k in keys):
                    finished.add((a,s))
                    report()
                    print("completed",a,s,flush=True)
            for k in keys:
                if k in seen or panel(*k).exists():
                    continue
                seen.add(k)
                for rep in range(old.PANELS[k[-1]][1]):
                    parts[pool.submit(evaluate,*k,rep)]=k
        active=set(jobs)|set(parts)
        if active:
            done,_=wait(active,timeout=6,return_when=FIRST_COMPLETED)
            for fut in done:
                fut.result()
                if fut in jobs:
                    del jobs[fut]
                else:
                    k=parts.pop(fut)
                    if k not in parts.values():
                        combine(*k)
                        print("evaluated",*k,read(panel(*k))["metrics"]["arrival"],flush=True)
                        report()

def md_table(head,rows):
    return ["","| "+" | ".join(head)+" |","|"+"---|"*len(head),*["| "+" | ".join(map(str,row))+" |" for row in rows],""]


def report():
    models={}
    bycap={}
    history=[]
    for a,seeds in SEEDS.items():
        for s in seeds:
            for n in marks(a):
                if panel(a,s,n,"screen").exists():
                    m=read(panel(a,s,n,"screen"))["metrics"]
                    history.append([a,s,n,*[m[k] for k in anti.OUT],m["required_arrived"]])
            if not all(panel(a,s,n,"screen").exists() for n in marks(a)):
                continue
            m=get_result(a,s,choose(a,s))
            if m:
                models[f"{a}/{s}"]=m
            if a=="M2":
                m=get_result(a,s,choose(a,s,250000))
                if m:
                    bycap[str(s)]=m
    save(ROOT/"summary.json",dict(selected=models,m2_matched_250k=bycap))
    docs=["# 순수 PPO 단계별 주행 성능","",
          f"고교통 576회 최종 검증이 끝난 모델은 {len(models)}/18개다. 각 단계의 예상 효과는 학습 전에 기록했다. M0는 단순 보상 기준선이며 이후 단계에서 측정한 하락도 함께 표시한다.",
          "", "## 순수 PPO 인과 사다리","",
          "각 시드의 동일한 16개 경로와 교통 시드 576회를 비교했다. 변경 필요 출발은 지정·추가 256회 패널에 포함된 208회다. 유효/무효는 좌·우 명령 중 경계·실선 기준으로 구분한 건수다. 실제 변경은 SUMO 차로변경 출력, 컷인 충돌은 기존 진단의 3스텝·동적 간격 기준을 따른다."]
    totals=[]
    metrics=[*anti.OUT,"required","valid","invalid","moves","cutin"]
    def cells(row,mean=False):
        result=[]
        for k in metrics:
            v=row[k]
            if k in anti.OUT or k=="required":
                result.append(f"{v:.1f} ({100*v/(208 if k=='required' else 576):.1f}%)" if mean else f"{v} ({100*v/(208 if k=='required' else 576):.1f}%)")
            else:
                result.append(f"{v:.1f}" if mean else str(v))
        return result
    for a,seeds in SEEDS.items():
        items=[]
        for s in seeds:
            m=models.get(f"{a}/{s}")
            if m:
                totals.append([a,s,m["step"],*cells(m)])
                items.append(m)
        if len(items)==len(seeds):
            totals.append([a,"평균","—",*cells({k:sum(m[k] for m in items)/len(items) for k in metrics},True)])
    docs+=md_table(["단계","시드","선정 스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주 /208","유효 명령","무효 명령","실제 변경","컷인 충돌"],totals)
    docs+=["## 발표용 가설과 실제 결과"]
    effects=[]
    for prev,a,problem,change,expect,primary in DESIGN:
        deltas=[]
        for s in SEEDS[a]:
            first=models.get(f"{prev}/{s}")
            last=bycap.get(str(s)) if (prev,a)==("M1","M2") else models.get(f"{a}/{s}")
            if first and last:
                deltas.append({k:last[k]-first[k] for k in metrics})
        if len(deltas)==len(SEEDS[a]):
            diff={k:sum(d[k] for d in deltas)/len(deltas) for k in metrics}
            good=diff[primary]>0
            if a=="M4":
                good=diff["invalid"]<0 and diff["required"]>=0
            if a=="M5":
                good=diff["cutin"]<0 and diff["collision"]<0
            if a=="M5-FT":
                good=good and diff["collision"]<0
            result=(f"완주 {100*diff['arrival']/576:+.2f}%p, 충돌 {100*diff['collision']/576:+.2f}%p, "
                    f"이탈 {100*diff['off_route']/576:+.2f}%p, 변경 필요 완주 {diff['required']:+.1f}건, "
                    f"무효 {diff['invalid']:+.1f}회, 컷인 {diff['cutin']:+.1f}건")
            verdict="평균에서 충족" if good else "평균에서 미충족"
        else:
            result=f"같은 시드 대조 {len(deltas)}/{len(SEEDS[a])} 완료"
            verdict="집계 중"
        effects.append([prev+" → "+a,problem,change,expect,result,verdict])
    docs+=md_table(["대조","문제","변경","예상 효과","실제 변화 · 시드 평균","판정"],effects)
    docs+=["M1→M2는 두 모델 모두 5만~25만 중 동일한 선정 규칙을 적용했다. M2→M3와 M3→M4는 5만~30만 예산을 맞췄다. 모든 대조에서 양쪽에 공통인 시드만 평균에 포함했다.",
           "M0→M1은 보상·할인율·정지 종료를 합친 효과다. M5와 M5-FT의 차이는 출발 정책, 누적 학습량, Critic 워밍업, 학습률, 엔트로피와 Self-KL이 포함된 학습 방식의 차이다. 기존 연구에 사용한 576개 시드를 그대로 유지했으므로 이번 학습·선정과 분리된 반복 평가다."]
    docs+=md_table(["M2 같은 25만 예산","선정 스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주 /208","유효","무효","실제 변경","컷인"],
                   [[s,m["step"],*cells(m)] for s,m in bycap.items()])
    docs+=["## 128회 체크포인트 평가"]+md_table(["단계","시드","스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주 /52"],history)
    docs+=["[실행 코드](../../utils/jys3_ppo_clean_ladder.py) · [검증 집계](../../results/jys3_ppo_clean_ladder/summary.json)"]
    tmp=REPORT.with_suffix(".tmp")
    tmp.write_text("\n".join(docs)+"\n",encoding="utf8")
    tmp.replace(REPORT)


def reuse(a,s):
    f=source(a,s)
    for n in marks(a):
        for h in HOLDS:
            archived=f/f"eval_{n}_{h}.json"
            if a=="M3" and c.panel("Q1",s,n,h).is_file():
                archived=c.panel("Q1",s,n,h)
            if not archived.is_file():
                continue
            j=read(archived)
            rows=j["rows"]
            if not all(all(k in r for k in ("invalid","valid","moves","cutin")) for r in rows):
                continue
            assert [r["seed"] for r in rows]==final.planned(*old.PANELS[h])
            save(panel(a,s,n,h),j)


def audit():
    h=hashes()
    assert h==read(ROOT/"protocol.json")["protected"]
    for a,seeds in SEEDS.items():
        for s in seeds:
            m=get_result(a,s,choose(a,s))
            assert m and sum(m[k] for k in anti.OUT)==576 and m["cutin"]<=m["collision"]
            if a in ("M2","M3"):
                continue
            if a=="M5-FT":
                assert read(folder(a,s)/"warm_check.json")["actor_bits_identical"]
            step=0
            for line in (folder(a,s)/"updates.jsonl").read_text().splitlines():
                u=json.loads(line)
                assert 0<u["step"]-step==u["size"]<=2048
                assert u["metrics"]["minibatches"]==10*((u["size"]+63)//64)
                step=u["step"]
            assert step==max(marks(a))
    report()
    save(ROOT/"finished.json",dict(held_models=18,train_runs=12,reused_runs=6,new_ppo_steps=3000000,
                                   warmup_steps=20000,protected_end=h))


def main():
    torch.set_num_threads(1)
    preflight()
    for a in ("M2","M3"):
        for s in SEEDS[a]:
            reuse(a,s)
            screens(a,s)
    report()
    with ProcessPoolExecutor(4) as pool:
        stage(pool,[("M0",11),("M1",11),("M0",22),("M1",22),("M0",33),("M1",33),
                    *(("M2",s) for s in SEEDS["M2"]),*(("M3",s) for s in SEEDS["M3"])])
        stage(pool,[(a,s) for a in ("M4","M5") for s in SEEDS[a]])
        stage(pool,[("M5-FT",s) for s in SEEDS["M5-FT"]])
    audit()


if __name__=="__main__":
    main()
