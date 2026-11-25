"""Report the final, frozen-checkpoint last-fork pilot with its paired denominators."""
import json
import hashlib
from collections import Counter
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
DATA = HOME / "results/jys3_fork1314_pilot"
DOC = HOME / "docs/experiments/jys3-fork1314-pilot.md"
SEEDS = (11, 22, 33)
PANELS = {"screen":(0,16384,32768,50000), "holdout":(0,50000)}


def read(p):
    return json.loads(p.read_bytes())


def correct(r):
    return r["actual"] == r["expected"]


def metrics(rows):
    pairs = list(zip(rows[::2],rows[1::2]))
    assert all(a["route"]==13 and b["route"]==14 and a["start"]==b["start"]
               and a["traffic_seed"]==b["traffic_seed"] for a,b in pairs)
    outcomes=Counter(r["outcome"] for r in rows)
    return dict(episodes=len(rows), pairs=len(pairs), both=sum(correct(a) and correct(b) for a,b in pairs),
                both_arrived=sum(a["outcome"]==b["outcome"]=="arrival" for a,b in pairs),
                correct=sum(map(correct,rows)), **{k:outcomes[k] for k in ("arrival","collision","off_route","stall","timeout")},
                after_cross_collision=sum(correct(r) and r["outcome"]=="collision" for r in rows))


def subset(rows):
    return dict(n=len(rows), correct=sum(map(correct,rows)), arrival=sum(r["outcome"]=="arrival" for r in rows),
                collision=sum(r["outcome"]=="collision" for r in rows),
                off=sum(r["outcome"]=="off_route" for r in rows),
                missing=sum(r["actual"] is None for r in rows),left=sum(r["left"] for r in rows),
                right=sum(r["right"] for r in rows),changes=sum(r["changes"] for r in rows),
                actual=dict(Counter(str(r["actual"]) for r in rows)))


def frac(k,n):
    return f"{k}/{n} ({100*k/n:.1f}%)"


def main():
    panels={}
    raw=[]
    hashes={}
    learning={}
    for seed in SEEDS:
        f=DATA/f"seed{seed}"
        hashes[str(seed)]={str(mark):hashlib.sha256((f/f"policy_{mark}.pt").read_bytes()).hexdigest()
                           for mark in PANELS["screen"]}
        log=[json.loads(line) for line in (f/"updates.jsonl").read_text().splitlines()]
        assert log[-1]["step"]==50000 and len(log)==25
        episodes=[json.loads(line) for line in (f/"episodes.jsonl").read_text().splitlines()]
        end_step=0
        for r in episodes:
            end_step+=r["steps"]
            r["end_step"]=end_step
        assert end_step<=50000
        populated=0
        last_cut=0
        for update in log:
            populated+=any(last_cut<r["end_step"]<=update["step"] and r["change_needed"]
                           and r["outcome"]=="arrival" for r in episodes)
            last_cut=update["step"]
        needed=[r for r in episodes if r["change_needed"]]
        learning[seed]={"initial":read(f/"initial.json"),"updates":len(log),"last":log[-1],
                        "episodes":len(episodes),"arrivals":sum(r["outcome"]=="arrival" for r in episodes),
                        "change_needed":subset(needed),"route13_lane2":subset([r for r in episodes if r["route"]==13 and r["lane"]==2]),
                        "updates_with_change_needed_arrival":populated,
                        "short_edges_verified":sum(r["actual"] is not None and
                            r.get("first_sampled_external",r["actual"])!=r["actual"] for r in episodes)}
        learning[seed]["lane1_start"]={}
        for mark in (0,50000):
            obs=read(f/f"start_prob_{mark}.json")
            learning[seed]["lane1_start"][mark]={}
            for route in (13,14):
                lane1=[r for r in obs if r["route"]==route and r["lane"]==1]
                assert len(lane1)==8
                learning[seed]["lane1_start"][mark][route]={action:sum(r[action] for r in lane1)/len(lane1)
                    for action in ("left","keep","right")}
    for name,marks in PANELS.items():
        for mark in marks:
            rows=[]
            summaries={}
            for seed in SEEDS:
                f=DATA/f"seed{seed}"
                part=read(f/f"{name}_{mark}.json")
                assert len(part)==(18 if name=="screen" else 48)
                assert all(r["actual"] in (None,"169728835","797789364") for r in part)
                if mark:
                    base=read(f/f"{name}_0.json")
                    assert [(r["route"],r["traffic_seed"],r["start"]) for r in base]==[(r["route"],r["traffic_seed"],r["start"]) for r in part]
                summaries[str(seed)]=metrics(part)
                rows.extend(part)
                raw.extend(dict(panel=name,step=mark,model_seed=seed,**r) for r in part)
            all_stats=metrics(rows)
            by_lane={str(lane):metrics([r for r in rows if r["lane"]==lane]) for lane in range(3)}
            by_route_lane={f"{route}_{lane}":subset([r for r in rows if (r["route"],r["lane"])==(route,lane)])
                           for route in (13,14) for lane in range(3)}
            by_change={str(needed):subset([r for r in rows if r["change_needed"]==needed]) for needed in (False,True)}
            panels[f"{name}_{mark}"]=dict(all=all_stats,by_model=summaries,by_lane=by_lane,by_route_lane=by_route_lane,by_change=by_change)
    (DATA/"metrics.json").write_text(json.dumps(dict(panels=panels,policy_sha256=hashes,learning=learning),indent=2),encoding="utf8")
    (DATA/"evaluation.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw),encoding="utf8")
    lines=["## 구간 주행 결과","",
           "| 평가 | 스텝 | 두 경로 모두 지정 도로 진입 | 둘 다 완주 | 완주 | 충돌 | 실제 오진입 | 정지·시간초과 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,result in panels.items():
        name,mark=key.split("_")
        m=result["all"];n=m["episodes"];p=m["pairs"]
        cells=[frac(m["both"],p),frac(m["both_arrived"],p),frac(m["arrival"],n),frac(m["collision"],n),
               frac(m["off_route"],n),frac(m["stall"]+m["timeout"],n)]
        lines.append(f"| {'중간' if name=='screen' else '별도 검증'} | {int(mark):,} | "+" | ".join(cells)+" |")
    lines+=["","같은 시나리오를 각 학습 시드 모델에 반복했다. 별도 검증의 72쌍은 24개 고유 출발 조건 × 모델 3개다. 각 짝의 초기 배경차 상태, AV 위치·속도·차로와 첫 분기 지시의 반대 부호를 대조했다.","",
            "### 5만 스텝 별도 검증","",
            "| 학습 시드 | 두 경로 모두 지정 도로 진입 | 완주 | 충돌 | 실제 오진입 | 정지·시간초과 |","|---:|---:|---:|---:|---:|---:|"]
    final=panels["holdout_50000"]
    for s,m in final["by_model"].items():
        n=m["episodes"]
        lines.append(f"| {s} | {frac(m['both'],m['pairs'])} | {frac(m['arrival'],n)} | {frac(m['collision'],n)} | {frac(m['off_route'],n)} | {frac(m['stall']+m['timeout'],n)} |")
    lines+=["","| 경로 | 출발 차로 | 차로변경 필요 | 지정 도로 진입 | 다른 도로 진입 | 다음 외부 도로 관측 전 종료 | 완주 |",
            "|---:|---:|---|---:|---:|---:|---:|"]
    for key,m in final["by_route_lane"].items():
        route,lane=map(int,key.split("_"))
        needs=(route==13 and lane==2) or (route==14 and lane<2)
        lines.append(f"| {route} | {lane} | {'예' if needs else '아니오'} | {m['correct']}/{m['n']} | {m['n']-m['correct']-m['missing']} | {m['missing']} | {m['arrival']}/{m['n']} |")
    lines+=["","### 학습 중 확률적으로 행동을 선택했을 때","",
            "| 학습 시드 | 종료된 주행 | 완주 | 변경 필요 출발 완주 | 변경 필요 출발 완주가 포함된 PPO 업데이트 |",
            "|---:|---:|---:|---:|---:|"]
    for seed,item in learning.items():
        needed=item["change_needed"]
        lines.append(f"| {seed} | {item['episodes']} | {item['arrivals']} | {frac(needed['arrival'],needed['n'])} | {item['updates_with_change_needed_arrival']}/{item['updates']} |")
    missed=final["by_route_lane"]["13_2"]
    train13=[r["route13_lane2"] for r in learning.values()]
    n=sum(r["n"] for r in train13)
    arrived=sum(r["arrival"] for r in train13)
    lines+=["",f"13번 2차로에서는 학습 중 확률적으로 행동을 고른 전체 기간에 {arrived}/{n}회 완주했다. 5만 스텝 고정 정책의 별도 검증에서는 오른쪽 명령 {missed['right']}회, 실제 변경 {missed['changes']}회였고, {missed['off']}회가 14번 쪽 도로로 이탈했다. 이번 단독 시험에는 전체 경로 출발로 학습한 대조군이 없다."]
    lines+=["","### 동일한 1차로 출발 상태에서 왼쪽 명령 확률","",
            "각 경로의 관측을 짝지어 확인했다. 교통과 차량 상태가 같고 서로 다른 입력값은 도로 단위 다음 분기 방향(34번 인덱스) 하나였다. 아래 수치는 주행 전반의 빈도가 아니라 출발 순간 8개 시나리오의 평균이다. 1차로에서 13번은 유지해도 통과하며 14번은 왼쪽 2차로로 옮겨야 한다.","",
            "| 학습 시드 | 0스텝 13번 / 14번 | 5만 스텝 13번 / 14번 |","|---:|---:|---:|"]
    for seed,item in learning.items():
        pr=item["lane1_start"]
        pairs=[f"{pr[k][13]['left']:.1%} / {pr[k][14]['left']:.1%}" for k in (0,50000)]
        lines.append(f"| {seed} | {pairs[0]} | {pairs[1]} |")
    lines+=["","고정 정책 평가는 차로 명령 세 가지 중 가장 확률이 높은 것을 선택한다. 학습 중에는 세 명령을 확률에 따라 샘플링한다. 14번이 지나는 5.37m 연결 도로는 두 관측 사이에 통과할 수 있어 SUMO가 실제 실행한 경로와 도로 인덱스로 첫 분기를 검증했다. 초기 기록 검사에서 이 경우가 발견되어 세 시드를 같은 초기 가중치로 다시 실행했다.","",
            "[수치 및 정책 SHA-256](../../results/jys3_fork1314_pilot/metrics.json) · [출발 상태와 실제 도로 및 명령 기록](../../results/jys3_fork1314_pilot/evaluation.jsonl)"]
    doc=DOC.read_text(encoding="utf-8-sig").split("\n## 구간 주행 결과")[0].rstrip()
    DOC.write_text(doc+"\n\n"+"\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines[:10]))


if __name__=="__main__":
    main()
