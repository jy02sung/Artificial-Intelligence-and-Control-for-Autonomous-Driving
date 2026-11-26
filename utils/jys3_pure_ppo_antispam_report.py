"""Matched episode comparisons for the pure-PPO command penalty study."""
from utils import jys3_pure_ppo_antispam as s


def table(header, rows):
    return ["", "| "+" | ".join(header)+" |", "|"+"|".join(["---"]*len(header))+"|",
            *["| "+" | ".join(map(str,row))+" |" for row in rows], ""]


def percent(value,total):
    return f"{value} ({value/total:.1%})"


def summary(rows):
    m=s.measure(rows)
    required=s.measure([r for r in rows if r["seed"]>=91000000])
    assert required["required"]==208 and len(set(r["seed"] for r in rows))==m["n"]==576
    m.update(change208=required["required_arrived"],commands_per_episode=(m["left"]+m["right"])/576)
    return m


def generate():
    results={}
    comparisons={}
    trends=[]
    for seed in s.SEEDS:
        peak=s.PEAK[seed]
        reference=[r for panel in s.q.HOLDS for r in s.read(s.q.panel("Q1",seed,peak,panel))["rows"]]
        results["Q1",seed]=(peak,summary(reference))
        replays=[s.folder("Q1",seed)/f"eval_{peak}_{panel}.json" for panel in s.q.HOLDS]
        if all(f.exists() for f in replays):
            baseline_rows=[r for f in replays for r in s.read(f)["rows"]]
            results["Q1",seed][1].update(overlap_commands=sum(r["overlap_commands"] for r in baseline_rows),
                                        within_edge_moves=sum(r["within_edge_moves"] for r in baseline_rows))
        for n in range(200000,400001,50000):
            m=s.measure(s.read(s.q.panel("Q1",seed,n,"screen"))["rows"])
            trends.append(("Q1",seed,n,n,m))
        for track in ("S1","S2"):
            folder=s.folder(track,seed)
            for n in s.MARKS[track]:
                f=folder/f"eval_{n}_screen.json"
                if f.exists():
                    trends.append((track,seed,n,n+(peak if track=="S2" else 0),s.read(f)["metrics"]))
            if not (folder/f"eval_{s.MARKS[track][-1]}_screen.json").exists():
                continue
            n=s.select(track,seed)
            if not all((folder/f"eval_{n}_{panel}.json").exists() for panel in s.q.HOLDS):
                continue
            rows=[r for panel in s.q.HOLDS for r in s.read(folder/f"eval_{n}_{panel}.json")["rows"]]
            paired=s.old.paired(reference,rows)
            m=summary(rows)
            m["overlap_commands"]=sum(r["overlap_commands"] for r in rows)
            m["within_edge_moves"]=sum(r["within_edge_moves"] for r in rows)
            results[track,seed]=(n,m)
            changes=s.Counter((a["outcome"],b["outcome"]) for a,b in zip(reference,rows))
            paired["outcomes"]={f"{a}_to_{b}":count for (a,b),count in changes.items()}
            comparisons[f"{track}/{seed}"]=paired
    docs=["# 순수 PPO 차로변경 명령 비용과 겹침 벌점 실험","",
          f"중간 평가 {sum(a!='Q1' for a,*_ in trends)}/30개, 576회 홀드아웃 {len(results)-2}/4개 완료.",
          "","## 같은 576회에서 비교","",
          "차로 명령은 정책 출력이 좌·우였던 스텝 수다. 변경 필요 208회는 기존 Q1과 같은 두 256회 패널에서 집계했다."]
    header=["모델","시드","선정 PPO 스텝","완주","충돌","이탈","끝 정지","시간초과","좌 / 우 명령","명령 / 주행","변경 필요 완주 /208"]
    docs+=table(header,[[a,seed,n,*[percent(m[k],576) for k in s.OUT],f"{m['left']:,} / {m['right']:,}",
                         f"{m['commands_per_episode']:.2f}",percent(m["change208"],208)] for (a,seed),(n,m) in results.items()])
    docs+=["## Q1에서 바뀐 주행",""]
    deltas=[]
    for (a,seed),(n,m) in results.items():
        if a=="Q1":
            continue
        base=results["Q1",seed][1]
        pair=comparisons[f"{a}/{seed}"]
        deltas.append([a,seed,pair["kept"],pair["lost"],pair["gained"],m["collision"]-base["collision"],
                       f"{m['commands_per_episode']-base['commands_per_episode']:+.2f}", f"{base.get('overlap_commands','…')} → {m['overlap_commands']}",f"{base.get('within_edge_moves','…')} → {m['within_edge_moves']}"])
    docs+=table(["조건","시드","기존 완주 유지","상실","새 완주","충돌 증감","주행당 명령 증감","겹친 방향 명령","같은 도로 내 실제 차로변경"],deltas)
    docs+=["## 스텝별 고교통 128회 추이","",
           "누적 PPO 스텝은 S2 시작 모델의 기존 PPO 학습량을 포함한다. S2의 Critic 전용 1만 스텝은 별도로 계산한다. 홀드아웃 선정에는 S1·S2에서 새로 학습한 체크포인트만 사용했다."]
    docs+=table(["모델","시드","해당 런 PPO 스텝","누적 PPO 스텝","완주","충돌","이탈","끝 정지","시간초과","좌 / 우","주행당 명령"],
                [[a,seed,n,total,*[m[k] for k in s.OUT],f"{m['left']:,} / {m['right']:,}",f"{(m['left']+m['right'])/128:.2f}"] for a,seed,n,total,m in trends])
    docs+=["## 20만~35만의 충돌 변화","",
           "동일한 128회 출발에서 해당 구간의 첫 평가, 최다 충돌 평가, 마지막 평가의 충돌 수를 비교했다. S2 시드 33은 25만 스텝부터 이어 학습한다."]
    later=[]
    for seed in s.SEEDS:
        for arm in ("Q1","S1","S2"):
            curve={total:m for a,ss,n,total,m in trends if a==arm and ss==seed and 200000<=total<=350000}
            if arm=="S2":
                curve[s.PEAK[seed]]=s.measure(s.read(s.q.panel("Q1",seed,s.PEAK[seed],"screen"))["rows"])
            if 350000 not in curve:
                continue
            first,last=min(curve),max(curve)
            peak=max(sorted(curve),key=lambda n:curve[n]["collision"])
            later.append([arm,seed,first,curve[first]["collision"],peak,curve[peak]["collision"],curve[last]["collision"],
                          f"{curve[last]['collision']-curve[first]['collision']:+d}"])
    docs+=table(["모델","시드","구간 시작","첫 충돌","최다 시점","최다 충돌","35만 충돌","첫 평가 대비"],later)
    docs+=["## 실행 조건","",
           "- 전 경로 고교통 15,000대/시, 수정 분기, 고가 시야 보정, 도로 단위 부호 교정, 50차원 정책, 안전게이트 OFF를 Q1과 맞췄다. 같은 관측에 포함된 오른쪽 겹침 36번·왼쪽 겹침 37번을 명령 직전 읽었다.",
           "- 유지 이외 명령에 −0.02, 명령 방향이 겹치면 추가 −0.30, 충돌 −16을 적용했다. 행동 실행은 기존 시뮬레이터가 그대로 담당했다. 완주 +10, 분기 +2.5, 전진 상한 +6, 경로 이탈·끝 정지·시간초과 −8, 앞이 빈 저속 −0.02 및 50스텝 연속 −9 종료는 Q1과 같다.",
           "- 모든 PPO 업데이트에서 실제 Adam 스텝 수가 10에포크 전체 미니배치 수와 일치하는지 검사했다. S1은 같은 시드 Q1과 초기 가중치·주행 시드 편성을 맞췄다. S2는 Q1의 Actor와 분리형 Critic 가중치를 복원했고, 새 Adam과 별도 학습용 교통 시드 대역을 사용했다.",
           "- S2 Critic 워밍업은 기존 분리형 Critic 실험과 같은 Adam 2e-5로 1만 스텝 수집 후 각 롤아웃에서 가치 회귀를 수행했다. Actor는 비트 단위 동일성을 검사했다. 평가는 정책의 결정론적 출력만 사용하며 학습용 10초 종료는 껐다.",
           "- 선정 우선순위는 128회의 완주 최대, 충돌 최소, 끝 정지와 시간초과 합 최소, 이른 스텝이다. 각 트랙은 보상과 학습 일정이 함께 바뀐 구성으로 비교한다.",
           "", "[실행 코드](../../utils/jys3_pure_ppo_antispam.py) · [주행 원자료](../../results/jys3_pure_ppo_antispam/)"]
    temp=s.REPORT.with_suffix(".md.tmp")
    temp.write_text("\n".join(docs)+"\n",encoding="utf8")
    temp.replace(s.REPORT)
    s.save(s.ROOT/"summary.json",dict(models={f"{a}/{seed}":dict(selected_step=n,**m) for (a,seed),(n,m) in results.items()},
                                       comparisons=comparisons,screens=sum(a!="Q1" for a,*_ in trends)))
