"""Regenerate the Q sweep report from paired episodes and learner updates."""
from collections import Counter
import json
from statistics import mean, median, stdev

from utils import jys3_pure_ppo_q_sweep as q

p,old,read,save=q.p,q.old,q.read,q.save
OUT=("arrival","collision","off_route","stall","timeout")
LABELS=("완주","충돌","이탈","끝 정지","시간초과")


def table(head,rows):
    return ["","| "+" | ".join(head)+" |","|"+"|".join("---" for _ in head)+"|",
            *["| "+" | ".join(map(str,row))+" |" for row in rows],""]


def read_hold(arm,seed,n):
    if arm=="P0":
        return [r for k in q.HOLDS for r in read(p.panel_path(arm,n,k))["rows"]]
    if arm=="R5" and not all(q.panel(arm,seed,n,k).exists() for k in q.HOLDS):
        return [r for k in q.HOLDS for r in read(old.data_path("R5",0,k))]
    return [r for k in q.HOLDS for r in read(q.panel(arm,seed,n,k))["rows"]]


def describe(rows):
    assert len({r["seed"] for r in rows})==len(rows)==576
    rates={k:sum(r["outcome"]==k for r in rows) for k in OUT}
    assert sum(rates.values())==576
    long=[r for r in rows if r["seed"]>=91000000 and p.groups()[r["route"],r["lane"]]=="needs_change"]
    assert len(long)==208
    rates.update(required_arrived=sum(r["outcome"]=="arrival" for r in long),
                 slow10=sum(r["longest_slow"]>=50 for r in rows))
    for label,group in [("all",rows),("required",long)]:
        if "left_commands" in group[0]:
            l=sum(r["left_commands"] for r in group); h=sum(r["right_commands"] for r in group)
            rates[label]=dict(left=l,right=h,steps=sum(r["steps"] for r in group))
    rates["routes"]={str(i):dict(Counter(r["outcome"] for r in rows if r["route"]==i)) for i in range(1,17)}
    return rates


def training(arm,seed,n):
    f=q.folder(arm,seed)
    def logged(name):
        return [v for line in (f/name).read_text().splitlines() if (v:=json.loads(line))["step"]<=n]
    updates=logged("updates.jsonl")
    # Use full-size rollouts for the variance comparison, excluding shorter checkpoint boundaries.
    full=[u["metrics"]["route_steps"] for u in updates if u["size"]==(4096 if arm=="Q3" else 2048)]
    data=dict(updates=len(updates),kl_stops=sum(u["metrics"].get("kl_stop") is not None for u in updates),
              route_coverage=mean(sum(n>0 for n in c) for c in full),
              route_share_sd=mean(float(q.np.std(q.np.asarray(c)/sum(c))) for c in full),
              last_lr=updates[-1]["metrics"]["lr"])
    if arm=="Q5":
        eps=logged("episodes.jsonl")
        vals=[u["predictor"]["online_validation"] for u in updates if u["predictor"]["online_validation"]]
        pairs=sum(v["positive"]*(v["n"]-v["positive"]) for v in vals)
        data.update(accuracy=sum(v["correct"] for v in vals)/sum(v["n"] for v in vals),
                    auc=sum(v["auc"]*v["positive"]*(v["n"]-v["positive"]) for v in vals if v["auc"] is not None)/pairs if pairs else None,
                    open_next=sum(u["predictor"]["next_gate_open"] for u in updates),
                    shaped_steps=updates[-1]["predictor"]["applied_steps"],
                    r_exp=sum(e["r_exp"] for e in eps),episodes=len(eps))
    return data


def generate():
    winner_file=q.ROOT/"winner.json"
    winner=read(winner_file)["winner"] if winner_file.exists() else None
    runs=[(a,11) for a in q.ARMS]+([(winner,s) for s in q.SEEDS[1:]] if winner else [])
    screens=[]; rows_by={}; data={}; picks={}
    for model,seed,n in [("P0",11,250000),("R5",0,0)]:
        rows=read_hold(model,seed,n)
        rows_by[model]=rows
        data[model]=describe(rows)
    for a,s in runs:
        name=f"{a}/{s}"
        scored={}
        for n in q.MARKS:
            path=q.panel(a,s,n,"screen")
            if path.exists():
                m=read(path)["metrics"]
                scored[n]=m
                screens.append([a,s,n,*[m[k] for k in OUT],m["required_arrived"]])
        if len(scored)!=len(q.MARKS):continue
        n=q.best(a,s)
        picks[name]={"step":n,"screen":scored[n],"training":training(a,s,n)}
        if all(q.panel(a,s,n,k).exists() for k in q.HOLDS):
            rows=read_hold(a,s,n)
            old.paired(rows_by["P0"],rows)
            rows_by[name]=rows
            data[name]=describe(rows)
    progress=len(data)-2
    doc=["# 순수 PPO Q1~Q5와 세 학습 시드 비교","",
         f"고교통 중간 평가 {len(screens)}/56개, Q 선정 모델의 576회 홀드아웃 {progress}/7개 완료.",
         "", "## 같은 576개 출발에서 비교","",
         "학습과 선정에 사용하지 않은 16개 경로 × 36회를 합쳤다. 각 셀은 건수와 백분율이다. R5는 제출 후보인 보강형 DAgger다."]
    keys=list(data)
    def ratio(v,n):
        return f"{v} ({v/n:.1%})"
    doc+=table(["모델 / 학습 시드","선정 스텝",*LABELS],[[k,picks[k]["step"] if k in picks else "25만" if k=="P0" else "모방학습",
              *[ratio(data[k][v],576) for v in OUT]] for k in keys])
    comparisons={}
    for x,y in [("P0","Q1/11"),("Q1/11","Q2/11"),("Q2/11","Q3/11"),("Q2/11","Q4/11"),("Q2/11","Q5/11")]:
        if y not in data or x not in data:continue
        changes=old.paired(rows_by[x],rows_by[y])
        changes["delta"]={k:data[y][k]-data[x][k] for k in (*OUT,"required_arrived")}
        changes["transitions"]=dict(Counter(a["outcome"]+"→"+b["outcome"] for a,b in zip(rows_by[x],rows_by[y])))
        comparisons[x+"→"+y]=changes
    doc+=["## 변경별 대조",""]
    doc+=table(["대조","완주 증감","충돌 증감","이탈 증감","기존 완주 유지","상실","새 완주"],
               [[k,*[f"{v['delta'][j]:+d}" for j in OUT[:3]],v["kept"],v["lost"],v["gained"]] for k,v in comparisons.items()])
    if "Q1/11" in data:
        after=data["Q1/11"]
        r16=after["routes"]["16"].get("arrival",0)
        doc+=["",f"Q1의 75% 기준은 576회 중 432회다. 실제 완주는 {after['arrival']}회로 **{'달성' if after['arrival']>=432 else '미달'}**이다. "
              f"16번 완주는 P0의 0/36에서 Q1의 {r16}/36으로 바뀌었다. 16번 첫 분기의 지도 부호는 원래도 일치했다."]
    if "Q4/11" in data:
        after=data["Q4/11"]
        safe=after["collision"]/576<.10 and after["off_route"]/576<.10 and after["stall"]+after["timeout"]==0
        doc+=["",f"Q4의 충돌률 {after['collision']/576:.1%}, 이탈률 {after['off_route']/576:.1%}, 끝 정지·시간초과 {after['stall']+after['timeout']}건. "
              f"두 비율 모두 10% 미만이고 정지·시간초과가 0건인 목표는 **{'달성' if safe else '미달'}**이다."]
    if any(row[2]>=250000 for row in screens):
        doc+=["","### 25만 이후 같은 128회에서의 완주 수",""]
        late=q.MARKS[4:]
        trends=[]
        for arm in ("P0","Q1","Q2"):
            values=[]
            for step in late:
                source=p.panel_path(arm,step,"screen") if arm=="P0" else q.panel(arm,11,step,"screen")
                values.append(read(source)["metrics"]["arrival"] if source.exists() else None)
            trend="평가 중"
            if all(v is not None for v in values):
                trend=f"{values[-1]-values[0]:+d}건"
            trends.append([arm,*["—" if v is None else v for v in values],trend])
        doc+=table(["시드 11","25만","30만","35만","40만","25만→40만"],trends)
    doc+=["## 차로 명령과 변경 필요 출발","",
          "좌·우 비율의 분모는 좌·우 명령 수의 합이다. 변경 필요 208회는 두 256회 패널에서 지도 연결상 첫 분기 전에 차로를 옮겨야 하는 출발이다."]
    def mix(m):
        if not m:return "—"
        l,h=m["left"],m["right"]
        return f"{l:,} / {h:,} ({l/(l+h):.1%} / {h/(l+h):.1%})" if l+h else "0 / 0"
    doc+=table(["모델","전체 좌 / 우","변경 필요 출발 좌 / 우","변경 필요 출발 완주 /208"],
               [[k,mix(d.get("all")),mix(d.get("required")),ratio(d["required_arrived"],208)] for k,d in data.items()])
    doc+=["## 경로별 완주와 실패","", "각 칸은 완주 / 충돌 / 이탈 / 끝 정지 / 일반 시간초과 건수다. 경로별 분모는 36회다."]
    doc+=table(["경로",*keys],[[i,*[" / ".join(str(data[k]["routes"][str(i)].get(v,0)) for v in OUT) for k in keys]] for i in [16,5,11]+[i for i in range(1,17) if i not in (16,5,11)]])
    doc+=["### 16번의 출발 차로별 본선 복귀","",
          "0차로는 출구 전용이어서 본선 완주에 왼쪽 변경이 필요하다. 1차로에서는 그대로 본선을 선택할 수 있다. 각 출발 차로는 18회다."]
    lane_rows=[]
    for key,rows in rows_by.items():
        for lane in (0,1):
            episodes=[r for r in rows if r["route"]==16 and r["lane"]==lane]
            assert len(episodes)==18
            outcomes=Counter(r["outcome"] for r in episodes)
            left=sum(r["left_commands"] for r in episodes) if "left_commands" in episodes[0] else "—"
            lane_rows.append([key,lane,*[outcomes[v] for v in OUT],left])
    doc+=table(["모델","출발 차로",*LABELS,"왼쪽 명령 수"],lane_rows)
    doc+=["## Q2와 Q5의 자가 경험 보상"]
    if "Q2/11→Q5/11" in comparisons:
        d=comparisons["Q2/11→Q5/11"]["delta"]; g=picks["Q5/11"]["training"]
        gain=d["arrival"]>0 and d["off_route"]<0 and d["collision"]<=0
        doc+=["",f"Q2 대비 Q5의 완주 {d['arrival']:+d}건, 충돌 {d['collision']:+d}건, 이탈 {d['off_route']:+d}건. "
              f"완주 증가·이탈 감소·충돌 유지 또는 감소의 동시 달성은 **{'충족' if gain else '미달'}**이다.",
              "",f"선정 시점까지 Q5의 온라인 정확도는 {g['accuracy']:.1%}, AUC는 {g['auc']:.3f}였다. "
              f"다음 롤아웃의 게이트가 열린 업데이트는 {g['open_next']}/{g['updates']}회, 보상 계산은 {g['shaped_steps']:,}스텝이었다. "
              f"종료한 {g['episodes']}회 주행의 경험 보상 합은 {g['r_exp']:+.2f}다."]
    doc+=["","## 선정과 학습 안정성","", "학습 중 포함 경로 수와 경로별 스텝 점유율의 표준편차는 온전한 크기의 롤아웃에서 구해 평균했다. 경로 통과 성공률과는 다른 지표다."]
    doc+=table(["모델","선정 스텝","선정 완주 /128","포함 경로 수 /16","경로 점유율 표준편차","KL 조기중단 / 업데이트"],
               [[k,v["step"],v["screen"]["arrival"],f"{v['training']['route_coverage']:.2f}",f"{v['training']['route_share_sd']:.3f}",
                f"{v['training']['kl_stops']}/{v['training']['updates']}"] for k,v in picks.items()])
    if all(f"{a}/11" in picks for a in q.ARMS):
        doc+=["### 시드 11의 40만 스텝 전체 업데이트","",
              "5만 스텝 경계의 짧은 롤아웃도 포함했다. 온전한 롤아웃에서 10에포크를 모두 사용하면 미니배치 업데이트는 320회다."]
        budget=[]
        for arm in q.ARMS[1:]:
            path=q.folder(arm,11)/"updates.jsonl"
            updates=[json.loads(line)["metrics"] for line in path.read_text().splitlines()]
            budget.append([arm,len(updates),sum(m["kl_stop"] is not None for m in updates),
                           median(m["minibatches_applied"] for m in updates)])
        doc+=table(["조건","전체 업데이트","KL 조기중단","실제 적용 미니배치 중앙값"],budget)
    doc+=table(["조건","시드","스텝",*LABELS,"변경 필요 완주 /52"],screens)
    stats={}
    if winner and all(f"{winner}/{s}" in data for s in q.SEEDS):
        group=[data[f"{winner}/{s}"] for s in q.SEEDS]
        for metric in (*OUT,"required_arrived"):
            denominator=208 if metric=="required_arrived" else 576
            fractions=[v[metric]/denominator*100 for v in group]
            stats[metric]=dict(mean=mean(fractions),sd=stdev(fractions),rates=fractions)
        doc+=["## 우승 조건의 세 시드 재현성","",f"시드 11의 128회 평가로 **{winner}** 조건을 선정했다. 시드마다 독립적으로 선정한 체크포인트의 576회 비율을 사용했다. 표준편차는 학습 시드 3개의 표본 표준편차다."]
        doc+=table(["지표","시드 11","시드 22","시드 33","평균 ± 표준편차"],
                   [[label,*[f"{v:.2f}%" for v in stats[key]["rates"]],f"{stats[key]['mean']:.2f}% ± {stats[key]['sd']:.2f}%p"] for key,label in zip((*OUT,"required_arrived"),(*LABELS,"변경 필요 완주율"))])
    doc+=["## 실행 조건","",
          "- P0→Q1은 도로 단위 분기 방향과 해당 방향의 차로 연결 슬롯을 함께 교체했다. 지도상의 분기 7개에서 SUMO 차로 순서와 대조했다. 모든 Q 조건은 처음부터 고교통 15,000대/시와 전 경로에서 출발한다.",
          "- Q2는 선형 학습률 감쇠와 KL 조기중단을 함께 적용했다. 기존 로거와 같은 미니배치 평균 log π_old − log π_new가 0.015를 넘으면 그 미니배치부터 해당 업데이트를 끝낸다. Q3는 롤아웃과 미니배치 크기를 함께 두 배로 바꿨고, Q4는 충돌과 이탈의 벌점을 함께 바꿨다. 각 대조의 효과는 묶음 단위다.",
          "- Q5는 새로 결과가 판명된 자기 주행의 분기 상태를 예측기 학습 전에 채점한다. 가장 최근 검증 표본이 있는 롤아웃 정확도가 70% 이상일 때 다음 롤아웃에서 α=0.3, ±0.1 클립을 적용한다. 70% 미만으로 떨어지면 다시 끈다.",
          "- 평가 패널은 95380000 128회, 81260100 64회, 91000000·104320000 각 256회다. 모든 평가에서 결정론적 행동을 사용하며 학습용 10초 종료는 껐다. 원본 BC와 dagger5.pt의 SHA-256은 시작·종료 때 검사한다.",
          "- 128회에서 완주 최대, 충돌 최소, 변경 필요 출발 완주 최대, 이른 스텝 순으로 선정한다. 시드 11은 P0와 초기 가중치·교통 시드 편성을 맞췄다. 22·33은 초기화·행동 샘플링·경로 순서·교통 시드 대역을 바꾼 독립 학습이다.",
          "", "[실행 코드](../../utils/jys3_pure_ppo_q_sweep.py) · [원자료 집계](../../results/jys3_pure_ppo_q_sweep/summary.json) · [기존 P0 실험](jys3-pure-ppo-overnight.md)"]
    if stats:
        a=data["Q1/11"]; base=data["P0"]
        doc[4:4]=["## 결과 요약","",
                  f"- {winner} 조건이 선정 평가 1위를 기록했다. 세 시드 완주율 평균은 {stats['arrival']['mean']:.2f}% "
                  f"(표본 표준편차 {stats['arrival']['sd']:.2f}%p), 충돌률 평균은 {stats['collision']['mean']:.2f}%다. "
                  f"같은 시드 11에서 Q1의 완주는 P0 대비 {a['arrival']-base['arrival']:+d}건이었다. P0의 세 시드 평균은 이번 실험에서 측정하지 않았다.",
                  f"- Q2~Q5의 선정 정책이 576회에서 낸 좌·우 명령은 합계 "
                  f"{sum(sum(data[arm+'/11']['all'][k] for k in ('left','right')) for arm in q.ARMS[1:]):,}회였다. "
                  f"Q2의 전체 업데이트 {budget[0][1]}회 중 {budget[0][2]}회가 KL 기준에 도달했다. "
                  f"실제 적용한 미니배치 중앙값은 {budget[0][3]:g}회였다.",
                  f"- 보강형 DAgger R5는 576회 중 {data['R5']['arrival']}회 완주했고 충돌은 {data['R5']['collision']}건이었다. "
                  f"일곱 Q 모델 중 완주율 95%와 충돌률 5% 미만을 함께 충족한 모델은 "
                  f"{sum(d['arrival']/576>=.95 and d['collision']/576<.05 for k,d in data.items() if k.startswith('Q'))}개였다.",""]
    dest=q.REPORT; temp=dest.with_suffix(".md.tmp")
    temp.write_text("\n".join(doc)+"\n",encoding="utf8"); temp.replace(dest)
    save(q.ROOT/"summary.json",dict(winner=winner,screens=len(screens),q_holdouts=[k for k in data if k.startswith("Q")],
         models=data,selected=picks,comparisons=comparisons,three_seed=stats))
