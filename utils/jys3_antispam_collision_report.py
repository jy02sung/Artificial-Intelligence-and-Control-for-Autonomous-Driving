"""Classify reconstructed events from actual lanes, SUMO participants and policy observations."""
from collections import Counter
from statistics import median
from utils import jys3_diagnose_antispam as d

ROOT = d.ROOT
REPORT = d.exp.REPORT.parent/"jys3-antispam-collision-diagnosis.md"
NAMES = {"cut":"측면 컷인", "front":"동일 차로 전방 추돌", "merge":"합류·교차부", "other":"기타·왕복"}
LABELS = {"rear":"자차가 뒤에서 받힘", "move":"최근 실제 변경 후",
          "junction":"내부 링크 또는 합류 진입", "unknown":"기타 위치·상대 관계",
          "ping":"같은 차로 사이 2왕복", "front":"자차가 앞차를 추돌", "cut":"최근 목표 차로 여유 부족"}


def actual_moves(r):
    return [(t,a,b) for t,vehicle,a,b in r["lane_changes"] if vehicle==r["ego"] and a[0]==b[0] and a[1]!=b[1]]


def classify(r):
    ego=r["ego"]
    tail=r["tail"]
    last=tail[-1]
    end=r["steps"]
    events=r["events"]
    assert events, (r["seed"],"missing SUMO participant")
    moves=actual_moves(r)
    recent=[m for m in moves if end-m[0]<5]
    frames={f["step"]:f for f in tail}
    event=events[0]
    partner=event["victim"] if event["collider"]==ego else event["collider"]
    ego_hits=event["collider"]==ego
    loc=d.split_lane(event["lane"])
    cut=None
    for t,a,b in reversed(recent):
        if end-t>=3 or b!=loc:
            continue
        m=frames[t]["view"][str(1 if b[1]>a[1] else -1)]
        if m["tight"]:
            cut=dict(m,step=t,delay=end-t,
                     partner_on_target=frames[t]["cars"].get(partner,[])[:2]==b)
            break
    front=False
    own=last["cars"][ego]
    peer=last["cars"].get(partner)
    if peer and own[:2]==peer[:2]:
        front=peer[2]>=own[2] and ego_hits
    junction=loc[0].startswith(":") or event["kind"]=="junction" or r["merge_entry"]
    ping=False
    if len(moves)>=4 and end-moves[-1][0]<3:
        four=moves[-4:]
        a,b=four[0][1:]
        ping=all((x,y)==((a,b) if i%2==0 else (b,a)) for i,(_,x,y) in enumerate(four))
    if cut:
        kind,sub="cut","cut"
    elif junction:
        kind,sub="merge","junction"
    elif not recent and front:
        kind,sub="front","front"
    else:
        sub=("ping" if ping else "move" if recent else "rear" if not ego_hits and peer and own[:2]==peer[:2] and own[2]>peer[2] else "unknown")
        kind="other"
    physical=None
    if peer and own[:2]==peer[:2] and peer[2]>=own[2]:
        physical=dict(gap=peer[2]-peer[4]-own[2],closing=own[3]-peer[3])
    return dict(kind=kind,sub=sub,role="collider" if ego_hits else "victim",cut=cut,ping=ping,
                contact_lane=loc,partner=partner,multi=len(events),recent_moves=len(recent),
                sensor=last["view"]["0"],physical_front=physical,speed=last["speed"],mu=last["mu"])


def quant(values):
    v=[x for x in values if x is not None]
    if not v:
        return [0,None,None]
    return [len(v),*map(float,d.np.percentile(v,[50,90]))]


def table(headers,rows):
    return ["","| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|",
            *["| "+" | ".join(map(str,row))+" |" for row in rows],""]


def top(values):
    return sorted(Counter(values).items(),key=lambda e:(-e[1],e[0]))[:5]


def extensions(docs, data, pools, drift):
    import gzip
    import json

    compact=gzip.compress(json.dumps(data,ensure_ascii=False,separators=(",",":")).encode(),mtime=0)
    (ROOT/"event_traces.json.gz").write_bytes(compact)
    docs.extend(["## SUMO 충돌 기록과 실제 차로 이동","",
                 "SUMO가 작성한 lanechange-output의 차량 ID·시각·출발/도착 차로를 사용했다. 충돌 스텝에 제거된 차량의 변경도 이 기록에 남는다. SUMO의 side는 차로변경 중 측면 접촉이고 collision은 앞뒤 접촉이다. 컷인 분류에는 변경 직전 정책 관측의 간격 조건까지 적용했다. [SUMO 충돌 출력 정의](https://sumo.dlr.de/docs/Simulation/Output/Collisions.html)"])
    def event_summary(rows):
        types=Counter(r["events"][0]["kind"] for r in rows)
        cut=[r for r in rows if r["analysis"]["kind"]=="cut"]
        near=sum(r["analysis"]["cut"]["overlap"] for r in cut)
        return [len(rows), types["side"],types["collision"],types["junction"],len(cut),near]
    docs.extend(table(["모델","전체","SUMO 측면","SUMO 앞뒤","SUMO 교차","진단 컷인","컷인 중 목표 방향 겹침 관측"],[[name,*event_summary(rows)] for name,rows in pools.items()]))
    docs.extend(["## 시드 22의 완주 상실이 발생한 주행",""])
    paired=[]
    for seed in (22,33):
        def load_rows(track):
            step=d.exp.PEAK[seed] if track=="Q1" else 75000
            return {r["seed"]:r for panel in d.exp.q.HOLDS for r in d.read(d.exp.folder(track,seed)/f"eval_{step}_{panel}.json")["rows"]}
        old,new=load_rows("Q1"),load_rows("S2")
        assert len(old)==len(new)==576 and old.keys()==new.keys()
        lost=[(a,new[k]) for k,a in old.items() if a["outcome"]=="arrival" and new[k]["outcome"]=="off_route"]
        assert all(a["initial_sha256"]==new[k]["initial_sha256"] for k,a in old.items())
        paired.append([seed,sum(r["outcome"]=="off_route" for r in old.values()),sum(r["outcome"]=="off_route" for r in new.values()),len(lost),
                       sum(a["within_edge_moves"]>b["within_edge_moves"] for a,b in lost),
                       sum(b["within_edge_moves"]==0 for _,b in lost)])
    docs.extend(table(["시드","Q1 이탈 /576","S2 이탈 /576","완주→이탈","그중 실제 변경 감소","그중 실제 변경 0회"],paired))
    docs.extend(["시드 22의 관측된 실패는 필요한 변경까지 줄어든 것이다. 원래 완주하던 주행이 이탈로 뒤집힌 124건 모두 실제 변경이 줄었고 104건은 한 번도 바꾸지 않았다. 시드 33은 +7.5만에서 원본 변경 방향을 76.9% 유지했고 변경 필요 출발 완주가 선정 패널에서 39→45건이었다. 시드 22는 36.2% 유지에 그쳤고 완주가 33→8건이었다. 명령 비용·측면 겹침 벌점·충돌 벌점과 학습 일정이 함께 바뀌어 각 항의 독립 효과는 별도 실험 대상이다."])
    cases=d.read(ROOT/"command_history.json")
    docs.extend(["## 오른쪽 명령이 수백 회로 집계된 이유","",
                 "정책은 0.2초마다 한 번 판단한다. 좌·우 명령 수는 매 스텝의 출력이고 실제 변경은 같은 도로에서 차로 번호가 달라진 횟수다. Q1에는 좌·우 명령 자체의 비용이 없었다. 가장 오른쪽 0차로에서 다시 오른쪽을 요청하면 목표 차로가 범위를 벗어나 환경이 요청을 반환한다. 가감속과 전진은 별도로 계속 처리된다. 각 Q1 시드의 576회 중 오른쪽 명령이 가장 많았던 주행을 재생했다."])
    onset=[]
    for step in range(50000,400001,50000):
        cols=[]
        for seed in (22,33):
            rows=d.read(d.exp.q.panel("Q1",seed,step,"screen"))["rows"]
            cols.extend([sum(r["right_commands"] for r in rows),max(r["right_commands"] for r in rows)])
        onset.append([step,*cols])
    docs.extend(table(["Q1 학습 스텝","22 우 총합 /128회","22 한 주행 최대","33 우 총합 /128회","33 한 주행 최대"],onset))
    docs.append("같은 128개 조건에서 5만→10만 스텝 사이에 이미 반복 출력이 커졌다. 10만 시점 한 주행의 우 명령 최댓값은 시드 22에서 192회, 시드 33에서 983회였다. 20만 이후 성능 하락보다 먼저 나타난 행동이다.")

    trials=d.read(ROOT/"counterfactual.json")
    docs.extend(table(["Q1 시드","경로 / 교통 시드","종료","주행 스텝","우 명령","그중 외부 도로 0차로 우","우 명령 후 실제 우 변경","최장 연속 우"],[
        [x["model_seed"],f"{x['route']} / {x['seed']}",x["outcome"],x["whole"]["steps"],x["whole"]["right"],x["whole"]["right_in_zero"],x["whole"]["changed_right"],f"{x['longest']}회 ({.2*x['longest']:.1f}초)"] for x in cases if x["max_right_case"]]))
    docs.extend(table(["시드","0차로 우→유지로 바꾼 횟수","전 스텝 관측·보상·종료 일치"],[[x["model_seed"],x["replaced"]["commands"],x["identical"]] for x in trials]))
    docs.extend(["16번은 출구 전용 0차로에서 출발한 시드당 18회 전수를 별도로 재생했다. 아래 명령은 첫 진입 도로 86198728 위에서만 집계했다."])
    rows=[]
    for seed in (22,33):
        for reason in ("arrival","collision","off_route","timeout","stall"):
            group=[x for x in cases if x["model_seed"]==seed and x["route"]==16 and x["lane"]==0 and x["outcome"]==reason]
            if group:
                rows.append([seed,reason,len(group),sum(x["approach"]["left"] for x in group),sum(x["approach"]["right"] for x in group),
                             sum(x["approach"]["right_in_zero"] for x in group),sum(x["approach"]["left"]==0 for x in group)])
    docs.extend(table(["시드","종료","주행","좌 명령","우 명령","그중 0차로 우","좌 명령 0회인 주행"],rows))
    docs.extend(["Q1/22의 16번 오진입 10건에서는 첫 도로 우 명령이 총 7회, 좌 명령 0회인 주행이 9건이었다. 이 시드에서 주된 실패 행동은 필요한 좌 명령 누락이다. Q1/33의 오진입 2건은 첫 도로에서 좌 0회, 외부 0차로에서 우 133회였다. 결정론적 평가에서도 같은 출력을 반복하므로 여기서의 연속 명령은 탐험 샘플링으로 발생한 것이 아니다. 유효한 이동과 무효 요청에 같은 전진·완주 신호가 따라오는 환경에서 PPO가 두 출력을 구별할 유인이 약했다. 어느 학습 업데이트에서 우측 선호가 처음 형성됐는지는 이 평가가 측정한 범위 밖이다."])
    docs.extend(["[재생·정책 진단 코드](../../utils/jys3_diagnose_antispam.py) · [집계 코드](../../utils/jys3_antispam_collision_report.py) · [사건별 판정](../../results/jys3_diagnose_antispam/events.json) · [충돌·이탈 직전 관측과 SUMO 기록](../../results/jys3_diagnose_antispam/event_traces.json.gz) · [명령 대체 재생](../../results/jys3_diagnose_antispam/counterfactual.json)","",
                 "## 다음 순수 PPO S3 설계안","",
                 "1. **명령 비용을 반복 무효 요청에 한정해 비교한다.** 같은 방향으로 요청해 직전 스텝에도 이동하지 못했고 현재도 바깥 차로 경계인 경우에만 −0.02를 부과한다. 첫 정상 변경은 비용 0으로 둔다. 출발 시드 22·33에서 변경 필요 완주와 0차로 우 명령을 함께 본다.",
                 "2. **실제 컷인 직전의 앞·뒤 간격을 보상에 반영한다.** S2/33에서는 관측 겹침 플래그만으로 잡히지 않은 컷인이 많았다. 옆 차로 여유가 부족한 상태에서 실제 변경한 경우에 한해 기존 겹침 −0.30을 한 번 적용하는 조건을 별도로 비교한다. 충돌로 차량이 제거된 스텝도 포함한다. 앞이 이미 겹칠 만큼 가까운 경우와 뒤차가 닫아오는 경우를 따로 집계한다. 간격 기준은 기존 프로젝트의 식으로 시작한다.",
                 "3. **필요한 차로변경을 보존하는 Self-KL을 단독 비교한다.** 각 시드의 동결 Q1 정점 정책을 기준으로, 원본이 좌·우·유지를 선택한 관측을 나눠 차로 범주형 KL을 기록한다. 첫 후보는 유지 50%, 좌·우 각 25%의 그룹 가중치와 계수 0.1이다. 해당 그룹의 표본이 실제로 들어간 배치에서 계산한다. 수치는 시험할 초깃값이다. 시드 22에서 원래 필요했던 변경을 회복하는지, 시드 33에서 위험한 변경까지 다시 늘어나는지 검증한다.",
                 "", "각 변경을 같은 시작점에서 하나씩 대조하고, 최종 후보는 이번 원인 분석에 사용하지 않은 교통 시드에서 확인한다."])



def generate():
    data={}
    for tag,seed,step in d.SPECS:
        rows=[r for f in sorted((ROOT/"shards").glob(f"{tag}_{seed}_verified_*.json")) for r in d.read(f)]
        assert len(rows)==len({r["seed"] for r in rows})=={"S2":87,"S1":64,"Q1":50}[tag]
        for r in rows:
            if r["outcome"]=="collision":
                r["analysis"]=classify(r)
        data[f"{tag}/{seed}"]=rows
    pools={k:[r for r in rows if r["outcome"]=="collision"] for k,rows in data.items()}
    focus=pools["S2/33"]
    assert len(focus)==55
    cut=[r["analysis"]["cut"] for r in focus if r["analysis"]["kind"]=="cut"]
    front=[r["analysis"] for r in focus if r["analysis"]["kind"]=="front"]
    stats={}
    for k in ("front_gap","rear_gap","front_closing","rear_closing","ttc"):
        stats["cut."+k]=quant(x[k] for x in cut)
    for k in ("front_gap","front_closing"):
        stats["front."+k]=quant(x["sensor"][k] for x in front)
    for k in ("speed","mu"):
        stats["front."+k]=quant(x[k] for x in front)
    for k in ("gap","closing"):
        stats["physical."+k]=quant(x["physical_front"][k] for x in front)
    drift=d.read(ROOT/"drift.json")
    bymodel={}
    for k,rows in pools.items():
        a=[r["analysis"] for r in rows]
        bymodel[k]=dict(classes=Counter(x["kind"] for x in a),roles=Counter(x["role"] for x in a),
                        other=Counter(x["sub"] for x in a if x["kind"]=="other"),
                        ping=sum(x["ping"] for x in a),multi_events=sum(x["multi"]>1 for x in a),
                        moved=sum(bool(x["recent_moves"]) for x in a))
    events=[]
    for model,rows in data.items():
        for r in rows:
            event={k:r[k] for k in ("seed","route","lane","steps","outcome","final_lane","last_external","off_route","events")}
            event.update(model=model,analysis=r.get("analysis"))
            events.append(event)
    failures=[r for r in data["S2/33"] if r["outcome"]=="off_route"]
    assert len(failures)==32
    hotspots={}
    for name,rows in [("충돌",focus),("이탈",failures)]:
        def location(r):
            if name=="충돌":
                return r["analysis"]["contact_lane"]
            assert r["last_external"][0]==r["off_route"]["src"]
            return r["last_external"]
        hotspots[name]=dict(route=top(r["route"] for r in rows),
                            road=top(location(r)[0] for r in rows),
                            road_lane=top(tuple(location(r)) for r in rows),
                            route_road_lane=top((r["route"],*location(r)) for r in rows))
    d.save(ROOT/"events.json",events)
    d.save(ROOT/"summary.json",dict(models=bymodel,distributions=stats,hotspots=hotspots,drift=drift))
    docs=["# 순수 PPO 충돌 원인과 정점 이후 정책 변화","",
          "고교통 576회 패널에서 S2/33·Q1/22·S1/22의 충돌 전수와 S2/33의 실제 이탈 32건을 학습 없이 재생했다. "
          "초기 관측 SHA-256, 종료 사유, 종료 스텝, 좌·우 명령 횟수, 저장된 같은 도로 내 실제 변경 수를 검증했다.","","## 충돌 유형"]
    totals=[]
    for k,rows in pools.items():
        m=bymodel[k]
        totals.append([k,len(rows),*[f"{m['classes'][key]} ({m['classes'][key]/len(rows):.1%})" for key in NAMES],
                       m["roles"]["collider"],m["roles"]["victim"]])
    docs+=table(["모델","충돌",*NAMES.values(),"자차가 collider","자차가 victim"],totals)
    docs+=["동일 에피소드에서 실제 변경과 충돌 지점이 맞물린 유형을 분류했다. 보상이나 판단의 단독 인과효과는 이 재생에서 측정하지 않았다.",
           "", "1. 측면 컷인은 충돌 스텝 포함 최근 3스텝 이내 같은 도로에서 실제로 충돌 차로로 이동했고, 변경 직전 50차원 관측에서 여유가 부족한 경우다. "
           "앞 여유 기준은 3+0.45×자차속도+양의 접근속도, 뒤는 3+0.50×뒤차속도+양의 접근속도(m)이며 겹침도 포함한다. 이 간격식은 사후 분류 기준이다.",
           "2. 이어서 SUMO junction 충돌·내부 링크 접촉 또는 서로 다른 진입 도로가 모이는 차로에 최근 3스텝 이내 진입한 경우를 합류·교차부로 집계했다.",
           "3. 동일 차로 전방 추돌은 최근 5스텝에 자차의 실제 변경이 없고, 직전 자차와 상대가 같은 차로이며 상대가 앞에 있고 SUMO가 자차를 collider로 기록한 경우다.",
           "4. 나머지는 기타에 집계했다. 왕복은 직전 20스텝 내 동일 두 차로 사이에서 4번 실제 이동하여 두 번 되돌아오고 마지막 이동 후 3스텝 안에 충돌한 경우다. 왕복 여부는 각 유형과 독립적으로도 집계했다. 주행 의도·변경 필요성은 이 패턴에서 추론하지 않았다.",
           "", "도로 경계에서 차로 번호가 바뀐 것은 실제 변경에서 제외했다. 충돌 스텝의 실제 변경은 SUMO가 변경 순간에 기록한 출발·도착 차로를 대조했다. SUMO의 collider/victim은 시뮬레이터 상의 역할이다."]
    docs+=table(["모델","기타 세부 유형 · 건수","전체 중 최근 실제 변경(5스텝)","전체 중 2왕복"],[[k,", ".join(f"{LABELS[t]} {v}" for t,v in m["other"].items()),m["moved"],m["ping"]] for k,m in bymodel.items()])
    docs+=["## S2/33의 충돌 직전 물리량","",
           "컷인 표본은 실제 변경 직전, 추돌 표본은 충돌 직전 관측이다. 접근 속도는 가까워질 때 양수다. 미검출 표식은 결측으로 처리했다. "
           "TTC는 검출된 앞·뒤 중 양의 접근 속도가 있는 쪽의 최소 gap/closing이다. 표본 수는 유한값을 가진 사건 수다. 가감속 명령은 −1~+1 정규화 출력이다."]
    metrics=[("컷인 앞 간격 (m)","cut.front_gap"),("컷인 뒤 간격 (m)","cut.rear_gap"),("컷인 앞 접근 (m/s)","cut.front_closing"),("컷인 뒤 접근 (m/s)","cut.rear_closing"),("컷인 최소 TTC (s)","cut.ttc"),
             ("전방 추돌 앞 간격 (m)","front.front_gap"),("전방 추돌 앞 접근 (m/s)","front.front_closing"),("전방 추돌 자차 속도 (m/s)","front.speed"),("전방 추돌 가감속 명령","front.mu"),
             ("전방 추돌 실제 상대 간격 (m)","physical.gap"),("전방 추돌 실제 상대 접근 (m/s)","physical.closing")]
    docs+=table(["물리량","유효 건수","중앙값","90백분위"],[[label,stats[key][0],*[f"{v:.2f}" if v is not None else "—" for v in stats[key][1:]]] for label,key in metrics])
    docs+=[f"컷인 {len(cut)}건 중 변경 직전 상대가 이미 목표 차로에 있던 사례는 {sum(x['partner_on_target'] for x in cut)}건이다. 실제 상대 간격은 차량 위치와 앞차 길이로 계산했다. 관측 간격은 센서 노이즈·가림과 SUMO 간격 정의의 영향을 받는다."]
    docs+=["","## 경로·도로 집중 지점","",
           "충돌은 SUMO가 기록한 접촉 차로, 이탈은 오진입 직전 마지막 외부 진입 도로와 차로를 기준으로 했다. 각 지표를 독립 정렬했고 동률은 ID 순이다."]
    def items(xs):
        return " · ".join(f"{'/'.join(map(str,k)) if isinstance(k,tuple) else k} ({v}건)" for k,v in xs)
    docs+=table(["결과","경로 상위 5","도로 상위 5","도로/차로 상위 5"],[[n,items(m["route"]),items(m["road"]),items(m["road_lane"])] for n,m in hotspots.items()])
    docs+=table(["결과","경로/도로/차로 상위 5"],[[n,items(m["route_road_lane"])] for n,m in hotspots.items()])
    docs+=["## S2 시드별 정책 변화","",
           "같은 128개 출발에서 각 시드 원본 Q1을 재생하고 매 4스텝마다 수집한 두 시드의 관측을 합쳐 모든 정책에 동일하게 입력했다. "
           "KL은 원본→현재의 가감속 정규분포와 차로 범주분포의 해석적 KL 합(nat)이다. 변경 유지율은 원본이 좌·우를 선택한 관측에서 같은 방향을 고른 비율이다. "
           "비용은 학습 롤아웃에서 실제 부과된 비할인 −0.02 합이며, 변경 횟수·이탈·완주는 별도 128회 결정론적 선정 패널이다. 워밍업 후 행동 가중치와 출력이 동일해 원본 재생을 사용했다."]
    docs+=table(["시드","추가 PPO 스텝","공통 관측","KL 합","KL 차로 / 가속","변경 상태 차로 KL","변경 유지율","명령 비용 누계(워밍업 포함 / PPO만)","실제 변경 /128회","이탈 /128","변경 필요 완주 /52"],[
        [r["seed"],r["step"],r["n"],f"{r['kl']:.4f}",f"{r['kl_lane']:.4f} / {r['kl_accel']:.4f}",f"{r['kl_change']:.4f}",f"{r['retained']:.1%}",f"{r['total_cost']:.2f} / {r['ppo_cost']:.2f}",r["moves"],r["off_route"],r["required_arrivals"]] for r in drift])
    docs+=["", "[진단 코드](../../utils/jys3_diagnose_antispam.py) · [사건별 결과](../../results/jys3_diagnose_antispam/events.json) · [집계 원자료](../../results/jys3_diagnose_antispam/summary.json)"]
    extensions(docs, data, pools, drift)
    REPORT.write_text("\n".join(docs)+"\n",encoding="utf8")
