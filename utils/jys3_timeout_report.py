"""Summarize timeout measurements separately for training and fixed-policy evaluation."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

from utils.jys3_timeout_distances import DEST,ROOT

def numbers(rows):
    rem=[float(r["remaining_m"]) for r in rows]
    travel=[float(r["last30_m"]) for r in rows]
    return [len(rows),round(median(rem),1) if rem else None,round(median(travel),1) if travel else None,
            sum(x<=1 for x in travel),sum(x<=10 for x in travel),sum(x<=100 for x in rem),
            sum(x<=100 and y>=30 for x,y in zip(rem,travel))]

def export(path,rows):
    if not rows:return
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def write():
    with (DEST/"episodes.csv").open(encoding="utf-8-sig") as f: ev=list(csv.DictReader(f))
    header=["시간초과 건수","남은 거리 중앙값(m)","마지막 30초 주행거리 중앙값(m)","1m 이하 주행","10m 이하 주행","목적지 100m 이내","100m 이내이면서 30m 이상 진행"]
    lines=["# F 실험의 일반 시간초과 재생","",
           "SUMO 1,000스텝(200초)에서 일반 시간초과된 사례를 측정했다. 남은 거리는 지정 경로의 도로·연결부 길이를 합산한 기준 거리이며 차로별 길이 차이를 보정한다. 마지막 30초 주행거리는 850번째와 1,000번째 스텝 직후의 SUMO 누적 주행거리 차이다. 경로 이탈·텔레포트 여부와 종료 사유를 확인했다. 1m·10m·100m·30m 기준은 결과 설명용이며 학습이나 종료 판정에 사용하지 않았다.","",
           "## 고정 정책 평가","","같은 출발·교통 시드와 기존 종료 스텝을 확인했다. 표의 수치는 체크포인트별 반복 평가 사례다.",""]
    def block(groups):
        lines.extend(["|조건|"+"|".join(header)+"|","|---|"+"---:|"*len(header)])
        for k,rs in groups:
            lines.append("|"+k+"|"+"|".join("—" if x is None else str(x) for x in numbers(rs))+"|")
    groups=defaultdict(list)
    for r in ev:groups[(int(r["seed"]),int(r["step"]),r["arm"])].append(r)
    block([("전체",ev)]+[(f"{a} · 시드 {s} · {n:,}",rs) for (s,n,a),rs in sorted(groups.items())])
    train=DEST/"training"
    paths=list(train.glob("F*/verification.json"))
    trainrows=[];early={};phase_rows=[]
    if len(paths)==8:
        lines.extend(["","## 실제 PPO 학습","","원본 가중치와 모든 체크포인트의 텐서가 일치한 재현 학습이다. 완료한 에피소드만 집계한다. 구간은 에피소드 종료 시점 기준이다.",""])
        for p in sorted(paths):
            assert all(json.loads(p.read_bytes()).values())
            arm,seed=p.parent.name.split("_")
            records=[json.loads(x) for x in (p.parent/"training_outcomes.jsonl").read_text().splitlines()]
            for r in records:
                if r["outcome"]=="timeout":
                    m=r["timeout_distances"]
                    assert m and m["intact"]
                    trainrows.append(dict(arm=arm,seed=seed,train_step=r["step"],route=r["route"],traffic_seed=r["traffic_seed"],**m))
            for low,high in [(0,8192),(8192,32768),(32768,65536)]:
                subset=[r for r in records if low<r["step"]<=high]
                if not subset: continue
                cnt=Counter(r["outcome"] for r in subset)
                phase_rows.append(dict(arm=arm,seed=seed,start=low,end=high,episodes=len(subset),arrival=cnt["arrival"],collision=cnt["collision"],stall=cnt["stall"],timeout=cnt["timeout"]))
        block([("전체",trainrows)]+[(f"{p.parent.name}",[r for r in trainrows if f'{r["arm"]}_{r["seed"]}'==p.parent.name]) for p in sorted(paths)])
        export(DEST/"training_timeouts.csv",trainrows)
        export(DEST/"training_by_stage.csv",phase_rows)
        lines+=["","|학습 구간|종료 주행|완주|충돌|잘못된 차로 끝 정지|일반 시간초과|","|---|---:|---:|---:|---:|---:|"]
        for low,high in [(0,8192),(8192,32768),(32768,65536)]:
            rs=[r for r in phase_rows if r["start"]==low]
            lines.append(f"|{low:,}–{high:,}|"+"|".join(str(sum(r[k] for r in rs)) for k in ("episodes","arrival","collision","stall","timeout"))+"|")
    extra=DEST/"extended_subset.csv"
    if extra.exists():
        with extra.open(encoding="utf-8-sig") as f: trials=list(csv.DictReader(f))
        lines+=["","## 시간제한 1,500스텝 재생","","남은 거리 100m 이하 또는 최근 30초 주행거리 10m 초과 케이스만 연장했다. 두 지표는 1,000스텝에서 기존 실측과 일치한다. 0스텝에서 같은 초기 가중치를 공유하는 조건의 중복 사례가 포함된다.","","|모델|시드|가중치 스텝|경로|남은 거리(m)|최근 30초(m)|최종 스텝|결과|","|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in trials:
            lines.append("|"+"|".join(r[k] for k in ("arm","seed","step","route","remaining_m","last30_m","extended_steps","extended_outcome"))+"|")
    (DEST/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("evaluation",numbers(ev),"verified_training_runs",len(paths),"training",numbers(trainrows))
if __name__=="__main__":write()
