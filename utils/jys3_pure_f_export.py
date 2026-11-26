"""Export paired F experiment counts and road behavior as a Korean report."""
from pathlib import Path
from collections import Counter
import json
from utils.jys3_pure_f import ROOT,ARMS,STEPS

def table(stats):
    lines=["|조건·시드·스텝|완주|충돌|일반 시간초과|잘못된 차로 끝 정지|전방이 빈 저속 시간¹|실제 차로변경/주행|13·14번 최종 분기²|16번 0차로 명령→변경→본선→완주³|",
    "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for k,v in stats.items():
        c=Counter(v["outcomes"])
        values=[f"{c[o]}/64 ({100*c[o]/64:.1f}%)" for o in ("arrival","collision","timeout","stall")]
        q=v["sixteen"]
        flow=" → ".join(str(q[x]) for x in ("command","change","main","finish"))
        lines.append("|"+"|".join([k,*values,f'{v["slow_s"]}초',str(v["moves"]),f'{v["exit13"]}/4 · {v["exit14"]}/4',f'{flow} (총 {q["n"]}건)'])+"|")
    return lines

def main():
    summ=json.loads((ROOT/"summary.json").read_bytes())
    lines=["# 순수 PPO의 전진 보상과 두 번째 분기 안내","",
    "F0는 기준, F1은 경로 전진 보상(+2/완주 거리), F2는 다음 두 번째 분기 방향·거리 관측, F3는 두 변경을 결합했다. 고교통 16개 경로, 1,000스텝 상한, 동일 초기 가중치의 52차원 정책을 사용했다. 주행 평가는 확률 최대 행동을 선택했다.","",
    "## 학습 시드 11 · 네 조건 비교","",*table(summ),"","¹ 속도 0.5m/s 이하이며 같은 차로 앞차가 없거나 25m 초과인 시간의 64회 평균. 양보 대기도 포함될 수 있다. 충돌로 일찍 끝나면 누적 시간도 짧아진다.",
    "² 자연 출발에서 각 경로의 최종 선택 도로로 들어간 횟수. ³ 지표는 행동 횟수가 아니라 각 단계에 도달한 에피소드 수다.","","## 실제 주행","","![같은 출발에서 시간에 따른 속도와 차로 번호](drives.png)","","도표는 시드 11의 32,768스텝 모델, 13번 평가 회차 1과 16번 회차 0을 5초 간격으로 표시한다. 도로가 바뀌면 차로 번호도 새 도로 기준으로 기록된다.","",
    "## 51m 공통 접근 도로 국소 재생","","13·14번을 같은 위치·속도에서 시작시켰다. 차로 0·1·2와 교통 시드 2개를 조합한 6쌍이다. 자연 출발 평가와 별도로 집계했다.","","|조건|13번 최종 분기|14번 최종 분기|두 경로 모두 통과|","|---|---:|---:|---:|"]
    for arm in ARMS:
        rows=json.loads((ROOT/f"local_{arm}.json").read_bytes())
        pairs=list(zip(rows[::2],rows[1::2]))
        assert all(a["initial"][:50]==b["initial"][:50] for a,b in pairs)
        lines.append(f'|{arm}|{sum(a["passed"] for a,b in pairs)}/6|{sum(b["passed"] for a,b in pairs)}/6|{sum(a["passed"] and b["passed"] for a,b in pairs)}/6|')
    path=ROOT/"confirmation_summary.json"
    if path.exists():
        lines+=["","## 독립 검증 · 학습 시드 22·33","","F0와 F2를 각 65,536스텝까지 학습하고 32,768·65,536스텝에서 같은 새 교통 시드로 평가했다. 각 행은 64회다.","",*table(json.loads(path.read_bytes()))]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8")
if __name__=="__main__":main()
