"""Paired summaries for the three redesign tracks. All files are written from completed runs."""
from collections import Counter
import json
from statistics import mean

import torch

from utils import jys3_ppo_redesign_v2 as run

old, final, ROOT = run.old, run.final, run.ROOT
read, save = run.read, run.save
PANELS = old.PANELS
OUT = ("arrival","collision","stall","timeout","off_route","slow10")
LABEL = {"screen":"선정 128회","hold64":"64회","designated":"지정 256회","additional":"추가 256회"}


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def table(headers,rows):
    return ["", "| "+" | ".join(headers)+" |", "|"+"|".join(["---"]*len(headers))+"|",
            *["| "+" | ".join(map(str,row))+" |" for row in rows], ""]


def counts(t):
    return " / ".join(str(t[k]) for k in OUT)


def rates(t,n):
    return f"{100*t['arrival']/n:.2f}% / {100*t['collision']/n:.2f}%"


@torch.inference_mode()
def probe(arm,steps):
    x,y=final.dataset(read(final.ROOT/"valid/episodes.json"),"valid")
    cls=(1-y[:,1]).long()
    assert (cls!=1).sum()==29
    ref=final.network(old.ORIG[old.SOURCE[arm]])
    base_log=ref(x)[2].log_softmax(-1)
    records={}
    for step in [0]+steps:
        model=final.network(run.path_for(arm,step)) if step else ref
        z=model(x)[2]
        metrics=run.diagnosis.fmetrics(cls,z.argmax(1))
        lp=z.log_softmax(-1)
        kl=(lp.exp()*(lp-base_log)).sum(-1)
        metrics.update(correct=sum(metrics["matrix"][i][i] for i in (0,2)),
                       kl_keep=kl[cls==1].mean().item(),kl_change=kl[cls!=1].mean().item())
        records[step]=metrics
    return records


def needs(rows,geometry):
    subset=[r for r in rows if geometry[r["route"],r["lane"]]=="needs_change"]
    return dict(n=len(subset),arrival=sum(r["outcome"]=="arrival" for r in subset),
                left=sum(r["left_commands"] for r in subset),right=sum(r["right_commands"] for r in subset),
                command_drives=sum(bool(r["left_commands"] or r["right_commands"]) for r in subset))


def generate(choices):
    torch.set_num_threads(1)
    before=read(old.ROOT/"selection.json")
    diagnostic=read(old.ROOT/"diagnosis/analysis.json")
    hashes=old.check_sources()
    geometries=run.diagnosis.geometry()
    results={}; checkpoints={}; probes={}; stages=[]
    paired_rows=[]; main_rows=[]; detail_rows=[]; checkpoints_table=[]; probe_rows=[]; group_rows=[]
    for arm in "ABC":
        step=choices[arm]
        results[arm]={}
        for panel in PANELS:
            a=read(old.data_path(arm,before[arm],panel))
            b=read(run.data_path(arm,step,panel))
            pair=old.paired(a,b)
            c,d=final.tally(a),final.tally(b)
            assert sum(d[k] for k in OUT[:5])==len(b)==16*PANELS[panel][1]
            res=dict(old=c,new=d,old_step=before[arm],step=step,paired=pair)
            if arm=="C":
                res["needs_change"]=needs(b,geometries)
            else:
                source=read(old.data_path(old.SOURCE[arm],0,panel))
                res["original_paired"]=old.paired(source,b)
            results[arm][panel]=res
            main_rows.append([arm,LABEL[panel],rates(c,len(b)),rates(d,len(b)),
                              f"{d['arrival']-c['arrival']:+d}",f"{d['collision']-c['collision']:+d}"])
            detail_rows.append([arm,LABEL[panel],counts(c),counts(d)])
            paired_rows.append([arm,LABEL[panel],pair["kept"],pair["lost"],pair["gained"]])
        interval=50000 if arm=="C" else old.MARK
        steps=list(range(interval,read(ROOT/arm/"done.json")["step"]+1,interval))
        checkpoints[arm]={}
        if arm in old.SOURCE:
            probes[arm]=probe(arm,steps)
            assert probes[arm][0]["correct"]==(22 if arm=="A" else 25)
            previous=diagnostic["checkpoints"][arm]
            old_metrics={0:previous["base_metrics"],**{r["step"]:r["metrics"] for r in previous["trajectory"]}}
            for s,m in probes[arm].items():
                prior=old_metrics.get(s)
                prior_text=f"{sum(prior['matrix'][i][i] for i in (0,2))}/29 · {prior['side_macro_f1']:.3f}" if prior else "—"
                probe_rows.append([arm,s,prior_text,f"{m['correct']}/29 ({m['side_recall']:.1%})",
                                   f"{m['side_macro_f1']:.3f}",f"{m['kl_keep']:.5f}",f"{m['kl_change']:.5f}"])
        for s in steps:
            rows=read(run.data_path(arm,s,"screen"))
            d=final.tally(rows)
            old_file=old.data_path(arm,s,"screen")
            same=final.tally(read(old_file)) if old_file.exists() else None
            if same:
                old.paired(read(old_file),rows)
            checkpoints[arm][s]=dict(old=same,new=d)
            checkpoints_table.append([arm,s,counts(same) if same else "—",counts(d)])
            if arm=="C":
                g=needs(rows,geometries)
                checkpoints[arm][s]["needs_change"]=g
                checkpoints[arm][s]["all_commands"]={k:sum(r[k] for r in rows)
                    for k in ("steps","left_commands","right_commands")}
                group_rows.append([f"{s:,}",LABEL["screen"],f"{g['arrival']}/{g['n']} ({g['arrival']/g['n']:.1%})",g["left"],g["right"],g["command_drives"]])
    for panel in PANELS:
        g=results["C"][panel]["needs_change"]
        group_rows.append(["선정",LABEL[panel],f"{g['arrival']}/{g['n']} ({g['arrival']/g['n']:.1%})",g["left"],g["right"],g["command_drives"]])
    g_all={k:sum(results["C"][p]["needs_change"][k] for p in ("designated","additional"))
           for k in ("n","arrival","left","right","command_drives")}
    old_groups=[x for x in diagnostic["groups"] if x["step"]==before["C"] and x["group"]=="needs_change"
                and x["panel"] in ("designated","additional")]
    assert sum(x["n"] for x in old_groups)==g_all["n"]==208
    assert all(not x["outcomes"].get("arrival") and not x["left"] and not x["right"] for x in old_groups)

    early_rows=lines(ROOT/"C/returns.jsonl")
    old_eps=lines(old.ROOT/"C/ppo_episodes.jsonl")
    stage_rows=[]
    for start,end in ((0,50000),(50000,100000),(100000,200000)):
        old_part=[x for x in old_eps if start<x["step"]<=end]
        part=[x for x in early_rows if start<x["step"]<=end]
        assert all(x["vph"]==(3000 if start==0 else 15000) for x in part)
        a=Counter(r["outcome"] for r in old_part)
        b=Counter(r["outcome"] for r in part)
        stops=[r["discounted_return"] for r in part if r["outcome"]=="clear_stop"]
        stages.append(dict(start=start,end=end,prior=a,current=b,n=len(part),prior_n=len(old_part),
                           stop_positive=sum(v>0 for v in stops),stop_mean=mean(stops) if stops else None))
        stage_rows.append([f"{start:,}~{end:,}",part[0]["vph"],
                           f"{a['collision']}/{len(old_part)} ({a['collision']/len(old_part):.1%})",
                           f"{b['collision']}/{len(part)} ({b['collision']/len(part):.1%})",
                           b["arrival"],b["off_route"],b["clear_stop"],
                           f"{sum(v>0 for v in stops)}/{len(stops)}",f"{mean(stops):+.2f}" if stops else "—"])
    models=[("원본 R5",old.data_path("R5",0,"designated")),
            ("기존 KL-PPO",old.data_path("B",before["B"],"designated")),
            ("재설계 KL-PPO",run.data_path("B",choices["B"],"designated"))]
    focus={name:{r["seed"]:r["outcome"] for r in read(path) if r["route"]==8 and r["seed"] in (91001008,91001208)}
           for name,path in models}
    assert list(focus["원본 R5"].values())==["stall","stall"]
    promotion=True; improved=False; r5rows=[]
    for panel in PANELS:
        rows=read(old.data_path("R5",0,panel)); n=len(rows)
        c=final.tally(rows); d=results["B"][panel]["new"]
        r5rows.append([LABEL[panel],counts(c),counts(d)])
        if panel!="screen":
            meets=d["arrival"]/n>=.95 and d["collision"]/n<.05 and d["arrival"]>=c["arrival"] and d["collision"]<=c["collision"]
            promotion &= meets
            improved |= d["arrival"]>c["arrival"] or d["collision"]<c["collision"]
    promote=promotion and improved
    conclusion="B로 교체" if promote else "원본 DAgger R5 유지"
    mixed={}
    for arm in ("A","B"):
        updates=lines(ROOT/arm/"updates.jsonl")
        upto=[r["metrics"]["mixed_batch_fraction"] for r in updates if r["step"]<=choices[arm]]
        mixed[arm]=mean(upto)
    save(ROOT/"summary.json",dict(selected=choices,prior_selected=before,panels=results,checkpoints=checkpoints,
        validation=probes,mixed_batches=mixed,training_C=stages,C_required_512=g_all,route8_cases=focus,
        recommendation=conclusion,source_hashes_after=hashes))
    doc=["# PPO 재설계 전후 비교","",
         f"발표 모델 판정은 **{conclusion}**다. 선정 128회에서 채택한 학습 스텝은 A {choices['A']:,}, B {choices['B']:,}, C {choices['C']:,}이다.",
         "", "A는 BC에서, B는 보강형 DAgger R5에서 PPO를 시작했다. C는 무작위 초기화한 순수 PPO다. 첫 표는 같은 교통 시드와 출발 차로에서 각 실험의 선정 모델을 짝지은 결과다."]
    doc+=table(["트랙","고교통 패널","기존 완주율 / 충돌률","재설계 완주율 / 충돌률","완주 증감(건)","충돌 증감(건)"],main_rows)
    hold_rows=[]
    for arm in "ABC":
        totals=[Counter(),Counter()]
        for panel in ("hold64","designated","additional"):
            for agg,key in zip(totals,("old","new")):
                agg.update(results[arm][panel][key])
        hold_rows.append([arm,rates(totals[0],576),rates(totals[1],576),
                          counts(totals[0]),counts(totals[1])])
    doc+=["세 홀드아웃을 합치면 모델당 서로 다른 고교통 출발 576회다. 선정에 사용한 128회는 이 합계에서 제외했다."]
    doc+=table(["트랙","기존 완주율 / 충돌률","재설계 완주율 / 충돌률","기존 6개 지표(건)","재설계 6개 지표(건)"],hold_rows)
    doc+=["## 모든 종료 사유와 완주 전환","", "각 셀은 완주 / 충돌 / 차로 끝 정지 / 일반 시간초과 / 실제 이탈 / 10초 이상 저속 합류 대기 순의 건수다. 마지막 열의 대기는 종료 사유와 중복될 수 있다."]
    doc+=table(["트랙","패널","기존","재설계"],detail_rows)
    doc+=table(["트랙","패널","양쪽 완주","기존 완주 상실","새 완주"],paired_rows)
    src_rows=[]
    for arm in ("A","B"):
        for panel in PANELS:
            trans=results[arm][panel]["original_paired"]
            src_rows.append([old.SOURCE[arm],arm,LABEL[panel],trans["kept"],trans["lost"],trans["gained"]])
    doc+=table(["PPO 시작 모델","트랙","패널","원본 완주 유지","원본 완주 상실","새 완주"],src_rows)
    doc+=["## 학습 단계별 고교통 128회", "", "같은 학습 스텝의 직전 실험과 대조했다. A·B의 스텝은 가치망 워밍업 20,000스텝 이후부터 센다."]
    doc+=table(["트랙","스텝","기존","재설계"],checkpoints_table)
    for arm in ("A","B"):
        ext=read(ROOT/arm/"extension.json")
        doc.append(f"- {arm}의 8,192스텝 간격 첫 네 번 완주 수는 {ext['counts']}였다. 65,536스텝 연장 여부는 {'연장' if ext['extend'] else '32,768에서 종료'}였다.")
    doc+=["","## 전문가가 차로변경한 29개 상태","",
          "재현율은 전문가와 같은 방향을 선택한 수 / 29다. F1은 유지 11,708개를 포함한 검증 전체에서 좌·우 각각의 F1을 구해 평균했다. KL은 전문가 정답으로 나눈 그룹의 차로 분포 KL이다. 학습 시 그룹은 동결 원본의 예측으로 정했다."]
    doc+=table(["트랙","스텝","기존 정답 · F1","재설계 정답 (재현율)","재설계 F1","유지 KL","변경 KL"],probe_rows)
    for arm in ("A","B"):
        s=choices[arm]; m=probes[arm][s]; orig=probes[arm][0]
        doc.append(f"- {arm} 선정 모델은 원본의 {orig['correct']}/29에서 {m['correct']}/29로, 좌·우 평균 F1은 {orig['side_macro_f1']:.3f}에서 {m['side_macro_f1']:.3f}로 바뀌었다. 선정 시점까지 90:10이 실제 적용된 미니배치 비율은 {mixed[arm]:.1%}였다.")
    doc+=["","## 순수 PPO의 교통 전환, 정지 수익, 차로변경","",
          "종료한 학습 주행을 종료 스텝으로 나눴다. 전환 시 이어지던 저교통 주행은 마지막 관측의 가치로 부트스트랩하고 고교통에서 새 시드로 출발했다. 전환 자체는 실패 집계에서 제외했다. 양수 수익은 10초 정지로 끝난 주행의 실제 할인 누적 보상이 0을 넘은 경우다."]
    doc+=table(["학습 스텝","재설계 수요(대/시)","기존 충돌","재설계 충돌","완주","이탈","10초 정지","양수 수익 / 10초 정지","정지 평균 할인 수익"],stage_rows)
    cmd_rows=[]
    for s,item in checkpoints["C"].items():
        ac=item["all_commands"]
        cmd_rows.append([f"{s:,}",ac["steps"],ac["left_commands"],ac["right_commands"],
                         f"{(ac['left_commands']+ac['right_commands'])/ac['steps']:.1%}"])
    doc+=table(["C 고교통 128회","주행 스텝 합계","좌 명령","우 명령","변경 명령 비율"],cmd_rows)
    doc+=["지도 연결선을 첫 분기까지 따라갔을 때 출발 차로에서 한 번 이상 옮겨야 하는 주행만 아래에 포함했다. 명령 수는 해당 주행 전체에서 낸 좌·우 명령의 합이다."]
    doc+=table(["C 스텝","고교통 패널","변경 필요 출발 완주율","좌 명령","우 명령","변경 명령을 낸 주행"],group_rows)
    doc += [f"두 256회 패널을 합치면 기존 C는 해당 출발에서 0/208 완주, 좌·우 각 0회였다. 재설계 후에는 {g_all['arrival']}/208 완주, 좌 {g_all['left']}회, 우 {g_all['right']}회였다.",
            "","## 8번 경로와 발표 모델",""]
    doc+=table(["시드","원본 R5","기존 B","재설계 B"],[[s,*[focus[m][s] for m,_ in models]] for s in (91001008,91001208)])
    doc+=table(["패널","원본 R5","재설계 B"],r5rows)
    doc+=["승격 기준은 세 홀드아웃에서 각각 완주율 95% 이상, 충돌률 5% 미만, 원본 이상의 완주 수와 원본 이하의 충돌 수를 충족하고 적어도 한 패널에서 개선되는 것이다.",
          "", "## 비교 조건","",
          "시드 11, ReLU 256×256, 관측 50개, 수정 분기, 고가 교차 20쌍 가림 보정, 안전게이트 OFF를 공통으로 썼다. 평가에는 학습용 10초 정지 종료를 적용하지 않았다. 모든 행의 경로·출발 차로·시드·첫 관측 SHA-256이 기존 주행과 일치했다. 평가 워커는 세 트랙 합계 최대 4개였다.",
          "", "A·B는 가치망 전용 복제와 미니배치별 유지·변경 KL 가중치를 함께 적용했다. 가치 손실의 역전파 분리와 워밍업 전후 행동 가중치의 비트 일치를 확인했다. 198배는 직전 B에서 측정한 변경 상태와 유지 상태의 차로 KL 비율이다. 이번 비교에서 각 변경의 단독 성능 효과는 분리할 수 없다.",
          "", "C는 가치망 분리, 첫 5만 스텝 교통량 감소, 10초 정지 벌점 −7→−10을 함께 적용했다. 전진 누적 +6, 분기 +2.5, 완주 +10, 충돌 −10, 이탈·끝 정지·시간초과 −8, 앞이 빈 저속 −0.02/스텝, 할인율 0.995는 세 트랙에 공통이다. 교통량은 SUMO에 설정한 시간당 유입 수요다.",
          "", "128회 평가에서 완주 최대, 충돌 최소, 끝 정지와 일반 시간초과의 합 최소, 이탈 최소, 이른 스텝 순으로 선정했다. 네 번째 평가에서 첫 번째와 세 번째보다 완주 건수가 모두 증가하면 상승 추세로 기록했다. 지정·추가 홀드아웃은 이번 학습 및 선정과 분리한 기존 비교용 시드다.",
          "", "원본 체크포인트의 시작·종료 SHA-256은 모두 일치했다.",
          "", *[f"- {key} `{value}`" for key,value in hashes.items()],
          "", "[실행 코드](../../utils/jys3_ppo_redesign_v2.py) · [집계 JSON](../../results/jys3_ppo_redesign_v2/summary.json) · [직전 비교](jys3-relu-ppo-comparison.md) · [실패 진단](jys3-ppo-failure-diagnosis.md)"]
    (final.DOCS/"jys3-ppo-redesign-v2.md").write_text("\n".join(doc)+"\n",encoding="utf-8")

