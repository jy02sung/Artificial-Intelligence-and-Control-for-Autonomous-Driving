"""Measure saved teacher trajectories; these numbers are NOT BC/PPO performance."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from comparison_setup import MODELS

def improvement(reference,value):
    return None if reference==0 else 100*(reference-value)/abs(reference)

def episode_metrics(ep,d,ix,dt,horizon):
    speed=d['obs'][ix,0]*33.33
    valid=d['action_valid'][ix,0].astype(bool);a=d['action_applied'][ix,0]
    braking=valid & (a < -3.)
    onsets=int(np.sum(braking & ~np.r_[False,braking[:-1]]))
    consecutive=valid[1:] & valid[:-1]
    jerk=(np.diff(a)/dt)[consecutive]
    duration=len(ix)*dt
    success=ep['end_reason']=='arrival'
    result=dict(episode=ep['episode'],success=float(success),timeout=float(ep['end_reason']=='time_limit'),
                collision=float(ep['ego_collisions']>0),teleport=float(ep['ego_teleports']>0),
                duration_s=duration,arrival_time_s=duration if success else None,
                penalized_time_s=duration if success else horizon,
                mean_speed_mps=float(np.mean(speed)),stopped_s=float(np.sum(speed<.1)*dt),
                stopped_fraction=float(np.mean(speed<.1)),hard_brake_events=onsets,
                hard_brakes_per_min=onsets/(duration/60),
                accel_rms=float(np.sqrt(np.mean(a[valid]**2))) if valid.any() else 0.,
                jerk_rms=float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.,
                lane_changes=int(np.count_nonzero(d['action_applied'][ix,1][d['action_valid'][ix,1].astype(bool)])))
    result.update({k:ep[k] for k in ['seed','route','traffic_scale','pair_id','repeat','split'] if k in ep})
    return result

def analyze(folder):
    m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    assert m['complete'] and len(m['episodes'])==30
    with np.load(folder/'dataset.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    rows=[episode_metrics(e,d,np.flatnonzero(d['episode']==e['episode']),m['step_length_s'],m['collection_config']['episode_limit_s']) for e in m['episodes']]
    unsafe=np.zeros(len(rows),int);observed=np.zeros(len(rows),int);leader_samples=np.zeros(len(rows),int);minimum=[None]*len(rows)
    with gzip.open(folder/'transitions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);ep=r['episode'];v=r['state']['ego']['speed'];dec=r['ego_rule_decision']
            rows[ep]['last_observed_edge']=r['state']['ego']['edge']
            rows[ep]['last_observed_lane']=r['state']['ego']['lane']
            rows[ep]['last_observed_speed_mps']=v
            observed[ep]+=1
            leader_samples[ep]+='leader_speed' in dec
            if 'leader_speed' in dec and v>dec['leader_speed']:
                ttc=max(0,dec['physical_gap_m'])/(v-dec['leader_speed'])
                unsafe[ep]+=ttc<1.5
                minimum[ep]=ttc if minimum[ep] is None else min(minimum[ep],ttc)
    q=json.loads((folder/'quality_report.json').read_text(encoding='utf-8'))
    splits={s:{'episodes':[],'bc_eligible_episodes':[],'transitions':0,'bc_eligible_transitions':0} for s in ['train','validation','test']}
    for row,ep,w in zip(rows,m['episodes'],q['warnings']):
        idx=ep['episode'];row.update(low_ttc_fraction=float(unsafe[idx]/observed[idx]),leader_observation_fraction=float(leader_samples[idx]/observed[idx]),min_ttc_s=minimum[idx],
            ego_emergency_requests=ep['rule_counts']['ego_emergency_requests'],
            ego_emergency_crossings=ep['rule_counts']['ego_emergency_crossings'],
            ordinary_solid_crossings=ep['rule_counts']['ordinary_solid_crossings'],
            background_teleport_warnings=w['background_teleport_warnings'],
            collision_warning_lines=w['collision_warning_lines'],steps=ep['steps'])
        # Initial eligibility only: a successful trace is not proof of optimal expert behavior.
        eligible=bool(row['success'] and not w['background_teleport_warnings'] and not w['collision_warning_lines'] and not row['ordinary_solid_crossings'])
        row['bc_initial_eligible']=eligible
        s=splits[ep['split']];s['episodes'].append(idx);s['transitions']+=ep['steps']
        if eligible:s['bc_eligible_episodes'].append(idx);s['bc_eligible_transitions']+=ep['steps']
    (folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    (folder/'episode_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return m,rows,splits

METRICS=[('success','완주율','%',False),('timeout','시간 제한 비율','%',True),('collision','ego 충돌 회차 비율','%',True),
         ('penalized_time_s','실패 포함 소요 시간','s',True),('stopped_s','정지 시간','s',True),
         ('mean_speed_mps','평균 속도','m/s',False),('hard_brakes_per_min','급제동 빈도','회/min',True),
         ('jerk_rms','가속도 변화 RMS','m/s³',True),('low_ttc_fraction','TTC<1.5s 노출 비율','%',True)]

def paired_interval(ref,candidate,rows,percent=False,lower=True):
    rng=np.random.default_rng(6324)
    groups={}
    for i,row in enumerate(rows):groups.setdefault((row['route'],row['traffic_scale']),[]).append(i)
    sample=np.concatenate([rng.choice(ids,(5000,len(ids)),replace=True) for ids in groups.values()],axis=1)
    r=np.asarray(ref)[sample].mean(axis=1);c=np.asarray(candidate)[sample].mean(axis=1)
    change=(r-c) if lower else (c-r)
    if percent:
        if np.any(r==0):return None
        change=100*change/np.abs(r)
    lo,hi=np.quantile(change,[.025,.975])
    return [float(lo),float(hi)]

def report(root):
    results={model:analyze(root/model) for model in MODELS}
    manifests={k:v[0] for k,v in results.items()};rows={k:v[1] for k,v in results.items()}
    ref=manifests['IDM']
    for model,m in manifests.items():
        for a,b in zip(ref['episodes'],m['episodes']):
            assert all(a[k]==b[k] for k in ['pair_id','seed','route','traffic_scale','split','demand_sha256','observed_ego_parameters']),(model,a['episode'])
        assert m['source_sha256']==ref['source_sha256'],model
        a=dict(m['ego_parameters']);b=dict(ref['ego_parameters']);a.pop('carFollowModel');b.pop('carFollowModel');assert a==b
    summary={};comparison={}
    for model in MODELS:
        summary[model]={key:float(np.mean([r[key] for r in rows[model]])) for key,_,_,_ in METRICS}
        summary[model].update(transitions=manifests[model]['transitions'],arrivals=sum(r['success'] for r in rows[model]),
            ego_emergency_requests=sum(r['ego_emergency_requests'] for r in rows[model]),
            ego_emergency_crossings=sum(r['ego_emergency_crossings'] for r in rows[model]),
            ordinary_solid_crossings=sum(r['ordinary_solid_crossings'] for r in rows[model]))
        comparison[model]={}
        for key,_,unit,lower in METRICS:
            base=summary['IDM'][key] if 'IDM' in summary else float(np.mean([r[key] for r in rows['IDM']]))
            val=summary[model][key]
            c={'absolute_difference':val-base,'relative_improvement_pct':improvement(base,val) if lower else improvement(base,val)*-1 if base else None}
            if unit=='%':c.update(difference_pp=100*(val-base),relative_improvement_pct=None)
            c['paired_95pct_interval']=paired_interval([r[key] for r in rows['IDM']],[r[key] for r in rows[model]],rows['IDM'],percent=unit!='%',lower=lower)
            if unit=='%' and c['paired_95pct_interval'] is not None:c['paired_95pct_interval']=[100*x for x in c['paired_95pct_interval']]
            comparison[model][key]=c
        common=[i for i in range(30) if rows['IDM'][i]['success'] and rows[model][i]['success']]
        baseline=float(np.mean([rows['IDM'][i]['duration_s'] for i in common])) if common else None
        candidate=float(np.mean([rows[model][i]['duration_s'] for i in common])) if common else None
        comparison[model]['common_success_time']={'pairs':len(common),'IDM_seconds':baseline,'candidate_seconds':candidate,
                                                 'improvement_pct':improvement(baseline,candidate) if baseline else None}
    result={'baseline':'IDM','episodes_per_controller':30,'summary':summary,'comparisons':comparison,
            'equal_bc_train_budget_upper_bound':min(v[2]['train']['bc_eligible_transitions'] for v in results.values()),
            'interval_method':'5000 paired bootstrap resamples stratified by route and load; seed variability only, exploratory',
            'not_learned_model_results':True,'markings_partially_verified':True}
    (root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    all_rows=[dict(controller=model,**r) for model in MODELS for r in rows[model]]
    with (root/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    lines=['# 동일 조건 제어기별 데이터 비교','', '**기준: 이번 실험의 IDM. 아래 결과는 시범 제어기 성능이며, BC·PPO 학습 성능이 아닙니다.**','',
           '제어기당 30회: 경로 2개 × 교통량 3단계 × 시드 5개. 동일한 차량 종류·출발 일정, 목표 속도 배율 1, 공통 ego 파라미터, 동일한 실선·위반·긴급 회피 규칙을 사용했습니다. 교통 상호작용 때문에 실행 중 차량 궤적은 달라집니다.','',
           '| 지표 | IDM 기준 | Krauss | EIDM | ACC |','|---|---:|---:|---:|---:|']
    for key,title,unit,lower in METRICS:
        cells=[]
        for model in ['IDM','Krauss','EIDM','ACC']:
            val=summary[model][key];c=comparison[model][key]
            text=f'{val*100:.2f}%' if unit=='%' else f'{val:.3f} {unit}'
            if model!='IDM':
                if unit=='%':text+=f" ({c['difference_pp']:+.2f}%p)"
                elif c['relative_improvement_pct'] is None:text+=' (기준 0: 상대율 없음)'
                else:text+=f" (개선 {c['relative_improvement_pct']:+.1f}%)"
            cells.append(text)
        lines.append('| '+title+' | '+' | '.join(cells)+' |')
    lines+=['','시간·급제동·가속도 변화·위험 노출은 낮을수록, 완주율·평균 속도는 높을수록 유리한 방향으로 표시했습니다. 정지하거나 느리게 운전하면 승차감 지표만 좋아질 수 있으므로 단독 순위로 쓰지 않습니다.','',
            '## 성공·실패를 공정하게 비교하기','',
            '- 실패 포함 시간 = 완주 시 실제 관찰 시간, 실패·시간 제한·충돌 시 300초. 중도 실패를 빠른 주행으로 잘못 평가하지 않습니다.',
            '- 급제동 = 실제 가속도 < −3m/s²인 연속 구간 한 번. 가속도 변화 RMS는 연속 유효 가속도 차이/0.2초이며 회차 경계를 연결하지 않습니다.',
            '- TTC는 진행 경로상의 앞차와 가까워질 때 물리적 간격/접근속도입니다. 지도상 가까운 고가도로 차량을 앞차로 취급하지 않습니다. 현재 규칙 기록은 일반 edge에서 100m 이내 앞차만 조회하고 내부 교차로에서는 조회하지 않습니다. 노출 비율의 분모는 전체 관찰 스텝이므로 미관측 구간의 안전을 보증하지 않습니다. episode_metrics.json에 앞차 관측 비율도 저장합니다.',
            '- 지표는 회차별 값을 계산한 뒤 동일 가중치로 평균합니다. 실패 기록도 전체 비교에 포함합니다.',
            '- 비율은 %p 차이, 나머지는 IDM 대비 유리한 방향의 상대 변화율입니다. 기준 0이면 나눗셈을 하지 않습니다.','',
            '| 비교 | 둘 다 완주한 쌍 | IDM 시간 | 해당 제어기 시간 | 시간 개선 |','|---|---:|---:|---:|---:|']
    for model in ['Krauss','EIDM','ACC']:
        c=comparison[model]['common_success_time']
        lines.append(f"| {model} | {c['pairs']} | {c['IDM_seconds']:.2f}s | {c['candidate_seconds']:.2f}s | {c['improvement_pct']:+.2f}% |" if c['pairs'] else f'| {model} | 0 | — | — | — |')
    lines+=['','## 차이의 불확실성','', '동일 조건의 쌍을 경로·교통량별로 5,000회 재표집한 95% 구간입니다. 0을 포함하면 이 표본만으로 우위를 단정하기 어렵습니다. 각 조건당 시드 5개인 탐색 실험이며 현실 교통에 대한 신뢰구간은 아닙니다.','',
            '| 비교 | 완주율 개선 구간 (%p) | 실패 포함 시간 개선 구간 (%) |','|---|---:|---:|']
    for model in ['Krauss','EIDM','ACC']:
        a=comparison[model]['success']['paired_95pct_interval'];b=comparison[model]['penalized_time_s']['paired_95pct_interval']
        lines.append(f'| {model} | {a[0]:+.2f} ~ {a[1]:+.2f} | {b[0]:+.2f} ~ {b[1]:+.2f} |')
    lines+=['','## 조건별 완주 수 / 실패 포함 평균 시간','',
            '| 경로 | 교통량 배율 | IDM | Krauss | EIDM | ACC |','|---|---:|---:|---:|---:|---:|']
    for route,scale in sorted({(r['route'],r['traffic_scale']) for r in rows['IDM']}):
        cells=[]
        for model in ['IDM','Krauss','EIDM','ACC']:
            selected=[r for r in rows[model] if r['route']==route and r['traffic_scale']==scale]
            cells.append(f"{sum(int(r['success']) for r in selected)}/5 · {np.mean([r['penalized_time_s'] for r in selected]):.1f}s")
        lines.append(f'| {route} | {scale} | '+' | '.join(cells)+' |')
    lines+=['','## 데이터 품질과 학습 분할','',
            '| 제어기 | 전체 전이 | 초기 BC 적합 회차 | 학습용 적합 전이 | 배경 텔레포트 경고 | 일반 차량 실선 위반 |','|---|---:|---:|---:|---:|---:|']
    for model in ['IDM','Krauss','EIDM','ACC']:
        lines.append(f"| {model} | {summary[model]['transitions']:,} | {sum(r['bc_initial_eligible'] for r in rows[model])}/30 | {results[model][2]['train']['bc_eligible_transitions']:,} | {sum(r['background_teleport_warnings'] for r in rows[model])} | {summary[model]['ordinary_solid_crossings']} |")
    lines+=['','## BC+PPO에 사용할 때','',
            '각 폴더의 splits.json은 같은 시드 조건을 train 18회 / validation 6회 / test 6회로 분리합니다. 시간 프레임을 무작위로 섞어 회차가 양쪽에 들어가는 누수를 피합니다. bc_eligible_episodes는 완주 및 경고 기반 초기 필터이며 최적 운전 정답을 보증하지 않습니다.',
            f"동일 BC 학습량 비교에 사용할 수 있는 공통 상한은 현재 초기 필터 기준 **{result['equal_bc_train_budget_upper_bound']:,}개 전이**입니다. 실제 학습에서는 동일한 샘플 수·학습 횟수와 경로/교통량 구성을 함께 맞춰야 합니다.",
            'PPO는 BC 가중치에서 시작하되 동일한 추가 상호작용 예산으로 학습하고, 별도의 공통 평가 시드에서 비교해야 합니다. 시험 시드로 추가 학습하지 않습니다. 이 보고서로 시험 결과를 이미 살펴봤으므로 최종 논문식 평가에는 새 미사용 평가 시드를 추가하는 것이 좋습니다.',
            '관측은 기본 54차원 + rule_context 4차원입니다. 행동은 실제 속도 변화/차선 변화이며 명령값이 아닙니다. 50m 밖 정보나 제어기 내부 상태가 관측에 없으므로 일부 시범 행동은 현재 관측만으로 복원하기 어려울 수 있습니다.',
            '도로 표시는 기존 v2의 부분 검증 상태를 유지했습니다. 기존 baseline_v1 / rules_v2 데이터는 변경하지 않았습니다.','',
            '## 저장 파일','', '- comparison.json: 전체 요약·기준 대비 변화·95% 구간','- episodes.csv: 120회 각각의 수치와 조건','- 각 제어기/dataset.npz: 전체 학습용 배열','- 각 제어기/manifest.json, episode_metrics.json, splits.json: 재현 조건·회차 지표·분할','- 로컬 results의 원본 폴더: 전체 JSONL·SUMO 경고·규칙 이벤트·실행 코드 스냅샷','']
    (root/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);print(json.dumps(report(p.parse_args().root),indent=2))
