"""Summarize saved evaluation evidence and instrumentation-only timeout replays."""
import collections
import csv
import json
import xml.etree.ElementTree as E
from utils.analyze_trial_timeouts import cases, key, OUT, t

NAMES=dict(baseline='기본',obs_presence='차량 존재 표시',obs_risk='접근 위험',reward_gap='속도별 안전거리',reward_block='막힘 벌점 제거',reward_lane='위험 차선변경 벌점')

def classify(r):
    if not r['reproduced']:return '재현 미확인'
    tail=r['tail'][-50:];s=r['stats']
    if r['original_category']!='route_lane_blocked':
        if all(x['action'][0]<0 and x['speed']<.05 and not x['leader'] for x in tail):
            return '앞차 없이 지속 감속 명령으로 정지'
        return '추가 분석 필요'
    if not s.get('toward_request_steps',0):return '필요 방향 차선변경 요청 없음'
    if s.get('toward_actual_same_edge_steps',0):return '필요 방향 이동도 했지만 최종 차로 선택 실패'
    if not s.get('toward_sent_steps',0):return '필요 방향 요청을 환경이 전달하지 않음'
    return '필요 방향 요청 전달됐으나 실제 이동 미확인'

def main():
    original=cases();replays={p.stem:t.read(p) for p in (OUT/'cases').glob('*.json')}
    for p in (OUT/'extended').glob('*.json'):replays[p.stem]=t.read(p)
    assert len(replays)==len(original)==211
    network=E.parse(t.OUT/'inputs/reference/network.net.xml')
    edges={e.get('id'):e for e in network.findall('edge')}
    all_evals=[r for p in (t.OUT/'runs').glob('*/policy-*-evaluation.json') for r in t.read(p)]
    compact=[]
    for row in original:
        rr=replays[key(row)];last=row['diagnostic']['samples'][-1];tail=rr['tail'][-50:]
        entry=next((x for x in rr['entries'] if x['edge']==last['edge']),{})
        compact.append(dict(condition=row['condition'],checkpoint=row['checkpoint'],traffic=row['traffic'],route=row['route'],seed=row['seed'],
            outcome=row['outcome'],terminal_category=row['diagnostic']['category'],replay_category=classify(rr),reproduced=rr['reproduced'],
            edge=last['edge'],lane=last['lane_index'],allowed_lanes=str(last['allowed_lanes']),remaining=last['remaining'],
            leader_gap=last['leader_gap'],speed=last['speed'],replay_seconds=rr['seconds'],
            target_edge_length=float(edges[last['edge']].find('lane').get('length')),
            target_edge_entry_speed=entry.get('speed'),target_edge_entry_nav=json.dumps(entry.get('nav'),ensure_ascii=False),
            edge_has_change_restriction=any(l.get('changeLeft') or l.get('changeRight') for l in edges[last['edge']].findall('lane')),
            terminal_accel_mean=sum(x['action'][0] for x in tail)/len(tail),terminal_reward_mean=sum(x['reward'] for x in tail)/len(tail),
            terminal_correct_nav=all(x['nav']['current_targets']==last['allowed_lanes'] for x in tail),
            upstream_ambiguous_seen=any(x['nav']['ambiguous'] for x in rr['entries']),
            terminal_outside_lane_requests=sum(not 0<=x['lane']+x['direction']<len(edges[x['edge']].findall('lane')) for x in tail),
            **{k:rr['stats'].get(k,0) for k in ('wrong_lane_steps','toward_request_steps','toward_sent_steps','toward_actual_same_edge_steps','hold_steps','away_request_steps','ambiguous_steps')}))
    dest=t.ROOT/'docs/experiments/ic-observation-reward-timeouts-20261114.md'
    with dest.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    final=[r for r in compact if r['checkpoint']=='policy-200000.pt' and r['traffic']=='reference']
    counts=collections.Counter(r['replay_category'] for r in compact)
    lines=['# 관측·보상 실험 기타 실패 전수 분석','',
        '## 범위와 검증 방법','',
        '6조건 × 중간/최종2모델 × (교통48회+무교통16회) = 768회 평가에서 기타 실패211건을 전수 분석했다. 모두10,800초 시간 초과이며 삽입 실패·잘못된 출구 종료는0건이다. 최종 교통 평가의 기타 실패는78건이다.',
        '',f"동일 정책·맵·seed·교통 입력으로 재현한 사례: {sum(r['reproduced'] for r in compact)}/211. 재현은 원래 종료 위치·차로에서10초 연속 정지할 때 조기 종료했다. 원래10,800초 평가를 대체하거나 새 완주율을 계산하지 않는다. 1차 재현 상한900초,6프로세스이며 일치하지 않은 사례는 별도extended 기록으로 원래10,800초까지 확인한다. 행동 보정·재학습 없음.",
        '', '## 종료 상태와 행동 원인','',
        '- 원래 마지막59초 기록: 경로 비연결 차로 끝 정지198건, 그 밖의 정지13건. 저속 앞차 대기열로 분류된 사례0건. 마지막 상태만으로 초기 교통 영향까지 배제하지 않는다.',
        '', '| 재현에서 확인한 행동 | 전체 건수 | 최종 교통 건수 |','|---|---:|---:|']
    for label,n in counts.items():lines.append(f"|{label}|{n}|{sum(r['replay_category']==label for r in final)}|")
    lines+=['','## 최종 교통 조건별 결과','','| 조건 | 기타 실패 | 필요 방향 요청 없음 | 이동도 했지만 최종 실패 | 전달 후 이동 미확인 |','|---|---:|---:|---:|---:|']
    for c in NAMES:
        g=[r for r in final if r['condition']==c];cnt=collections.Counter(r['replay_category'] for r in g)
        lines.append(f"|{NAMES[c]}|{len(g)}|{cnt['필요 방향 차선변경 요청 없음']}|{cnt['필요 방향 이동도 했지만 최종 차로 선택 실패']}|{cnt['필요 방향 요청 전달됐으나 실제 이동 미확인']}|")
    lines+=['','## 실패 경로와 구간','','| 경로 | 최종 교통 실패 건수(18회 중) |','|---|---:|']
    for route,n in collections.Counter(r['route'] for r in final).most_common():lines.append(f'|{route}|{n}|')
    lines+=['','| 구간ID | 전체/최종 교통 실패 | 길이(m) | 종료 차로→필요 차로 |','|---|---:|---:|---|']
    for edge,n in collections.Counter(r['edge'] for r in compact).most_common():
        g=[r for r in compact if r['edge']==edge];pairs=sorted({str(r['lane'])+'→'+r['allowed_lanes'] for r in g})
        lines.append(f"|{edge}|{n}/{sum(r['edge']==edge for r in final)}|{g[0]['target_edge_length']:.2f}|{' ; '.join(pairs)}|")
    lines+=['','차로 번호는 SUMO index(0이 가장 오른쪽)다. 원시CSV에 조건·checkpoint·route·seed별 전체211건과 요청/전달/실제 변경 횟수를 기록했다.',
        '', '## 확인한 사실과 해석의 한계','',
        f"- 재현 마지막10초에 실제 경로 연결 차로와 안내가 일치: {sum(r['terminal_correct_nav'] for r in compact)}/211. 이는 종료 지점 안내의 정확성 검증이며 전 경로 안내가 항상 충분하다는 증명은 아니다.",
        f"- 상류edge 진입 기록에서 준비 차로 역추적 실패(ambiguous=True)가 관측된 사례: {sum(r['upstream_ambiguous_seen'] for r in compact)}/211. 현재 기본 관측은 이를 별도 유효성 값으로 전달하지 않고 준비 방향0으로 인코딩한다. 상류 차로에서 앞으로 차로변경 없이 이어지는 경로만 역추적하므로, 합류 후 차로변경이 필요한 경우에도 빈 집합이 된다.",
        f"- 마지막10초 동안 도로 바깥의 존재하지 않는 차로 요청이 한 번 이상 나온 사례: {sum(r['terminal_outside_lane_requests']>0 for r in compact)}/211. 이런 요청은 환경에서 거부되지만 올바른 방향 요청이 막힌 것과는 다르다.",
        f"- 실패한 종료edge 중 명시적 changeLeft/changeRight 제약을 가진 사례: {sum(r['edge_has_change_restriction'] for r in compact)}/211. 환경의 고정 실선 차로변경 차단이 해당 끝점 실패의 공통 원인인지는 이 수치와 전달 기록으로 판단한다.",
        '- 명령 전달 성공은 실제 차로변경 완료와 다르다. 실제 변경은 같은edge의 이전/이후 lane index로 따로 계산했다. 경계 통과 중 변화는 이 집계에 포함하지 않았다.',
        '- 행동 집계 범위는 재현한 종료edge 진입부터 원래 위치에서10초 정지할 때까지다. 원래10,800초 전체 행동을 기록한 것은 아니므로 이후 모든 순간에도 같은 명령이었다고 확대 해석하지 않는다.',
        '- 막힘 벌점 제거/최종/11번/seed19102는900초에는206426298에서 멈췄지만 약2705초에 벗어났다. 이후844484877의 비연결 차로 끝에 도달해2732.2초에 원래 종료상태가 재현됐다. 일시 정지를 곧바로 최종 실패로 판단하면 안 되는 사례다.',
        '- 올바른 방향을 한 번 요청했다고 충분히 일찍 준비한 것은 아니다. 요청했지만 이동하지 못한 사례는 남은 거리·요청 시점·SUMO 상태를 개별 기록에서 추가 구분해야 하며 전부 정체로 부르지 않는다.',
        '- 무교통에서 앞차 없이 지속 감속으로 멈춘 사례는 차로 선택 실패와 별도다. 정지 상태의 실제 보상과 가감속 명령을 함께 기록했다.',
        '- 대표 기준 최종/교통/13번/seed19101: 170059996 진입 시 분기까지243.0m지만 준비 대상이 빈 집합이다. 이후170059994에3번 차로로 진입하면 필요한0·1·2번 차로 안내는 정상(-1/3)이나 정책은 반대 방향(+1)을 요청한다. 마지막에는 존재하지 않는4번 차로 요청이 거부되고0속도·0보상으로 남는다. 상류 안내의 한계와 정책의 방향 선택 실패가 함께 나타난다.',
        '- 현재 보상은 앞차가 없는 정지에0점이며 경로 차로를 놓친 상태·불가능한 차선변경 요청·시간초과에 별도 벌점이 없다. 이는 코드와 실제 정지 보상 기록으로 확인했다. 이 설계가 학습 실패의 유일한 원인이라는 인과 결론은 아직 아니며 계수 비교가 필요하다.',
        '- 현재 기준은 seed11로 새로 학습한20만스텝 정책이다. 과거 무교통에서 높은 완주율을 보였던60만스텝 정책과 같은 모델이 아니다. 학습량·seed·환경이 달라 성능차를 보상 하나의 효과로 설명할 수 없다.',
        '- 같은768회 평가 묶음에서16경로 모두 성공 사례가 있다(13·14번은 각각6회). 따라서 모든 정책이 통과할 수 없는 경로 단절로 설명할 수는 없다. 짧은 구간이 요구하는 난이도와 사전 준비 정보는 별도 문제다. 원래 평가의 텔레포트·강제보정은 모두0건이다.',
        '', '## 다음 검증 우선순위','',
        '1. 경로를 놓치는 주 구간에서 목표 방향 안내와 실제 선택의 불일치를 줄이는 독립 실험. 현재 세 안전 보상은 목적지 진행 자체를 직접 보상하지 않는다.',
        '2. 51.07m 연결부처럼 짧은 구간에서는 이전edge부터 준비하도록 안내가 전달되는지 진입 기록을 확인하고, 안내 확장과 보상 변경은 각각 분리한다.',
        '3. 정책이 멈추면0점인 상태와, 주행 중 충돌 위험으로 음의 보상을 받는 상태를 비교한다. 정지 정책을 무조건 정체로 처리하지 않는다. 계수 문제는 별도 단독 실험으로 검증한다.',
        '4. 기준 정책 포함 충분한 학습량·반복seed 비교가 필요하다. 단일seed20만스텝으로 추가 기능의 유효성을 최종 판정하지 않는다.',
        '', '## 재현 자료','',
        '- 실행: `python -m utils.analyze_trial_timeouts` / 집계: `python -m utils.summarize_trial_timeouts`',
        '- 원시재현: 로컬 `results/observation_reward_timeout_audit_20261114/cases/` (정책hash,학습commit,진입상태,마지막60초 행동 포함)',
        '- 원래평가: 로컬 `results/observation_reward_trials_20261114/runs/*/policy-*-evaluation.json`',
        '- 지도·보상·학습 정책을 변경하지 않았으며 원래 실패 결과는 보존했다.']
    dest.write_text('\n'.join(lines)+'\n',encoding='utf8')
    t.save(OUT/'summary.json',dict(total=211,reproduced=sum(r['reproduced'] for r in compact),categories=dict(counts),final_reference=78))
    print(json.dumps(t.read(OUT/'summary.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
