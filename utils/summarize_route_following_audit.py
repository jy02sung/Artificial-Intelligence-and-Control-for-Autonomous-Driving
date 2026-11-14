"""Evidence report separating official evaluation from shortened diagnostic runs."""
import collections
import csv
from utils.audit_route_following import OUT,t,runner

def main():
    rs=[t.read(p) for p in OUT.glob('*.json') if p.name!='status.json']
    assert len(rs)==96 and all(r['route_overrides']==0 for r in rs)
    names=dict(baseline='기준',guidance='사전 안내 보완',route_reward='목적지 차로 보상')
    dest=t.ROOT/'docs/experiments/ic-route-following-analysis-20261114.md'
    lines=['# 사전 안내와 차로 보상: 관측 활용 및 보상 발생 분석','','## 범위와 한계','',
        '저장된 정책을 수정하지 않고96개 진단 주행을 실행했다. 최종3조건 무교통48회, 보상조건 중간모델 무교통16회, 보상조건 중간/최종 교통각16회(seed19101)다. 진단은10초 같은 위치 정지 또는900초에 끝나므로 원래10,800초 공식 평가와 별개다. 원래 완주율에는 공식 평가만 사용한다. 진단의 일시정지를 공식 시간초과로 바꾸지 않는다.',
        '', '학습 중 보상 성분별 횟수는 저장하지 않았다. 아래 보상 발생 수치는 저장 모델의 진단 주행에서 새로 계측한 것이며 학습 중 정확한 발생 횟수/합계는 복원할 수 없다.',
        '', '## 공식 최종 평가','','|조건|교통 완주|충돌|기타|무교통 완주|','|---|---:|---:|---:|---:|']
    for c in names:
        p=runner.OUT/'runs'/c/'policy-200000-evaluation.json'
        if not p.exists():lines.append(f'|{names[c]}|평가 중|—|—|—|');continue
        rows=t.read(p);ref=[r for r in rows if r['traffic']=='reference'];none=[r for r in rows if r['traffic']=='none']
        counts=collections.Counter(r['outcome'] for r in ref)
        lines.append(f"|{names[c]}|{counts['arrival']}/48|{counts['collision']}|{48-counts['arrival']-counts['collision']}|{sum(r['outcome']=='arrival' for r in none)}/16|")
    lines+=['','## 무교통에서는 왜 멈췄나','','|조건|진단 정지 경로|직접 관측한 행동|','|---|---|---|']
    for c in names:
        group=[r for r in rs if r['condition']==c and r['checkpoint']==200000 and r['traffic']=='none' and r['outcome']=='stationary_10s']
        endstop=[];brake=[]
        for r in group:
            x=r['tail'][-1]
            if x['speed']<.05 and x['action'][0]<0 and x['remaining']>1:brake.append(r['route'])
            else:endstop.append(r['route'])
        if endstop:lines.append(f"|{names[c]}|{', '.join(endstop)}|경로 비연결 차로 끝에서 유지 또는 반대방향 요청|")
        if brake:lines.append(f"|{names[c]}|{', '.join(brake)}|앞차 없이 감속 명령을 계속 내며 도로 중간 정지|")
    lines+=['','예: 기준04·13·14번은 오른쪽 변경이 필요한데 마지막 상태에서 좌61.6%/유지16.8%/우21.5%로 왼쪽을 선택했다. 보상조건도 같은 지점에서 좌49.9%/유지36.9%/우13.1%로 잘못된 왼쪽을 선택했다. 안내보완02번은 왼쪽 변경이 필요한데 좌3.8%/유지56.0%/우40.2%라 유지한다. 실제3분류 확률이며 연속출력이 차선변경 임계치에 못 미친 문제가 아니다.',
        '', '## 관측을 전혀 안 쓰는가','','주행 중 잘못된 차로에 있는 표본을1초 간격으로 추출했다. 같은 물리상태에서 경로 채널을 가린 입력의 정책 출력만 계산하고 실제 행동은 바꾸지 않았다. 가린 입력은 학습분포 밖일 수 있으므로 민감도 검사이며 인과 효과/새 평가점수는 아니다. 조건마다 방문상태가 달라 조건 간 확률을 직접 성능순위로 비교하지 않는다.',
        '', '|최종 모델|표본|안내 있음: 필요방향 평균확률|안내 가림|argmax 변경 표본|','|---|---:|---:|---:|---:|']
    for c in names:
        probes=[x for r in rs if r['condition']==c and r['checkpoint']==200000 and r['traffic']=='none' for x in r['sensitivity']]
        n=len(probes)
        lines.append(f"|{names[c]}|{n}|{100*sum(x['correct_probability'] for x in probes)/n:.1f}%|{100*sum(x['hidden_correct_probability'] for x in probes)/n:.1f}%|{sum(x['choice_changed'] for x in probes)}/{n}|")
    lines+=['','안내가 출력에 영향을 주는 증거는 있다. 하지만 필요한 방향이 최댓값이 될 만큼 학습됐다는 뜻은 아니다. “관측을 전혀 못 읽는다”보다는 “일부 반응하지만 올바른 행동 선택으로 충분히 연결되지 않는다”가 현재 근거에 맞다.',
        '', '## 보상은 실제로 얼마나 발생했나','','각 행은seed19101·16경로 진단구간 합계다. 교통 진단은 이른 정지에서 종료될 수 있으므로 공식48회 평가 전체 합계와 혼동하지 않는다.',
        '', '|모델|교통|차로 접근 보상 횟수/합계|유지보상 합계|이탈 횟수/벌점|추가보상 순합계|','|---|---|---:|---:|---:|']
    for step in (100352,200000):
        for traffic in ('none','reference'):
            g=[r for r in rs if r['condition']=='route_reward' and r['checkpoint']==step and r['traffic']==traffic];count=collections.Counter()
            for r in g:count.update(r['metrics'])
            lines.append(f"|{step}|{traffic}|{count['selection_events']}/{count['selection_total']:.3f}|{count['maintenance_total']:.3f}|{count['departure_events']}/{count['departure_penalty']:.3f}|{count['bonus_total']:.3f}|")
    lines+=['','최종 무교통은16회 중12회에서 차로 접근 보상이0회다. 교통 진단은16회 중13회가0회다. 특히13·14번은 상류 차로 유지로각+0.5를 받지만 실제 실패 구간의 차로 접근 보상은0회다. 무교통 양의 추가보상 중 유지 성분 비중은약94.3%,교통 진단에서는약96.9%다.',
        '', '추가보상 자체가 미적용된 버그는 아니다. 어려운 변경을 하지 않으면 접근 보상이 발생하지 않는 희소성, 이미 맞는 차로에서 받는 유지 보상 편중이 확인됐다. +0.1이라는 계수의 크기 부족은 가능한 가설이지만 현재 결과만으로 확정하지 않는다. 발생0인 사례에 계수만5배를 해도 그 상태의 보상은0이다.',
        '', '## 다음 실험','',
        '[다음 독립 실험 설계](ic-route-following-next-20261114.md): 무교통 기준부터 다시 검증하고, 간결한 안내/기존선택보상/보상크기5배/잘못된차로지속벌점을 분리한다. 학습 중 성분별 발생 횟수를 반드시 저장한다. 새 학습은 이 보고서 작성 과정에서 시작하지 않았다.',
        '', '원시 진단: 로컬 `results/route_following_audit_20261114/`. 정책sha와학습commit, 마지막20초 입력·확률·행동, 입력가림 검사와 보상 분해를 보존했다.']
    dest.write_text('\n'.join(lines)+'\n',encoding='utf8')
    with dest.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        data=[dict(condition=r['condition'],checkpoint=r['checkpoint'],route=r['route'],traffic=r['traffic'],outcome=r['outcome'],seconds=r['seconds'],**r['metrics']) for r in rs]
        fields=sorted({k for row in data for k in row});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    print('Saved 96 diagnostic records and analysis report')

if __name__=='__main__':main()
