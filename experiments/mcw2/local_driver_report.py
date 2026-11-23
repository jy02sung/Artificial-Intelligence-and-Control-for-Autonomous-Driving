"""Cross-check ancestry, distinct seeds, paired scores and the frozen holdout."""
import hashlib,json,math,re
from collections import Counter
from itertools import combinations
from pathlib import Path
from .local_driver_ppo import D,R,BC,SOURCE,FRESH,sha,model,tasks,read,parents
def rates(d):
    n=sum(d.values());return [f"{d.get(k,0):,} ({100*d.get(k,0)/n:.2f}%)" for k in ('arrival','collision','timeout')]
def evaldata(k,stage):
    d=read(D/f'{k}_{stage}.json')
    seq=tasks(stage);rows=d['rows']
    assert [(r['route'],r['seed']) for r in rows]==seq
    assert Counter(r['outcome'] for r in rows)==d['counts']
    assert set(d['counts'])<={'arrival','collision','timeout'}
    assert d['sha256']==sha(model(k))
    return d
def report():
    families={'saved':'A. 저장된 DAgger 직후 모델','fresh':'B. 최신 운전자에 DAgger 재적용','lane':'C. B와 같은 DAgger에서 차선 선택만 PPO'}
    selection={k:read(D/f'{k}_dev_best.json')['name'] for k in families}
    frozen=read(D/'recommended_freeze.json');nn=frozen['name'];assert frozen['sha256']==sha(model(nn))
    def key(k):
        c=evaldata(k,'dev')['counts']
        return c.get('arrival',0)>=380 and c.get('collision',0)<20,c.get('arrival',0),-c.get('collision',0)
    assert nn==max(selection.values(),key=key)
    holdout=evaldata(nn,'final');a,c,t=rates(holdout['counts'])
    seeds=[set(sd for _,sd in tasks(s)) for s in ('screen','dev','final')]
    training=set()
    for k in parents:
        for i in (2,6,12):
            nm=f'{k}_{i}';d=read(D/f'{nm}_train.json');assert d['sha256']==sha(model(nm)) and d['parent_sha256']==sha(parents[k])
            training.update(s for _,s in d['episodes'])
    for i in (2,8):
        nm=f'lane_{i}';d=read(D/f'{nm}_train.json');assert d['sha256']==sha(model(nm)) and d['parent_sha256']==sha(FRESH)
        assert d['max_lane_weight_change']>0
        training.update(s for _,s in d['episodes'])
    assert all(not a&b for a,b in combinations(seeds+[training],2))
    dr=read(D/'fresh_ready.json');assert dr['mentor_sha256']==sha(SOURCE) and dr['bc_sha256']==sha(FRESH)
    passes=[i for i in (1,2,3) if sha(FRESH)==sha(FRESH.parent/f'model_pass_{i}.pt')]
    # torch.save may vary container bytes by filename; compare tensor contents for the pass.
    if not passes:
        import torch
        state=torch.load(FRESH,map_location='cpu',weights_only=True)
        passes=[i for i in (1,2,3) if all(torch.equal(v,state[k]) for k,v in torch.load(FRESH.parent/f'model_pass_{i}.pt',map_location='cpu',weights_only=True).items())]
    assert len(passes)==1
    n=len(holdout['rows']);p=holdout['counts'].get('collision',0)/n;z=1.96
    centre=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    lines=['# mcw2 운전자 DAgger → PPO 검증','',f'개발 평가로 선택한 {nn}을 고정해 고교통 16개 경로를 100회씩 주행했다. 완주 {a}, 충돌 {c}, 시간초과 {t}. 충돌 비율의 참고용 95% Wilson 구간은 {100*(centre-half):.2f}–{100*(centre+half):.2f}%이다.', '',
    '## 출발점과 훈련','',
    '- A는 팀 브랜치의 마지막 DAgger 단독 저장점(mc w2 안건 2, SHA-256 앞 12자리 f2126af64a90)에서 출발했다. 이 모델은 이미 앞 단계 PPO와 DAgger를 거쳤으며, 직후 스냅샷을 그대로 사용했다.'.replace('mc w2','mcw2'),
    f'- B는 팀의 최신 가림 환경 운전자 4dab15658aa3을 교사·학생의 초기값으로 복사해 원래 distill_safe_policy를 3패스(패스마다 6에포크, 학습률 0.00008) 수행했다. 브랜치 기준으로 선택된 {passes[0]}번째 패스의 가중치를 PPO의 시작점으로 사용했다.',
    '- 팀 교사는 기존 정책의 가감속과 차선 선택을 바탕으로 안전 간격과 경로상의 차선 준비를 반영한 시범 행동을 만든다. 첫 패스는 이 시범 행동으로 주행하고 둘째는 교사와 학생 행동을 섞으며 셋째는 학생이 주행한다. 매 패스 누적된 교사 라벨을 가중 지도학습한다.',
    '- 이어지는 PPO에는 학생 자신의 환경 상호작용을 사용했다. 두 경우 모두 팀의 47차원 관측, 하이브리드 행동, 원래 보상, 고교통(시간당 유입 15,000대 설정), 120m 시야, 가림, 노이즈를 사용했다. 고정된 학습 정책이 평가 행동을 직접 출력하고 자동 속도 제어·차선 변경 안전 필터는 껐다. 주행 시 원래 브랜치의 목적 경로 기준 차선별 120m 연결성 입력이 사용된다.',
    '- A는 PPO 학습률 0.000025, B는 0.000008이었다. PPO는 각각 24,576스텝까지 진행했다. 두 조건은 BC 출발점과 PPO 학습률·시드가 함께 다르므로, A와 B 사이 성능 차이를 단일 요인에 귀속하지 않는다.',
    '- C는 B의 DAgger 직후 가중치에서 가감속·공통 특성층을 고정하고 차선 선택 출력층과 가치 추정층을 업데이트했다. 관측 간격·접근속도에 비해 위험한 차선 변경 시도와 실제 충돌에 훈련 보상 감점을 더했고, 02·05·11·13·14번 경로를 더 자주 훈련했다. 차선 정책의 학습률은 0.00008, PPO 클립 범위는 0.12다. 평가 환경과 모델 구조는 A·B와 같다.',
    '- A·B의 4,096 / 12,288 / 24,576스텝과 C의 4,096 / 16,384스텝을 공통 80개 시드로 선별했다. 각 계열에서 완주 최다·동률 시 충돌 최소인 모델을 BC 직후 모델과 같은 400개 개발 시드에 평가했다. 세 계열 중 전체 완주율 95% 이상·충돌률 5% 미만을 만족하는 후보를 우선하고, 완주 최다·동률 시 충돌 최소 기준으로 모델을 동결했다. 최종 1,600회는 별도의 교통 시드로 진행했다.', '',
    '## 같은 400개 시드에서 본 PPO 추가 학습 효과','',
    '| 계열 | 저장점 | 완주 | 충돌 | 시간초과 |','|---|---|---:|---:|---:|']
    for k,title in families.items():
        pairs=[] if k=='lane' else [(k,'DAgger 직후')]
        for nm,label in pairs+[(selection[k],f"PPO {read(D/(selection[k]+'_train.json'))['steps']:,}스텝")]:
            res=evaldata(nm,'dev')
            lines.append('| '+' | '.join([title,f'[{label}]({nm}_dev.json)',*rates(res['counts'])])+' |')
    standard=read(D/'freeze.json')['name']
    lines+=['','## 동결한 PPO 두 모델 비교','',
        '일반 PPO의 최종 후보를 먼저 고정해 평가했고, 차선만 학습한 계열을 포함한 발표 후보도 개발 400회로 고정했다. 아래 평가는 두 모델에 같은 교통 시드를 사용했다.','',
        '| 모델 | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|']
    for nm in dict.fromkeys((standard,nn)):
        d=evaldata(nm,'final');lines.append('| '+' | '.join([f'[{nm}]({nm}_final.json)',*rates(d['counts'])])+' |')
    lines+=['','## 1,600회에서 확인한 경로별 결과','',f'동결 모델 [{nn}]({nn}.pt) · [개별 주행 로그]({nn}_final.json)','',
        '| 경로 (각 100회) | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|']
    for i in range(1,17):
        rt=f'route_{i:02}';g=Counter(q['outcome'] for q in holdout['rows'] if q['route']==rt);assert sum(g.values())==100
        lines.append(f"| {i:02} | {g['arrival']} | {g['collision']} | {g['timeout']} |")
    lines+=['','[브랜치 원본 학습 코드](../../utils/mcw_train.py) · [일반 PPO 실행 코드](../../experiments/mcw2/local_driver_ppo.py) · [차선 PPO 실행 코드](../../experiments/mcw2/local_driver_lane_ppo.py) · [새 DAgger 계보](fresh_ready.json) · [최종 후보 동결 기록](recommended_freeze.json)']
    (D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for href in re.findall(r'\]\(([^)]+)\)', '\n'.join(lines)):assert (D/href).is_file(),href
    print(nn,holdout['counts'])
if __name__=='__main__':report()
