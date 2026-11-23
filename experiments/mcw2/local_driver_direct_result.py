"""Freeze original-branch -> PPO candidates on development seeds; test on new traffic seeds."""
from . import local_driver_ppo as d
from collections import Counter
import json,random,sys
import math
D=d.D
def tasks_fresh():
    seed=1430000000
    ids=iter(random.Random(seed).sample(range(seed,seed+1000000),1600))
    return [(rt,next(ids)) for _ in range(100) for rt in d.m.ROUTES]
def run():
    for family in ['direct','latest']:d.wait(D/(family+'_dev_best.json'))
    families=['saved','direct','latest']
    choices=[d.read(D/(f+'_dev_best.json')) for f in families]
    def sc(z):
        c=z['counts'];return c['arrival']>=380 and c.get('collision',0)<20,c.get('collision',0)<20,c['arrival'],-c.get('collision',0)
    for kind in ('gap','gaplatest','merge'):
        d.wait(D/(kind+'_dev_best.json'));families.append(kind);choices.append(d.read(D/(kind+'_dev_best.json')))
    winner=max(choices,key=sc)
    nm=winner['name'];p=d.model(nm);scores=winner['counts']
    plan=tasks_fresh();assert len(set(seed for _,seed in plan))==1600
    manifest=dict(name=nm,choices=choices,sha256=d.sha(p),seeds=[list(q) for q in plan])
    freeze_file=D/'original_only_freeze.json'
    if freeze_file.exists():assert d.read(freeze_file)==manifest
    else:d.x.write_json(freeze_file,manifest)
    final=D/(nm+'_original_holdout.json')
    if not final.exists():
        with d.ProcessPoolExecutor(6,initializer=d.setup,initargs=(p,)) as pool:
            rows=[]
            for row in pool.map(d.drive,plan):
                rows.append(row)
                if len(rows)%320==0:print(nm,'original holdout',len(rows),flush=True)
        d.x.write_json(final,dict(rows=rows,counts=Counter(r['outcome'] for r in rows),sha256=d.sha(p)))
    data=d.read(final)
    assert [(r['route'],r['seed']) for r in data['rows']]==plan
    assert all(100==sum(r['route']==f'route_{i:02}' for r in data['rows']) for i in range(1,17))
    assert data['sha256']==d.sha(p) and Counter(r['outcome'] for r in data['rows'])==data['counts']
    print(nm,data['counts'],flush=True)
    c=data['counts'];k=c['collision'];n=1600
    z=1.96;pct=k/n;centre=(pct+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(pct*(1-pct)/n+z*z/(4*n*n))/(1+z*z/n)
    lines=['# 기존 mcw2 가중치에서 DAgger 추가 없이 PPO만 학습','',
           f'개발 400회에서 선정한 모델 {nm}의 별도 1,600회(경로당 100회) 결과는 완주 {c["arrival"]:,}/1,600 ({100*c["arrival"]/n:.2f}%), 충돌 {k}/1,600 ({100*k/n:.3f}%), 시간초과 {c.get("timeout",0)}회다. 충돌률의 참고용 95% Wilson 구간은 {100*(centre-half):.2f}–{100*(centre+half):.2f}%다.','',
           'saved와 direct의 출발점은 안건 2의 DAgger 직후 체크포인트 f2126af64a90이다. latest는 이미 DAgger와 가중치 혼합, 가림 적응 PPO를 거친 팀 브랜치 최신 체크포인트 4dab15658aa3에서 바로 시작했다. 세 계열 모두 이 로컬 실험에서 DAgger나 교사 라벨을 추가하지 않았다.','',
           'saved는 브랜치의 기본 PPO를 사용했다. direct와 latest는 가감속과 공통층을 고정하고 차선 선택 출력층·가치 추정층을 PPO로 업데이트했다. 훈련 시 좁은 간격의 차선 변경과 충돌을 감점하고, 안전하게 실제 이동해 경로 연결성이 개선될 때 가점을 줬다. 5개 어려운 경로의 훈련 비중도 높였다. 평가에는 원래 mcw2의 관측과 네트워크만 사용했다. 여기에는 경로를 따라 차선이 120m 안에서 얼마나 이어지는지가 포함된다.','',
           '## 같은 개발 시드 400회','', '| 출발점 / 학습 | PPO 스텝 | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|---:|']
    for f,label in [('saved','DAgger 직후 기준선'),('mentor','브랜치 최신 기준선')]:
        q=d.read(D/(f+'_dev.json'))['counts'];lines.append(f"| {label} | 0 | {q['arrival']} | {q.get('collision',0)} | {q.get('timeout',0)} |")
    for name,z0 in zip(families,choices):
        chk=z0['name'];tr=d.read(D/(chk+'_train.json'));q=z0['counts']
        parent=d.model('gap_4') if name=='merge' else d.SOURCE if name in ('latest','gaplatest') else d.BC
        assert tr['parent_sha256']==d.sha(parent) and tr['sha256']==d.sha(d.model(chk))
        if name=='merge':assert d.read(D/'gap_4_train.json')['parent_sha256']==d.sha(d.BC)
        assert not (set(sd for _,sd in plan)&set(sd for _,sd in tr['episodes']))
        steps=tr['steps']+(8192 if name=='merge' else 0)
        lines.append(f"| [{chk}]({chk}_dev.json) | {steps:,} | {q['arrival']} | {q.get('collision',0)} | {q.get('timeout',0)} |")
    if 'gap' in families:
        lines+=['','gap은 같은 저장 DAgger 원본에서, gaplatest는 팀 최신 체크포인트에서 시작해 PPO의 차선 선택뿐 아니라 공통층도 학습했다. 정책이 직접 샘플링한 행동에 대해 즉시 위험 감점과 장기 주행 결과의 이득(advantage)을 따로 계산한 뒤 PPO의 클립 목적함수를 적용했다. 초기 정책에서 가감속과 차선 선택 확률이 급격히 달라지는 것을 제한했다. [해당 PPO 구현](../../experiments/mcw2/local_driver_gap_ppo.py)']
    if 'merge' in families:
        lines+=['','merge는 같은 DAgger에서 시작한 gap의 8,192스텝 체크포인트를 이어서, 분기 전 연결성이 줄어든 상태에서 안전하게 진로를 확보하는 보상을 높였다. 즉시 보상의 차선별 PPO 목적함수를 세 행동의 기존 확률로 평균해 드문 행동의 학습 신호를 보강했다. 이 보상 계산은 훈련에만 사용한다.']
    lines+=['','## 최종 경로별 결과','', '| 경로 | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|']
    for i in range(1,17):
        split=Counter(r['outcome'] for r in data['rows'] if r['route']==f'route_{i:02}')
        lines.append(f"| {i:02} | {split['arrival']} | {split['collision']} | {split['timeout']} |")
    impl='local_driver_gap_ppo.py' if nm.startswith(('gap','merge')) else 'local_driver_lane_ppo.py' if nm.startswith(('direct','latest')) else 'local_driver_ppo.py'
    lines+=['',f'[선정 가중치]({nm}.pt) · [최종 주행 기록]({final.name}) · [동결 기록](original_only_freeze.json) · [선정 모델의 PPO 훈련 코드](../../experiments/mcw2/{impl})']
    (D/'original_only.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':run()
