"""Re-run the exact pre-PPO weights on the frozen PPO traffic seeds."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json,math,sys
import numpy as np
from . import local_driver_ppo as d
from .local_driver_direct_result import tasks_fresh
D=d.D
def precompute(parent):
    name=D/(parent+'_paired_holdout.json');plan=tasks_fresh()
    checkpoint=d.model(parent)
    if name.exists():return
    with ProcessPoolExecutor(5,initializer=d.setup,initargs=(checkpoint,)) as pool:
        rows=[]
        for row in pool.map(d.drive,plan):
            rows.append(row)
            if len(rows)%320==0:print(parent,'baseline',len(rows),flush=True)
    assert [(r['route'],r['seed']) for r in rows]==plan
    d.x.write_json(name,dict(rows=rows,counts=Counter(r['outcome'] for r in rows),sha256=d.sha(checkpoint)))
def run():
    d.wait(D/'original_only_freeze.json')
    frozen=d.read(D/'original_only_freeze.json')
    selected=frozen['name']; t=d.read(D/(selected+'_train.json'))
    origin=t['parent_sha256']
    if selected.startswith('merge_'):
        assert origin==d.sha(d.model('gap_4'));origin=d.read(D/'gap_4_train.json')['parent_sha256']
    parent='saved' if origin==d.sha(d.BC) else 'mentor'
    checkpoint=d.model(parent);assert origin==d.sha(checkpoint)
    tasks=tasks_fresh();assert [list(pair) for pair in tasks]==frozen['seeds']
    dst=D/(parent+'_paired_holdout.json')
    if parent=='saved':d.wait(dst)
    else:precompute(parent)
    d.wait(D/(selected+'_original_holdout.json'))
    ref=d.read(dst);policy=d.read(D/(selected+'_original_holdout.json'))
    assert ref['sha256']==d.sha(checkpoint) and policy['sha256']==frozen['sha256']
    assert [(r['route'],r['seed']) for r in ref['rows']]==[(r['route'],r['seed']) for r in policy['rows']]==tasks
    def changes(key):
        before=np.asarray([r['outcome']==key for r in ref['rows']]);after=np.asarray([r['outcome']==key for r in policy['rows']]);contrast=(after.astype(int)-before.astype(int)).reshape(100,16).T
        rng=np.random.default_rng(4895)
        draws=np.zeros(10000)
        for rt in contrast:draws+=rt[rng.integers(0,100,(len(draws),100))].sum(1)
        return dict(net=int(contrast.sum()),gained=int((contrast==1).sum()),lost=int((contrast==-1).sum()),pp_interval=np.quantile(draws*100/1600,[.025,.975]).tolist())
    diffs={k:changes(k) for k in ('arrival','collision')}
    d.x.write_json(D/'paired_effect.json',dict(before=ref['counts'],after=policy['counts'],parent=parent,model=selected,changes=diffs))
    lines=['# 같은 1,600개 교통 시드에서 측정한 PPO의 기여','',f'출발 가중치 {parent} → PPO {selected}. 두 모델 모두 경로당 100회씩 동일한 교통 시드와 센서 노이즈 시드에서 시작했다. PPO 이후에는 새 DAgger 패스나 교사 라벨을 사용하지 않았다.','',
        '| 결과 | PPO 전 | PPO 후 | 변화(%p) | 경로별 짝 부트스트랩 95% 구간(%p) |','|---|---:|---:|---:|---:|']
    for k,label in [('arrival','완주'),('collision','충돌'),('timeout','시간초과')]:
        x=ref['counts'].get(k,0);y=policy['counts'].get(k,0)
        interval=diffs[k]['pp_interval'] if k in diffs else None
        ci=f'{interval[0]:+.2f} ~ {interval[1]:+.2f}' if interval else '—'
        lines.append(f'| {label} | {x}/1600 ({100*x/1600:.2f}%) | {y}/1600 ({100*y/1600:.2f}%) | {100*(y-x)/1600:+.2f} | {ci} |')
    lines+=['',f"기존 실패에서 완주로 바뀐 주행은 {diffs['arrival']['gained']}회, 기존 완주에서 실패로 바뀐 주행은 {diffs['arrival']['lost']}회였다. 충돌을 피한 주행은 {diffs['collision']['lost']}회, 새로 충돌한 주행은 {diffs['collision']['gained']}회였다.",'',
        '각 구간은 동일 경로의 시드 쌍을 함께 복원 추출(10,000회)해 계산했다. 한 쌍의 고정된 정책과 이 도로망에서의 교통 변동을 나타낸다.', '',
        f'[PPO 전 개별 주행]({dst.name}) · [PPO 후 개별 주행]({selected}_original_holdout.json) · [PPO 전후 변화량](paired_effect.json) · [PPO 설계 및 경로별 결과](original_only.md)']
    (D/'paired_effect.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(ref['counts'],policy['counts'],diffs,flush=True)
if __name__=='__main__':
    precompute(sys.argv[1]) if len(sys.argv)>1 else run()
