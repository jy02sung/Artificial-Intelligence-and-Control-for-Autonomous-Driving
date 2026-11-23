"""Verify the final pair and include route counts and the exact frozen lineage in the local report."""
import hashlib,json,re,time
from collections import Counter
from pathlib import Path
D=Path(__file__).resolve().parents[2]/'results'/'local_mcw2_driver_ppo'
def read(n):return json.loads((D/n).read_bytes())
def wait(p):
    while not p.exists():time.sleep(2)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    for filename in ('original_only.md','paired_effect.md'):wait(D/filename)
    e=read('paired_effect.json');base,nn=e['parent'],e['model'];freeze=read('original_only_freeze.json')
    first,second=read(base+'_paired_holdout.json'),read(nn+'_original_holdout.json')
    checked=[(r['route'],r['seed']) for r in first['rows']]
    assert [(r['route'],r['seed']) for r in second['rows']]==checked and len(set(s for _,s in checked))==1600
    assert dict(Counter(r['outcome'] for r in first['rows']))==e['before'] and dict(Counter(r['outcome'] for r in second['rows']))==e['after']
    assert second['sha256']==digest(D/(nn+'.pt'))==freeze['sha256']
    training=[]
    nm=nn
    while True:
        t=read(nm+'_train.json');training.extend(t['episodes']);assert t['sha256']==digest(D/(nm+'.pt'))
        parent=Path(t.get('parent',D.parent/'mcw_runs'/'mcw2_agenda2_dagger_v2'/'best_model.pt'))
        parent=(D.parents[1]/parent).resolve()
        assert t['parent_sha256']==digest(parent)
        if parent.parent!=D:break
        nm=parent.stem
    assert digest(parent)==first['sha256']
    eval_seeds={s for _,s in checked}
    for stage in ('screen','dev'):
        assert not eval_seeds&{r['seed'] for r in read(nn+'_'+stage+'.json')['rows']}
    assert not eval_seeds&{s for _,s in training}
    assert not eval_seeds&set(range(12001,12500)) and not eval_seeds&set(range(40001,41000))
    lines=['## 경로별 짝 비교','',f'16개 경로를 각각 100회 주행했다. 전후에 같은 교통 및 센서 시드를 썼다. 출발 가중치 SHA-256은 {first["sha256"][:12]}, 동결한 PPO 가중치는 {second["sha256"][:12]}이다.','',
           '| 경로 | 완주 전→후 | 충돌 전→후 | 시간초과 전→후 |','|---|---:|---:|---:|']
    for idx in range(1,17):
        rt=f'route_{idx:02}';ca,cb=[Counter(r['outcome'] for r in z['rows'] if r['route']==rt) for z in (first,second)];assert sum(ca.values())==sum(cb.values())==100
        lines.append('| '+' | '.join([f'{idx:02}',*[f"{ca[k]} → {cb[k]}" for k in ('arrival','collision','timeout')]])+' |')
    doc=D/'paired_effect.md';s=doc.read_text(encoding='utf8').split('\n## 경로별 짝 비교')[0].rstrip()
    doc.write_text(s+'\n\n'+'\n'.join(lines)+'\n',encoding='utf8')
    for nm in ('original_only.md','paired_effect.md'):
        for link in re.findall(r'\]\(([^)]+)\)',(D/nm).read_text(encoding='utf8')):assert (D/link).is_file(),link
    print('Verified',nn,e['before'],e['after'],e['changes'],flush=True)
