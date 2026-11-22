"""Generate final C report from held-out episodes with cross-condition start audits."""
import csv,json,itertools
from collections import Counter,defaultdict
from pathlib import Path
o=Path('results/experiment_c_v1')
design=json.loads((o/'design.json').read_text())
A=design['arms']; S=design['train_seeds']; T=['none','low','high_real_ratio']
pairdefs=design['pairs']
def load(p):return json.loads(p.read_text())
def write(name,rows):
 with (o/name).open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,rows[0]);wr.writeheader();wr.writerows(rows)
def fmt(x,n):return f'{x}/{n} ({100*x/n:.1f}%)' if n else '0/0'
lines=['# 실험 C 최종 보고서','','12개 PPO 모델의 고교통 100만 스텝 학습. 중간 평가 756조건 × 32회, 미사용 시드 검증 60조건 × 128회. 대표와 최종 가중치가 같으면 주행을 한 번만 계산했다.']
def table(title,heads,rows):
 lines.extend(['','## '+title,'','|'+'|'.join(heads)+'|','|'+'|'.join(['---']*len(heads))+'|'])
 lines.extend('|'+'|'.join(map(str,row))+'|' for row in rows)
stages={'none':'무교통','low':'저교통','high_real_ratio':'고교통'}
g=defaultdict(Counter); per=[]; casepairs=[]; drop=defaultdict(Counter); lane=defaultdict(Counter); cyc=defaultdict(Counter); scenes=[]; root_starts={}
for part,n,base in [('select',32,design['select_base']),('hold',128,design['holdout_base'])]:
 files=list(o.glob(f'C*_*/{part}_*.json'))
 assert len(files)==(756 if part=='select' else 60)
 for f in files:
  stage=f.stem.split('_',2)[2];rows=load(f)
  starts={(r['route'],r['sim_seed']):r['start'] for r in rows}
  assert len(starts)==len(rows)==n and set(s for _,s in starts)=={b+l for b in base for l in (0,1)}
  assert starts==root_starts.setdefault((stage,part),starts)
for a,s in itertools.product(A,S):
 d=o/f'{a}_{s}';sel=load(d/'selection.json');best=sel['selected']
 for model,step in [('대표',best),('100만',1000000)]:
  for t in T:
   rows=load(d/f'hold_{step}_{t}.json');key=(model,t,a); k=Counter(); lookup={(r['route'],r['sim_seed']):r for r in rows}
   for pair in pairdefs:
    ok=0
    for sim in {r['sim_seed'] for r in rows}:
     ra,rb=[lookup[route,sim] for route in [pair['exit'],pair['main']]]
     assert ra['start']==rb['start']
     ok+=int(all(x['first']['actual']==x['first']['expected'] for x in [ra,rb]))
    k['pair']+=ok
    if (pair['exit'],pair['main'])==('route_13','route_16'):k['p13']=ok
    casepairs.append(dict(model=model,condition=a,seed=s,step=step,traffic=t,exit=pair['exit'],main=pair['main'],success=ok,n=8,pct=100*ok/8))
   for r in rows:
    k[r['outcome']]+=1;k['n']+=1
    x=r['first']; passed=bool(x and x['actual']==x['expected'])
    if passed:k['after_'+r['outcome']]+=1
    for road,ds in r['drops'].items():
     q=drop[key,road];q['n']+=1
     for field in ['needed','passed','wrong_end']:q[field]+=ds[field]
     if not ds['passed']:q[r['outcome']]+=1
    for fx in r['forks'].values():
     if not fx.get('initially_ready') or not fx['wrong_entries']:continue
     ev=[z for z in r['reward_events'] if z['edge']==fx['edge']]
     for label in ['all']+(['1063527748'] if fx['edge']=='1063527748' else []):
      q=cyc[key,label];q['out']+=fx['wrong_entries'];q['back']+=fx['returns']
      for z in ev:
       if z['lane'] not in fx['targets'] and z['after'] in fx['targets'] and z['prep']:
        q['paid']+=1;q['amount']+=z['prep']
    if r['route'] in ('route_13','route_16'):
     q=lane[key,r['route'],r['lane_case']];q['pass']+=passed;q['n']+=1;q[r['outcome']]+=1
     if not passed and r['route']=='route_16' and r['lane_case']==0:
      q['fail']+=1;q['left']+=r['actions16'].get('1',0);q['changes']+=r['actions16'].get('actual',0)
      if t==T[-1]:
       scenes.append(dict(model=model,condition=a,seed=s,traffic_seed=r['sim_seed'],outcome=r['outcome'],left=r['actions16'].get('1',0),changes=r['actions16'].get('actual',0),samples=json.dumps(x['samples'])))
   assert k['arrival']+k['timeout']+k['collision']==128
   g[key].update(k)
   per.append(dict(model=model,condition=a,seed=s,step=step,traffic=t,pair=fmt(k['pair'],64),pair13=fmt(k['p13'],8),arrival=fmt(k['arrival'],128),collision=fmt(k['collision'],128),timeout=fmt(k['timeout'],128),after_first_collision=k['after_collision'],after_first_timeout=k['after_timeout']))
write('holdout_by_seed.csv',per);write('eight_pairs.csv',casepairs);write('route16_failed_commands.csv',scenes)
table('고교통에서 출발 차로별 첫 분기 통과와 전체 완주',['모델','조건','13번 1차로','16번 0차로','13번 0차로','16번 1차로'],[[m,a,*[f"{v['pass']}/12 · {v['arrival']}/12" for v in [lane[(m,T[-1],a),r,l] for r,l in [('route_13',1),('route_16',0),('route_13',0),('route_16',1)]]]] for m,a in itertools.product(['대표','100만'],A)])
lines.append('각 셀은 첫 분기 통과 · 전체 완주 순이다. 각 조건은 3개 학습 시드 × 4개 배경 교통 시드다.')
table('고교통 일반 차로 단절',['모델','조건','도로','접근/전체','변경 필요/접근','통과/접근','잘못된 끝 정지/접근','통과 전 충돌/접근','통과 전 시간초과/접근'],[[m,a,road,fmt(q['n'],384),*[fmt(q[f],q['n']) for f in ('needed','passed','wrong_end','collision','timeout')]] for m,a,road in itertools.product(['대표','100만'],A,['844484877','206426298','86198708']) for q in [drop[(m,T[-1],a),road]]])
lines.append('잘못된 끝 정지는 에피소드 중 1회 이상이며 이후 차로를 바꿔 통과한 경우도 포함한다.')
table('교통 수준별 최종 검증',['모델','교통','조건','8개 쌍','13·16번만','완주','충돌','시간초과','첫 분기 성공 후 충돌/시간초과'],[[m,stages[t],a,fmt(q['pair'],192),fmt(q['p13'],24),*[fmt(q[f],384) for f in ['arrival','collision','timeout']],f"{q['after_collision']}/{q['after_timeout']}"] for (m,t,a),q in sorted(g.items())])
table('조건 간 차이 (%p, 동일 출발 사례)',['모델','교통','비교','쌍 성공','완주','충돌','시간초과'],[[m,stages[t],a+'−'+b,*[f"{100*(g[m,t,a][f]-g[m,t,b][f])/n:+.1f}" for f,n in [('pair',192),('arrival',384),('collision',384),('timeout',384)]]] for m,t in itertools.product(['대표','100만'],T) for a,b in zip(A[1:],A)])
table('고교통에서 적합 차로로 접근한 뒤 이탈·복귀',['모델','조건','범위','이탈','복귀','준비 보상 받은 복귀','복귀 보상 합계'],[[m,a,label,q['out'],q['back'],q['paid'],round(q['amount'],2)] for m,a,label in itertools.product(['대표','100만'],A,['all','1063527748']) for q in [cyc[(m,T[-1],a),label]]])
table('고교통 16번 0차로 출발 실패 전 구간',['모델','조건','실패/전체','실패 주행의 좌측 명령 총합','실제 변경 총합'],[[m,a,fmt(q['fail'],12),q['left'],q['changes']] for m,a in itertools.product(['대표','100만'],A) for q in [lane[(m,T[-1],a),'route_16',0]]])
table('팀 목표',['모델','조건','완주+충돌','100%까지 차이','충돌률','5% 기준 초과분'],[[m,a,fmt(q['arrival']+q['collision'],384),f"{100*q['timeout']/384:.1f}%p",fmt(q['collision'],384),f"{100*q['collision']/384-5:+.1f}%p"] for m,a in itertools.product(['대표','100만'],A) for q in [g[m,T[-1],a]]])
lines+=['','## 실패 장면 예시','']
for a in A:
 z=next(r for r in scenes if r['condition']==a and r['model']=='100만')
 line=[f"{a} 학습 시드 {z['seed']}, 교통 시드 {z['traffic_seed']}, {z['outcome']}, 전체 왼쪽 명령 {z['left']}회"]
 for t,v in json.loads(z['samples']).items():
  if isinstance(v,dict):
   gaps=['/'.join('미검출' if x is None else f'{x[0]:.1f}m' for x in v['adjacent'][side]) for side in ['1','-1']]
   line.append(f"{v['distance']:.1f}m 전 {v['lane']}차로, 좌·우 앞/뒤 {gaps}, 가감속 {v['accel']:.2f}, 명령 {v['command']}, 실제 변경 {int(v['actual_change'])}, 준비 보상 {v['prep']:.2f}")
  else:line.append(t+'m '+str(v))
 lines.append('- '+' · '.join(line))
lines+=['','## 해석','','고교통 100만 모델에서 시간초과 처리 수정(C1−C0)은 충돌을 63→32건으로 줄이고 완주를 200→220건으로 늘렸다. 대표 모델에서는 충돌이 60→51건으로 줄었으나 완주도 254→191건으로 줄었다. 수정은 경계 처리의 정합성을 회복했지만 모든 지표의 개선을 보장하지 않았다.',
'새 보상 묶음(C2−C1)은 고교통 대표 모델의 16번 0차로 복귀를 0→5/12로 개선했으나 13번 1차로 진출은 11→9/12, 전체 쌍은 119→105/192였다. 100만에서는 복귀 개선이 0→1/12에 그쳤다. 통과 보너스와 준비 보상을 분리한 실험은 아니다.',
'400m 물리적 지도 관측(C3−C2)은 고교통 100만 모델에서 전체 쌍 87→119/192, 13번 1차로 진출 8→10/12, 16번 0차로 복귀 1→3/12로 개선됐다. 13번 0차로를 유지한 12/12가 정상 진출했으므로 오른쪽 끝 차로를 언제나 회피했다고 볼 수 없다. 16번 1차로 본선 유지는 8→7/12였고 C3에서 13번은 두 출발 차로 모두 전체 완주가 0이었다. 대표 모델에서는 C2→C3의 16번 0차로 복귀가 5→3/12로 줄었다.',
'400m 예고를 추가한 100만 모델의 세 단절 도로 통과는 C2 88/92에서 C3 103/117로 바뀌었다. 더 많이 도달하고 통과했지만 도달한 경우의 통과율은 95.7%→88.0%로 낮아졌다. 접근 집단이 달라 동일한 도로 진입 상태에서의 인과 효과로 해석할 수 없다.',
'지급 이력의 수명은 400m 준비 구간에서 시작한다. 고교통 C2·C3의 복귀 지급 사례에서 차량은 약 438~483m 전에 적합 차로를 벗어난 뒤 활성 구간에 부적합 차로로 들어갔다. C2에서는 371m 전 또는 차로 끝, C3에서는 차로 끝에서 복귀해 +0.3을 받았다. 구간 안의 중복 억제는 통과했으나 도로 진입 이후 모든 불필요한 왕복을 차단하지는 못했다.',
'분기 쌍은 본선 경로 일부를 공유한다. 무교통·저교통·고교통 결과와 시드별 분산을 함께 보아야 한다. primary 선정 기준은 고교통 쌍 성공이고 시드별 1등 체크포인트를 고른 뒤 별도 검증했다.','',
'## 검증 및 파일','','동일 시드의 네 학습 조건에서 공통 에피소드 3,050·2,912·2,858개의 경로, 교통 시드, 실제 출발 조건이 일치했다(training_audit.json). 모든 중간·검증 파일의 사례별 실제 출발과 시드가 동일 교통 수준 내 다른 가중치와 일치함을 확인했다. 체크포인트·고정 행동 검증·원본 코드 해시는 design.json, source_manifest.json, source_snapshot.zip에 있다. 대표 선정 점수는 각 모델의 selection.json에 있다.',
'holdout_by_seed.csv는 학습 시드별 결과, eight_pairs.csv는 경로 쌍별 건수와 비율, checkpoint_metrics.csv는 전체 체크포인트 지표다. lane_drops.csv, first_fork_lanes.csv, ready_lane_excursions.csv, holdout_failures.csv, route16_failed_commands.csv에서 상세 사례를 확인할 수 있다.']
(o/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('verified',len(per),len(casepairs),len(scenes))
