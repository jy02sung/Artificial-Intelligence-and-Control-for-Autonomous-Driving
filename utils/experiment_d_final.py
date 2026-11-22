"""Publish matched held-out D comparisons and independent lane-cycle audit."""
import csv,itertools,json
from pathlib import Path
from collections import Counter,defaultdict
o=Path('results/experiment_d_v1')
read=lambda p:json.loads(p.read_text())
arms=('D0','D1'); seeds=(11,22,33); stages=('none','low','high_real_ratio')
models=('대표','100만')
def fmt(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else '0/0'
def write(name,rs):
 if not rs:return
 with (o/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def table(name,cols,rows):
 lines.extend(['','## '+name,'','|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|'])
 lines.extend('|'+'|'.join(map(str,r))+'|' for r in rows)
all_counts=defaultdict(Counter);drops=defaultdict(Counter); lanes=defaultdict(Counter)
seedrows=[];cycles=[];scenes=[];miss=[];training=[];starts={}
for s in seeds:
 logs=[]
 for a in arms:
  d=o/f'{a}_{s}'
  assert len(list(d.glob('policy_*.pt')))==21 and not list(d.glob('episodes_resume*'))
  e=[json.loads(l) for l in (d/'episodes.jsonl').read_text().splitlines()]
  assert all(x[0]==i and x[2]==s*1000000+i for i,x in enumerate(e))
  assert all(len(set(x[1] for x in e[i:i+16]))==len(e[i:i+16]) for i in range(0,len(e),16))
  logs.append(e)
 n=min(map(len,logs));assert logs[0][:n]==logs[1][:n]
 training.append(dict(seed=s,D0=len(logs[0]),D1=len(logs[1]),matched=n))
for m,a,s,t in itertools.product(models,arms,seeds,stages):
 d=o/f'{a}_{s}';step=1000000 if m=='100만' else read(d/'selection.json')['selected']
 rs=read(d/f'hold_{step}_{t}.json');lookup={(r['route'],r['sim_seed'],r['lane_case']):r for r in rs}
 sk={k:v['start'] for k,v in lookup.items()};assert sk==starts.setdefault(t,sk)
 assert len(rs)==len(lookup)==128
 ok=lambda r:r['first']['actual']==r['first']['expected']
 c=Counter(); key=m,a,t
 for sim in {r['sim_seed'] for r in rs}:
  for l in (0,1):assert lookup['route_13',sim,l]['start']==lookup['route_16',sim,l]['start']
  c['joint']+=ok(lookup['route_13',sim,1]) and ok(lookup['route_16',sim,0])
 for r in rs:
  c[r['outcome']]+=1
  first=r['first']
  if first and ok(r):c['post_'+r['outcome']]+=1
  if r['route'] in ('route_13','route_16'):
   q=lanes[key,r['route'],r['lane_case']];q['passed']+=ok(r);q[r['outcome']]+=1
   if not ok(r) and r['route']=='route_16' and r['lane_case']==0:
    q['fails']+=1;q['no_left']+=r['actions16'].get('1',0)==0
    if t==stages[-1]:scenes.append(dict(model=m,arm=a,seed=s,step=step,sim=r['sim_seed'],outcome=r['outcome'],actions=json.dumps(r['actions16']),samples=json.dumps(first['samples'],ensure_ascii=False)))
  for road,z in r['drops'].items():
   q=drops[key,road];q['approach']+=1;q['need']+=z['needed'];q['passed']+=z['passed'];q['need_pass']+=z['needed'] and z['passed'];q['stop']+=z['wrong_end']
  recorded={(z['constraint'][0],z['step']) for z in r['cycles']}
  for z in r['cycles']:
   c['cycles']+=1;c['paid']+=z['prep']>0;c['prep']+=z['prep'];c['outside']+=z['departure']['distance']>400
   cycles.append(dict(model=m,arm=a,seed=s,traffic=t,route=r['route'],sim=r['sim_seed'],start_lane=r['lane_case'],constraint=z['constraint'],departure=z['departure']['distance'],return_distance=z['distance'],paid=z['prep'],source='constraint'))
  for k,f in r['forks'].items():
   away=None
   for ev in r['reward_events']:
    if ev['edge']!=f['edge'] or ev['lane']==ev['after'] or ev['after'] is None:continue
    prev=ev['lane'] in f['targets'];nxt=ev['after'] in f['targets']
    if prev and not nxt:away=ev
    if away and not prev and nxt:
     if (int(k),ev['step']) not in recorded:
      miss.append(dict(model=m,arm=a,seed=s,traffic=t,route=r['route'],sim=r['sim_seed'],start_lane=r['lane_case'],constraint=[k,f['edge'],f['expected']],departure=away['distance'],return_distance=ev['distance'],paid=ev['prep'],source='fork-events'))
     away=None
 assert sum(c[k] for k in ('arrival','collision','timeout'))==128
 all_counts[key].update(c)
 seedrows.append(dict(model=m,arm=a,seed=s,traffic=t,step=step,joint=fmt(c['joint'],4),arrival=fmt(c['arrival'],128),collision=fmt(c['collision'],128),timeout=fmt(c['timeout'],128),paying_cycles=c['paid'],cycles=c['cycles'],after_fork_timeout=c['post_timeout'],after_fork_collision=c['post_collision']))
write('holdout_by_seed.csv',seedrows);write('cycle_scope_audit.csv',cycles+miss);write('failed_route16.csv',scenes);write('matched_training.csv',training)
lines=['# 실험 D 최종 검증','','D0는 C3 보상, D1은 차로 제약의 이력을 일찍 기록한다. 행동·보상 지급 거리·PPO 시간초과 처리는 동일하다. 대표 선정 순서는 고교통 13번 1차로와 16번 0차로의 동일 시드 동시 성공, 낮은 충돌, 높은 완주, 이른 스텝이다. 각 조건과 모델의 고교통 주행은 학습 시드 3개 × SUMO 시드 4개 × 출발 차로 2개 × 경로 16개다. 양방향 점수의 분모는 12다.']
table('모델별 미사용 시드 검증',['모델','보상','교통','양방향 변경 성공','전체 완주','충돌','시간초과','첫 분기 성공 후 충돌·시간초과'],[[m,a,t,fmt(c['joint'],12),*[fmt(c[k],384) for k in ('arrival','collision','timeout')],f"{c['post_collision']} · {c['post_timeout']}"] for (m,a,t),c in all_counts.items()])
table('고교통 출발 차로별 첫 분기 통과와 전체 완주',['모델','조건','13번 1차로','16번 0차로','13번 0차로','16번 1차로'],[[m,a,*[f"{q['passed']}/12 · {q['arrival']}/12" for q in [lanes[(m,a,stages[-1]),r,l] for r,l in [('route_13',1),('route_16',0),('route_13',0),('route_16',1)]]]] for m,a in itertools.product(models,arms)])
table('고교통 일반 단절',['모델','조건','도로','접근/전체','변경 필요/접근','변경 필요 주행 중 통과','전체 접근 중 통과','끝 정지/접근'],[[m,a,e,fmt(q['approach'],384),fmt(q['need'],q['approach']),fmt(q['need_pass'],q['need']),fmt(q['passed'],q['approach']),fmt(q['stop'],q['approach'])] for ((m,a,t),e),q in sorted(drops.items()) if t==stages[-1]])
table('실제 정책의 제약별 왕복',['모델','조건','교통','왕복','400m 밖에서 이탈','보상 받은 복귀','준비 보상 합'],[[m,a,t,c['cycles'],c['outside'],c['paid'],round(c['prep'],2)] for (m,a,t),c in all_counts.items()])
lines+=['',f"독립 대조에서는 같은 분기 도로에서 실제 적합→부적합→적합 이동을 reward_events에서 재구성해 제약 이력의 복귀 시각과 맞췄다. 표에 포함되지 않은 추가 이동은 모델 역할별 합계 {len(miss)}건, 그중 양의 준비 보상이 기록된 이동은 {sum(z['paid']>0 for z in miss)}건이다. 대표와 100만 가중치가 같은 경우 역할별 집계에 중복된다. cycle_scope_audit.csv에서 source로 구별한다.",'']
table('고교통 16번 본선 복귀 실패 진단',['모델','조건','실패/12','전체 접근에서 왼쪽 명령 0회인 실패'],[[m,a,fmt(q['fails'],12),fmt(q['no_left'],q['fails'])] for m,a in itertools.product(models,arms) for q in [lanes[(m,a,stages[-1]),'route_16',0]]])
for a in arms:
 x=next(z for z in scenes if z['arm']==a and z['model']=='100만')
 lines.append(f"- {a} 시드 {x['seed']}, 교통 시드 {x['sim']}, {x['outcome']}.")
 for t,v in json.loads(x['samples']).items():
  if isinstance(v,dict):
   lines.append(f"  - {v['distance']:.1f}m 전 {v['lane']}차로. 왼쪽 앞뒤 간격 {[(p[0] if p else None) for p in v['adjacent']['1']]}m, 가감속 {v['accel']:.2f}, 차로 명령 {v['command']}, 실제 변경 {int(v['actual_change'])}, 준비 보상 {v['prep']:.2f}. None은 미검출이다.")
table('팀 목표 대비 고교통 결과',['모델','조건','완주+충돌','100%까지 차이','충돌률','5% 기준 초과분'],[[m,a,fmt(c['arrival']+c['collision'],384),f"{100*c['timeout']/384:.1f}%p",fmt(c['collision'],384),f"{100*c['collision']/384-5:.1f}%p"] for m,a in itertools.product(models,arms) for c in [all_counts[m,a,stages[-1]]]])
lines+=['','## 해석','','고교통 100만 모델에서 D1은 D0보다 준비 보상을 받은 왕복이 줄었다. 왕복 행동은 남았다. 양방향 변경 성공은 3→0/12, 충돌은 59→85/384였다. 대표 모델은 양방향 성공이 3→4/12, 충돌이 54→42/384였지만 완주가 199→169/384였다. 시드별 결과를 함께 보면 대표 선정과 마지막 스텝의 결과가 다르다. D1의 관측된 효과는 보상 허점 억제이며, 안정적 본선 복귀 개선은 확인되지 않았다.',
'고교통에서 변경이 필요했던 일반 단절 접근의 통과는 대표 D0 33/55, D1 16/48, 100만 D0 44/58, D1 37/54였다. 각 정책이 도달하는 접근 집단은 다르다.',
'13번 적합 차로 출발은 고교통 네 모델 역할 모두 첫 분기 12/12였고, 무교통에서는 네 역할 모두 동일 시드 양방향 성공 0/12였다. 고교통 결과만으로 교통량을 실패 원인으로 특정할 수 없다.',
'','## 단발·지속형 사전시험과 보상 재현','','C3의 고교통 16번 실패 9개 × 300m/100m 두 시점. 각 개입에서 단발·지속 모두 16/18회 적합 차로 이동, 9/18회 본선 통과·완주, 2회 충돌, 7회 시간초과였다. 개입 없는 주행은 18회 모두 시간초과였다. 지속형도 모든 경우 한 번만 명령했다. 두 충돌은 100m 전 왼쪽 앞 또는 뒤 간격이 10m 미만이었던 사례였다. 이 설계의 지속형 이득이 관측되지 않아 3개 학습 시드로 D0·D1만 진행했다.',
'고정 행동 시험에서 13번과 16번 모두 적합 차로에서 480m 전에 이탈하고 약 365m 전 복귀하면 D0 +0.3, D1 0이었다. 13번 부적합 출발의 최초 개선은 두 조건 모두 +0.3이었다. 4·14번 후속 도로 1063527748에서 200m 전 이탈 뒤 끝에서 복귀한 경우는 모두 0이었다. 이 수치는 정책이 자유롭게 선택한 왕복 빈도와 구별한다.',
'','## 검증·파일','','holdout_by_seed.csv(모델·시드·교통별), first_fork_lanes.csv, lane_drops.csv, lane_cycles.csv, cycle_scope_audit.csv, failed_route16.csv, matched_training.csv, checkpoint_metrics.csv. 원시 파일과 소스는 같은 폴더에 있다. 각 교통 수준에서 모든 미사용 시드 출발 상태를 조건·모델 간 대조했다. 각 조건에서 자체적으로 이탈·복귀한 사실과 지급액을 기록하며 행동의 의도는 직접 측정하지 않는다.']
(o/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('independent extra',len(miss),Counter((z['arm'],z['paid']>0) for z in miss),'training',training)
