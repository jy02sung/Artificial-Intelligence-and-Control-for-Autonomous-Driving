import collections,csv,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/collision_forensics_20261114'
def main():
 rs=[json.loads(p.read_text()) for p in OUT.glob('*route*.json')]
 totals=collections.Counter(); records=[]
 for r in rs:
  official=json.loads((ROOT/f"results/route_repeat_transfer_20261114/seed{r['seed']}/runs/{r['condition']}/policy-200000-evaluation.json").read_text())
  matched=next(x for x in official if x['traffic']=='reference' and x['seed']==29101 and x['route']==r['route'])
  assert matched['outcome']==r['outcome'] and abs(matched['seconds']-r['seconds'])<1e-6
  assert r['route_overrides']==0
  samples={x['t']:x for x in r['first']+r['tail']}
  errors=0;visible=0;command_errors=0;commands=0
  for x in samples.values():
   l=x['leader'];inrange=l is not None and l['gap']<=r['visibility']
   e=[max(0,l['gap'])/r['visibility'],max(-1,min(1,(l['speed']-x['speed'])/r['vmax']))] if inrange else [1,1]
   errors+=int(max(abs(a-b) for a,b in zip(e,x['front_obs']))>1e-5);visible+=int(inrange)
   if 'after_speed' in x:commands+=1;command_errors+=int(abs(x['target_speed']-x['after_speed'])>1e-4)
  totals.update(samples=len(samples),visible=visible,obs_errors=errors,commands=commands,command_deviations=command_errors)
  a=r['first'][0];z=r['tail'][-1];cols=[c for c in r['collisions'] if 'ego' in (c['collider'],c['victim'])]
  record=dict(seed=r['seed'],condition=r['condition'],route=r['route'],outcome=r['outcome'],seconds=r['seconds'],start_gap=a['leader']['physical_gap'] if a['leader'] else None,start_ttc=a['leader']['ttc'] if a['leader'] else None,final_speed=z['speed'],final_accel_raw=z['action'][0],final_gap=z['leader']['physical_gap'] if z['leader'] else None,collision_types=','.join(c['type'] for c in cols),final_edge=z['edge'],observation_errors=errors,visible_samples=visible,command_deviations=command_errors)
  records.append(record)
 collision=[r for r in records if r['outcome']=='collision']
 lines=['# 교통 전이 충돌 원인 재현 분석','','## 범위','',f'최종20만 모델 3조건×학습seed3개×16경로 중 {len(rs)}/144 재현 완료. 교통seed29101. 새 학습/행동보정 없음. 공식 평가 결과와 종료시각을 모두 대조했다. 이는 기존 전체432회 중144회에 대한 상세추적이다. 첫2초와 마지막30초의0.2초 입력/행동/물리상태를 보존했다.',
 '',f"현재 재현: {dict(collections.Counter(r['outcome'] for r in records))}. 충돌 직전 음의 가감속 출력: {sum(r['final_accel_raw']<0 for r in collision)}/{len(collision)}.",
 '',f"저장 표본 검증: {dict(totals)}. obs_errors는 감지범위50m 밖 차량을 '없음'으로 처리하는 기존 규칙까지 반영해 다시 계산했다. 원시 stats.observation_mismatch의 일부 초기 실행 값은 범위 밖 차량까지 비교한 계측 오류가 있으므로 사용하지 않는다. 이는 모델/환경 변경이 아니라 분석기 기대값 수정이다.",
 '', '## 확인된 실제 사례','',
 '|사례|관측과 행동|판단|','|---|---|---|',
 '|seed11 상세안내 route01|출발 앞차 약118m. 12.2초에 새 앞차 약16.74m, AV27.78m/s·앞차19.02m/s. 가속 raw+0.613, 이후14초 추돌|출발 근접이 아니라 합류 후 접근 상황에서 감속 미선택|',
 '|seed22 상세안내 route03/04|출발 자기차로 앞차31.11m. 0→1→2차로로0.4초 동안 이동. 0.2초 입력에 왼쪽 앞차 gap5.03m·상대속도−11.81m/s가 있었는데 왼쪽 변경+가속. 0.6초 추돌|초기 간격보다 위험한 연속 변경과 가속 선택|',
 '|기존 교통학습 baseline route01 대조|같은교통seed·시작조건. 12.2초 감속 raw−0.978, 속도27.78→26.72m/s. 34초 완주|감속 전달 경로가 작동하고 기존 정책은 위협에 반응|',
 '', '## 같은 관측에서 정책 비교','', '01번 새 baseline 충돌 기록의 동일63차원 입력을 두 정책에 그대로 넣었다. 12.2초 앞차15.42m에서 기존 교통정책 가감속−0.982, 새 무교통정책+0.454. 이후 충돌 전8개 위험표본 모두 기존 정책은 감속, 새 정책은 가속이었다. 도로/위치/관측 차이를 고정한 출력 비교이며, 반사실 정책의 완주 성능을 뜻하지는 않는다.', '', '09번 seed11 상세안내는 첫 입력의 오른쪽 앞차 gap=0인데도 오른쪽 변경을 실행해0.2초 측면충돌했다. 10번 seed33 기준도 왼쪽 gap=0에서 왼쪽 변경 후0.2초 측면충돌했다. 출발 자기차로 간격 부족과 구분한다.', '', '## 원인과 수정 대상','',
 '1. 현재 차로 선행 관측은 기록 표본에서 실제 SUMO 선행차와 대조한다. 관측이 들어왔어도 양의 가속을 내는지 구분하며, 단순 관측 누락이라고 단정하지 않는다.',
 '2. env/sumo_env.py의 step은 accel_raw를 가속도로 바꿔 setSpeed에 전달한다. speedMode=0, laneChangeMode=0, action_safety=False이므로 자동 제동이나 위험한 차로변경 차단은 없다. 기존/새 조건에 공통이며 이번에 새로 켜거나 끈 것이 아니다.',
 '3. 무교통 학습에서는 주변 차량이 없어 거리/상대속도 입력이 위험 상태를 경험하지 않는다. 길찾기 완주는 추종·양보·안전 차로변경 능력의 검증이 아니다. 기존 교통학습 정책을 폐기하거나 이 모델로 대체해서는 안 된다.',
 '4. 앞차가 합류해 현재 선행차로 잡히는 순간 간격이 급변한다. 현재 차로 관측만의 신뢰성과 합류 전 차량 예측 정보의 충분성은 다른 문제다. merge 상황 관측 보강 필요성은 별도 진단 대상이다.',
 '5. 다음 구현 우선순위: 기존 안전반응 정책 보존 → 동일 관측에서 기존/새 정책의 제동 반응 고정평가 → 저교통에서 추종/합류/옆차 간격을 학습하는 단계 추가 → 길찾기 회귀 평가 병행. 지도나 출발 간격부터 일괄 완화하지 않는다.',
 '6. 보상 후보는 속도별 안전거리/TTC 및 위험한 목표차로 간격 벌점이다. 체류 벌점이 안전한 양보와 충돌하지 않는지 별도 비교한다. 자동 안전 게이트를 성능 개선처럼 숨겨 적용하지 않는다.',
 '', '## 해석 제한','',
 '후방 추돌과 측면 충돌을 분리해야 한다. 현재차로 getLeader 검증만으로 모든 주변 차량 감지가 완전하다고 주장할 수 없다. 진단 단계에서 정책·맵·행동 규칙을 수정하지 않았다. 속도 명령과 실제 속도의 차이가 있으면 연결부/종료부 물리 제약 여부를 별도 분석한다.',
 '', '원시 파일: results/collision_forensics_20261114/, 기존 교통 모델 대조: results/collision_traffic_control_20261114/.']
 lines+=['', '## 충돌 유형 및 기존 정책 대조', '', '충돌 유형: '+str(dict(collections.Counter(r['collision_types'] for r in collision))), '기존 정책도 모든 경로가 완벽한 것은 아니다. 아래 대조 결과를 유지하며 새 모델로 덮어쓰지 않는다.']
 for f in sorted((ROOT/'results/collision_traffic_control_20261114').glob('*route*.json')):
  q=json.loads(f.read_text());lines.append(f"- {q['route']}: {q['outcome']}, {q['seconds']}초")
 lines+=['', '재현 명령: `python -m utils.collision_forensics`, 기존 정책 대조: `python -m utils.collision_forensics --control`, 집계: `python -m utils.summarize_collision_forensics`.']
 dest=ROOT/'docs/experiments/ic-collision-forensics-20261114.md';dest.write_text('\n'.join(lines)+'\n',encoding='utf8')
 with dest.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print(len(rs),dict(totals),collections.Counter(r['collision_types'] for r in collision))
if __name__=='__main__':main()
