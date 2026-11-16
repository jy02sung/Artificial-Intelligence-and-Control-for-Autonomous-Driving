# 위험 상황 curriculum·선택적 모방 실행 계획

Goal: 기준 모델의 정상 주행을 보존하면서 위험 경험을 집중 학습하고, 강제 개입 없이 교통 완주 향상·충돌5%미만·기타실패0을 검증한다.
Architecture: 기존 환경/63차원 관측/보상/Hybrid PPO 가중치를 재사용한다. 별도 수집·curriculum·PPO 보조손실 모듈로 구성하며 원본은 불변 보존한다. 재개는 정책·optimizer·RNG·스케줄·실제 행동 이력을 저장하고 완전 reset+seed+행동 replay로 검증한다.
Tech stack: Python/PyTorch/libSUMO/NumPy. Spec: 로컬 results/curriculum_keep_20261116/user-spec.txt(사용자 원문).

- [x] 기존8학습의 마지막 정상 recovery를 별도복사하고 PID/명령을 확인해 전환. 대표평가 lane4/100352는 계속한다. recovery의환경·RNG부재는 기록하며 완전재개가능하다고 부르지 않는다.
- [x] 필수점검: 가감속 clip의 latent-action logprob정합, lane변환, terminal/time-limit GAE, 고정정규화와 가중치로드. 새 runner는 확인된 timeout경계수정 적용.
- [x] 과제 모방허용 확인. 로컬 project_materials/0_Project_Introduction.md는 강화/모방 모두허용,5_Project_Preparation.md는PPO/BC지원. 금지미발견, 최신 사용자허용에따라keep/safe사용.
- [x] D_keep: 기준모델 재생 중 안전거리·경로정합·변경실행을검수. 완주episode전체를무조건채택하지 않는다. 정상직진·분기준비·변경을층화한다. 훈련seed910001이상,개발79101~3및과거691xx와분리. D_safe는검수된안전시연확보후별도B.
- [x] curriculum: 전체16경로 + 합류(01/02)·안전변경(11)·연속분기(13/14/15) 중심. 실제전체SUMO를seed/warmup/route로초기화하며AV순간이동없음. 초기버전은원래입구부터시작해충돌직전상태강제초기화안함. warmup800/900/1000초와seed로차간격·접근속도·진입시각분포를변화시키고측정. 평가warmup900고정.
- [x] 환경스텝예산으로전체/위험약50:50. 실제위험노출·유형·route별카운트저장. 어려운상황만반복하지않음.
- [x] A: 기준관측/보상 + curriculum + keep KL + actor LR3e-5. shared trunk는actor영향이있어3e-5,critic head는3e-4. 보조손실별gradient크기계측. KL조기종료threshold0.015,clip0.2,다른hp기준유지. Critic예열은근거없어초기에는안함.
- [x] B: A와같은부모에서safe BC만추가. 교사행동을on-policy PPO버퍼로혼합하지않음. 데이터부족이면A먼저실행.
- [ ] smoke,초기행동일치,중간저장재개동일성검증후A 50k pilot실제실행. B별도50k. 최대A/B/C3구성,유망할때만200k확장.
- [ ] 대표개발48회완료+무교통16,후보3학습seed,확정후새320seed와기준동일조건비교. 최종seed991001~991020예약(이전학습/평가중복검사). 표본CI·한계기록.

기존goal의채팅점검10~20분요청유지. 별도앱예약자동화는생성하지않음. 실행내용은이문서와재개명령기록으로전환한다. 목표를확인하기전완료처리안함. 보고항목은변경/완료/비교/한계/다음작업.

참고: https://spinningup.openai.com/en/latest/algorithms/ppo.html (PPO KL early stopping), https://proceedings.mlr.press/v15/ross11a.html (DAgger). PPO+보조모방은RAPD-RL직접재현이아님.


## 2026-11-16 실행 전 검증

- 기존 lane4/100352 대표 평가 완료: 무교통 15/16 완주, 교통 44완주·2충돌·2시간초과/48. 기준 45·3·0 대비 충돌은 줄었지만 완주가 감소하고 기타 실패가 늘어 채택하지 않는다.
- 별도 unittest 7개 통과: Hybrid 행동 log-prob, 동일 정책 keep KL, KL 중단, optimizer 그룹, timeout GAE 경계 3개. pytest는 설치되어 있지 않아 unittest로 실행했다.
- 512스텝 연속 실행과 256스텝 저장 후 새 프로세스에서 512까지 재개한 상태를 재귀 비교했다. 정책·optimizer·SUMO 재현 상태·스케줄·Torch/NumPy/Python/보조배치 RNG·episode 통계 모두 일치. 최초 Python random 초기화 누락을 고쳐 최종 재검증했다.
- 평가 smoke는 무교통/교통 각 1회, 50환경스텝 제한으로 종료·저장 경로만 확인했다. 두 timeout은 의도된 짧은 제한 결과이며 성능 평가가 아니다.
- D_keep 1,434개: 직진837, 분기준비593, 차로변경4. 차로변경 커버리지는 제한적이다. D_safe326개는 검수된 감속·양보이며 안전한 차로변경 시연은 아직 없다. 사고2회/시간초과1회 episode는 시연에서 제외했다.
- 평가 도구의 Path import 누락도 smoke 전 수정했다. 평가 결과를 보고서에 쓸 때 smoke 경로는 제외한다.

## 최초 병렬 pilot

동일한 기준 부모에서 A/B를 각각 50,000 추가 환경스텝 학습한다. A는 curriculum+keep+낮은 actor LR, B는 A에 safe BC 가중치0.1만 추가한다. 보상·63차원 관측·평가 행동 방식은 기준을 유지한다. B의 추가 모방은 현재 감속·양보에 한정된다.

```powershell
python -X utf8 -u -m utils.run_curriculum_keep --condition A --name A-seed11 --seed 11 --budget 50000
python -X utf8 -u -m utils.run_curriculum_keep --condition B --name B-seed11 --seed 11 --budget 50000
python -X utf8 -u -m utils.evaluate_curriculum --name A-seed11 --step 10240
python -X utf8 -u -m utils.evaluate_curriculum --name B-seed11 --step 10240
```

각 학습은 Torch 1스레드, 두 프로세스 병렬. 매 rollout 체크포인트·optimizer·RNG·교통 재현 정보를 로컬 results/curriculum_keep_20261116에 보존한다. 같은 실행 명령으로 재개하며 살아 있는 PID가 있으면 중복 실행을 거부한다. 10,240/20,480/50,000스텝 모델을 무교통16+교통48 평가하고 기준과 비교한다. pilot 실행 여부는 progress.json과 살아 있는 프로세스로 확인한다. 아직 개선을 확인한 결과는 없다.
