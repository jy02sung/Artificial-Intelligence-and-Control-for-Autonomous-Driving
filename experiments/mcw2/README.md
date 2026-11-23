# mcw2 고교통 16경로 실험 기록

**발표 후보는 `merge_12`다.** 팀이 저장해 둔 DAgger 운전자에서 시작해 PPO를 32,768스텝 학습했다. 같은 교통·센서 시드로 각 경로를 100회씩 주행했을 때 완주가 **156회 늘고 충돌이 34회 줄었다**. 두 목표인 완주율 95% 이상과 충돌률 5% 미만을 16경로 합산에서 달성했다. [경로별 전후 결과와 변화량](../../results/local_mcw2_driver_ppo/paired_effect.md) · [최종 가중치](../../results/local_mcw2_driver_ppo/merge_12.pt)

| 같은 1,600회에서 비교 | 완주 | 충돌 | 600초 시간초과 |
|---|---:|---:|---:|
| 저장된 DAgger 운전자 | 1,381 (86.31%) | 84 (5.25%) | 135 (8.44%) |
| PPO 추가 학습 후 | 1,537 (96.06%) | 50 (3.125%) | 13 (0.81%) |
| 변화 | +9.75%p | −2.125%p | −7.625%p |

기존 실패가 완주가 된 시드는 192개, 완주가 실패가 된 시드는 36개였다. 13번 경로는 20→92회, 14번은 19→93회 완주했다. 두 경로의 시간초과가 135→13회로 줄면서 전체 개선의 대부분을 설명한다. 반면 02번은 개선 후에도 18/100회 충돌했고, 07번은 충돌이 0→5회로 늘었다. **경로마다 두 목표를 따로 적용하면 9/16개가 통과한다.** 경로별로 시드 쌍을 복원 추출한 95% 구간은 완주 변화 +8.44~+11.06%p, 충돌 변화 −3.38~−0.88%p다. 이는 두 고정 정책에 대한 교통·센서 시드 변동 범위다. 독립 학습을 여러 번 반복했을 때의 편차는 측정하지 않았다. [학습 전 개별 주행](../../results/local_mcw2_driver_ppo/saved_paired_holdout.json) · [학습 후 개별 주행](../../results/local_mcw2_driver_ppo/merge_12_original_holdout.json)

## 어떤 가중치에서 무엇을 배웠나

브랜치의 출발점은 [팀 mcw2 커밋 ac2f9bc](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving/commit/ac2f9bc7a63594fe4a1b45df83f0e8cd09314f73)이다. 이 커밋에는 여러 모델이 있다. 최초 1,600회 기준선으로 평가한 팀 최신 모델은 `mcw2_occlusion_ppo`이고, 최종 PPO의 직접 부모는 그보다 이른 `mcw2_agenda2_dagger_v2`다.

| 단계 | 학습 내용 | SHA-256 앞 12자리 |
|---|---|---|
| [팀의 DAgger 운전자](../../results/mcw_runs/mcw2_agenda2_dagger_v2/best_model.pt) | 팀의 안건 2 PPO 51,200스텝 모델에서 DAgger로 방문 상태에 교사 행동 라벨을 붙이고 BC 방식으로 지도학습한 저장점 | `f2126af64a90` |
| [gap_4](../../results/local_mcw2_driver_ppo/gap_4_train.json) | 시드 53171, 8,192스텝. 자차 주행에서 얻은 장기 보상과 목표 차선 앞뒤 간격·접근 속도·경로 연결성에 따른 즉시 보상을 함께 사용 | `ba4abf87c0e2` |
| [merge_12](../../results/local_mcw2_driver_ppo/merge_12_train.json) | 시드 53219, 24,576스텝 추가. 분기 전 안전하게 연결 차선을 확보하는 보상을 높이고, 좌·유지·우 세 행동의 즉시 보상에 대한 PPO 항을 기존 행동 확률로 가중 평균 | `9b1c0e92ae4e` |

두 추가 학습 단계에는 새 교사 라벨이나 DAgger 패스가 들어가지 않는다. 학습 중 정책이 직접 환경과 상호작용하고, 초기 정책에서 출력이 급격히 달라지지 않게 제한한다. 최종 주행은 기존 47차원 네트워크가 가감속과 차선 선택을 출력한다. 관측에는 차량 가림·노이즈가 있는 120m 시야와 **계획 경로를 따라 각 차선이 120m 안에서 얼마나 이어지는지**가 포함된다. 훈련용 위험 점수는 평가 때 행동을 고르는 데 쓰지 않는다. 수업 발표에서는 기존 팀 모델의 PPO·모방학습 이력과 추가 PPO의 보상 설계를 함께 밝히는 것이 정확하다. [팀 원본 학습 이력](../../docs/review/mcw2-agenda1-agenda2-report-20261121.md) · [보상·PPO 구현](local_driver_gap_ppo.py)

## 실험 순서와 원자료

표의 세 숫자는 **완주 / 충돌 / 시간초과** 순서다. 고교통 설정은 시간당 15,000대 유입 설정, 시뮬레이션 간격 0.2초다. 모델 선별용 80회, 개발용 400회, 확정용 1,600회는 용도가 다르다. W·T·V는 각각 별도 최종 시드를 썼고, 위 DAgger→PPO 차이는 정확히 같은 경로·시드를 짝지어 계산했다.

| 실험 | 확인한 질문과 결과 | 기록 |
|---|---|---|
| 팀 최신 모델 독립 검증 | 원래 80회는 77/3/0으로 종료 스텝까지 재현. 새 1,600회는 1,484/116/0. 충돌 116건을 재현하고 같은 경로·출발 차선에서 완주한 주행과 비교 | [실패 분석](../../results/local_holdout_1600/report.md) · [충돌별 기록](../../results/local_holdout_1600/incidents.csv) |
| X·Y | 취약 경로 집중 PPO(X)와 간격·접근 속도 보상 변경(Y). 같은 개발 400회에서 기준선 371/29/0, X 350/50/0, Y 354/46/0 | [X 설정·가중치·결과](../../results/experiment_X) · [X에서 새로 충돌한 35건](../../results/local_improvement/X_regressions.json) |
| Z·R·W·T·V | Z는 실제 충돌 직전 선택의 확률을 낮췄고, R은 성공한 차선 변경도 사용했다. W는 출력 편향을 탐색했다. T는 Z240에서 차선 편향을 조정해 최종 1,558/29/13, V는 추론 때 간격·경로 점수를 더해 1,575/10/15. 이 두 수치에는 해당 방법의 기여가 포함된다 | [후보별 방법·성적·실패](../../results/local_improvement/report.md) · [기존 팀 공유 요약](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving/blob/fb56565fc1c34de4d7d8a2b0ab266d80cdf2bbde/docs/experiments/high-traffic-16-route-failures.md) |
| SUMO 기본 차량 시범 BC→PPO | 팀 운전자와 다른 교사를 사용한 별도 시도. 기본·복귀·측방 복귀 시범 총 2,176회를 저장했다. 충돌을 줄인 후보의 독립 1,600회는 1,157/101/342 | [시범·실패 원인·모델별 결과](../../results/local_native_bc_ppo/report.md) · [원자료](../../results/local_native_bc_ppo) |
| 팀 최신 모델에 DAgger 추가 후 PPO | 추가 DAgger 직후 이미 개발 391/6/3. 최종 평가에서 일반 PPO `fresh_2`는 1,520/32/48, 차선 출력 중심 PPO `lane_2`는 1,537/47/16. 초기 성능이 높은 별도 실험군 | [DAgger 과정·실험군 비교](../../results/local_mcw2_driver_ppo/report.md) |
| 저장된 팀 모델에서 바로 PPO | 기존 DAgger 직후 모델과 팀 최신 모델에서 각각 일반 PPO, 차선 출력 중심 PPO, 간격·합류 보상을 비교했다. 기존 DAgger에서 일반 PPO는 개발 344/25/31→363/37/0, 최종 선정한 `merge_12`는 같은 개발 시드에서 391/6/3 | [후보·학습 길이·경로별 결과](../../results/local_mcw2_driver_ppo/original_only.md) · [최종 모델·시드 동결 기록](../../results/local_mcw2_driver_ppo/original_only_freeze.json) |

[가중치 전체 목록과 파일별 SHA-256](checkpoints.csv) · [학습·평가 실행 로그](logs) · [후보별 체크포인트와 주행 JSON](../../results/local_mcw2_driver_ppo)

## 재현하기

저장소 루트에서 실행한다. 첫 명령은 Python 표준 라이브러리만 사용한다. 출발·중간·최종 가중치의 SHA-256과 부모 관계, 최종 시드와 학습·선별 시드의 분리, 1,600쌍의 종료 사유와 변화량을 검사한다.

```powershell
python -m experiments.mcw2.reproduce verify
```

시뮬레이션 및 재학습을 검증한 환경은 Windows, Python 3.10.2, CPU PyTorch 2.14.0, NumPy 2.2.6, Gymnasium 1.3.0, Eclipse SUMO/libsumo 1.27.1이다. Eclipse SUMO를 설치하고 저장소 루트에서 아래 명령을 실행한다. `--workers`는 평가용 CPU 프로세스 수다.

```powershell
python -m pip install -r experiments/mcw2/requirements.txt
python -m experiments.mcw2.reproduce replay --per-route 2 --workers 4
python -m experiments.mcw2.reproduce train --output results/reproduced_merge12
```

`replay`는 두 정책을 모두 실행해 경로·시드·종료 사유·종료 스텝을 원래 기록과 대조한다. `--per-route 100`이면 총 3,200회다. `train`은 비어 있는 출력 폴더에 두 단계의 가중치를 생성하고 보관본과 해시를 비교한다. 게시 전 재학습에서 두 해시가 모두 일치했고, 경로당 2시드씩 두 정책을 다시 주행한 64회도 종료 스텝까지 일치했다. 과거 실험별 `local_*.py`는 원래 산출 경로에 결과를 쓰므로 새 결과를 만들 때는 별도 체크아웃에서 실행한다.
