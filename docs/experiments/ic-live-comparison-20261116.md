# 목표 근접 기준 모델의 후속 비교

동일 출발 모델 round-02/route/policy-200000.pt, 학습 seed11. 강제 보정 없음. 추가 학습 스텝 기준.
평가 seed79101~79103: 교통48회, 무교통16회. 완료된 전체 평가만 표시하며 중복 재평가는 합산하지 않는다.

|조건|추가 스텝|무교통 완주|교통 완주|교통 충돌|교통 기타|완주+충돌|
|---|---:|---:|---:|---:|---:|---:|
|교차 관측+보상|51,200|13/16|42/48 (87.50%)|6/48 (12.50%)|0/48|100.00%|
|기존 조건 추가학습|0|15/16|45/48 (93.75%)|3/48 (6.25%)|0/48|100.00%|
|기존 조건 추가학습|51,200|14/16|37/48 (77.08%)|4/48 (8.33%)|7/48|85.42%|
|기존 조건 추가학습|100,352|15/16|44/48 (91.67%)|4/48 (8.33%)|0/48|100.00%|
|기존 조건 추가학습|151,552|12/16|39/48 (81.25%)|8/48 (16.67%)|1/48|97.92%|
|기존 조건 추가학습|200,000|13/16|39/48 (81.25%)|3/48 (6.25%)|6/48|87.50%|
|교차 차량 관측|0|15/16|45/48 (93.75%)|3/48 (6.25%)|0/48|100.00%|
|교차 차량 관측|51,200|15/16|41/48 (85.42%)|3/48 (6.25%)|4/48|91.67%|
|교차 차량 관측|100,352|13/16|45/48 (93.75%)|3/48 (6.25%)|0/48|100.00%|
|교차 차량 관측|151,552|13/16|35/48 (72.92%)|6/48 (12.50%)|7/48|85.42%|
|교차 차량 관측|200,000|16/16|42/48 (87.50%)|6/48 (12.50%)|0/48|100.00%|
|교차 위험 보상|0|15/16|45/48 (93.75%)|3/48 (6.25%)|0/48|100.00%|
|교차 위험 보상|51,200|16/16|38/48 (79.17%)|10/48 (20.83%)|0/48|100.00%|
|교차 위험 보상|100,352|14/16|36/48 (75.00%)|11/48 (22.92%)|1/48|97.92%|
|교차 위험 보상|151,552|14/16|34/48 (70.83%)|13/48 (27.08%)|1/48|97.92%|
|교차 위험 보상|200,000|12/16|36/48 (75.00%)|7/48 (14.58%)|5/48|89.58%|
|다음 분기 안내|51,200|16/16|41/48 (85.42%)|7/48 (14.58%)|0/48|100.00%|
|다음 분기 안내|100,352|16/16|37/48 (77.08%)|11/48 (22.92%)|0/48|100.00%|
|옆 차로 위험 관측|51,200|16/16|38/48 (79.17%)|10/48 (20.83%)|0/48|100.00%|

## 판단 기준

- 기준 모델의 별도320회 검증은 완주287·충돌29·시간초과4였다. 이 표의48회와 표본을 섞지 않는다.
- 기존 모델을 보존한다. 더 오래 학습한 모델이나 새 기능을 넣은 모델을 자동으로 우수하다고 취급하지 않는다.
- 개발 평가에서 개선된 후보만 새로운 seed로 검증하고, 충돌률5% 미만 및 기타 실패0 여부를 별도로 확인한다.

## 평가 파일

- `results/junction_conflict_trials_20261116/runs/control/policy-0-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/control/policy-100352-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/observation/policy-100352-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/reward/policy-100352-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/control/policy-151552-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/observation/policy-151552-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/reward/policy-151552-evaluation.json`
- `results/junction_conflict_trials_20261116/early-evaluation/51200/combined/policy-51200-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/control/policy-51200-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/observation/policy-51200-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/reward/policy-51200-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/control/policy-200000-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/observation/policy-0-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/observation/policy-200000-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/reward/policy-0-evaluation.json`
- `results/junction_conflict_trials_20261116/runs/reward/policy-200000-evaluation.json`
- `results/lane_followup_20261116/early-evaluation/100352/future/policy-100352-evaluation.json`
- `results/lane_followup_20261116/early-evaluation/51200/future/policy-51200-evaluation.json`
- `results/lane_followup_20261116/early-evaluation/51200/lateral/policy-51200-evaluation.json`
