# 관측·보상 단독 비교 실행 기록

수정된 공통 검출기,경로안내300m,기본15,000대/h,16경로,seed11,각200,000스텝. 모두동일초기가중치에서시작하며강제보정은없다. 무거운원시자료와정책은로컬 `results/observation_reward_trials_20261114/`에보관한다.

| 조건 | 진행 | 마지막 학습 스텝 |
|---|---|---:|
|baseline|training|1272|
|obs_presence|training|1285|
|obs_risk|training|1272|
|reward_gap|training|1272|
|reward_block|training|1272|
|reward_lane|training|1272|

| 조건 | 체크포인트 | 교통 | 완주/전체 | 충돌 | 기타 |
|---|---|---|---:|---:|---:|
