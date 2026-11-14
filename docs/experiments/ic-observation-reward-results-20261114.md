# 관측·보상 단독 비교 실행 기록

수정된 공통 검출기,경로안내300m,기본15,000대/h,16경로,seed11,각200,000스텝. 모두동일초기가중치에서시작하며강제보정은없다. 무거운원시자료와정책은로컬 `results/observation_reward_trials_20261114/`에보관한다.

| 조건 | 진행 | 마지막 학습 스텝 |
|---|---|---:|
|baseline|complete|200000|
|obs_presence|complete|200000|
|obs_risk|complete|200000|
|reward_gap|complete|200000|
|reward_block|complete|200000|
|reward_lane|complete|200000|

| 조건 | 체크포인트 | 교통 | 완주/전체 | 충돌 | 기타 |
|---|---|---|---:|---:|---:|
|baseline|policy-100352.pt|none|13/16|0|3|
|baseline|policy-100352.pt|reference|33/48|15|0|
|baseline|policy-200000.pt|none|13/16|0|3|
|baseline|policy-200000.pt|reference|38/48|1|9|
|obs_presence|policy-100352.pt|none|6/16|0|10|
|obs_presence|policy-100352.pt|reference|30/48|8|10|
|obs_presence|policy-200000.pt|none|10/16|0|6|
|obs_presence|policy-200000.pt|reference|30/48|6|12|
|obs_risk|policy-100352.pt|none|8/16|0|8|
|obs_risk|policy-100352.pt|reference|22/48|8|18|
|obs_risk|policy-200000.pt|none|14/16|0|2|
|obs_risk|policy-200000.pt|reference|32/48|1|15|
|reward_block|policy-100352.pt|none|10/16|0|6|
|reward_block|policy-100352.pt|reference|22/48|9|17|
|reward_block|policy-200000.pt|none|8/16|0|8|
|reward_block|policy-200000.pt|reference|25/48|5|18|
|reward_gap|policy-100352.pt|none|13/16|0|3|
|reward_gap|policy-100352.pt|reference|13/48|32|3|
|reward_gap|policy-200000.pt|none|10/16|0|6|
|reward_gap|policy-200000.pt|reference|22/48|5|21|
|reward_lane|policy-100352.pt|none|7/16|0|9|
|reward_lane|policy-100352.pt|reference|24/48|5|19|
|reward_lane|policy-200000.pt|none|14/16|0|2|
|reward_lane|policy-200000.pt|reference|39/48|6|3|
