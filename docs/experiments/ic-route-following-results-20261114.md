# 관측·보상 단독 비교 실행 기록

수정된 공통 검출기,경로안내300m,기본15,000대/h,16경로,seed11,각200,000스텝. 모두동일초기가중치에서시작하며강제보정은없다. 무거운원시자료와정책은로컬 `results/route_following_trials_20261114/`에보관한다.

| 조건 | 진행 | 마지막 학습 스텝 |
|---|---|---:|
|baseline|complete|200000|
|guidance|complete|200000|
|route_reward|complete|200000|

| 조건 | 체크포인트 | 교통 | 완주/전체 | 충돌 | 기타 |
|---|---|---|---:|---:|---:|
|baseline|policy-100352.pt|none|13/16|0|3|
|baseline|policy-100352.pt|reference|33/48|15|0|
|baseline|policy-200000.pt|none|13/16|0|3|
|baseline|policy-200000.pt|reference|38/48|1|9|
|guidance|policy-100352.pt|none|8/16|0|8|
|guidance|policy-100352.pt|reference|15/48|10|23|
|guidance|policy-200000.pt|none|9/16|0|7|
|guidance|policy-200000.pt|reference|19/48|9|20|
|route_reward|policy-100352.pt|none|8/16|0|8|
|route_reward|policy-100352.pt|reference|21/48|6|21|
|route_reward|policy-200000.pt|none|11/16|0|5|
|route_reward|policy-200000.pt|reference|37/48|0|11|

## 안내를 따르는 행동 지표

정지 스텝의 과대표현을 피하기 위해 주행 중(0.5m/s 초과) 비율을 별도 표시한다. 요청률은 실제 변경 성공률이 아니다.

|조건|모델|교통|주행 중 필요방향 요청/필요 스텝|주행 중 올바른 차로/판정 스텝|목표 방향 실제 변경|맞는 차로 이탈|
|---|---|---|---:|---:|---:|---:|
|baseline|policy-100352.pt|none|3/29|931/960|3|1|
|baseline|policy-100352.pt|reference|7/628|2638/3266|19|3|
|baseline|policy-200000.pt|none|4/30|930/960|4|2|
|baseline|policy-200000.pt|reference|22/851|2424/3275|37|22|
|guidance|policy-100352.pt|none|0/100|729/829|0|0|
|guidance|policy-100352.pt|reference|0/407|1944/2351|0|1|
|guidance|policy-200000.pt|none|0/98|780/878|0|0|
|guidance|policy-200000.pt|reference|12/643|2469/3112|12|12|
|route_reward|policy-100352.pt|none|1/242|660/902|1|2|
|route_reward|policy-100352.pt|reference|2/830|2638/3468|4|1|
|route_reward|policy-200000.pt|none|1/336|757/1093|6|4|
|route_reward|policy-200000.pt|reference|5/805|2742/3547|18|9|
