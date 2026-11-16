# 교차 충돌 관측·보상 독립 및 조합 실험

출발 정책: round-02/route/policy-200000.pt 고정·보존. 표의 스텝은 추가학습 스텝이며0은 출발 정책이다. Adam 상태는 새로 시작. 모든 조건 reference교통 학습, seed11. 평가seed79101~79103. 강제보정 없음.

|조건|상태|추가 스텝|
|---|---|---:|
|control|complete|200000|
|observation|complete|200000|
|reward|complete|200000|
|combined|training|161706|

|조건|추가학습 모델|교통|완주|충돌|기타|위험시 감속요청/위험표본|실제감속|
|---|---|---|---:|---:|---:|---:|---:|
|control|policy-0-evaluation|none|15/16|0|1|0/0|0|
|control|policy-0-evaluation|reference|45/48|3|0|39/39|36|
|control|policy-100352-evaluation|none|15/16|0|1|0/0|0|
|control|policy-100352-evaluation|reference|44/48|4|0|37/37|34|
|control|policy-151552-evaluation|none|12/16|0|4|0/0|0|
|control|policy-151552-evaluation|reference|39/48|8|1|36/36|33|
|control|policy-200000-evaluation|none|13/16|0|3|0/0|0|
|control|policy-200000-evaluation|reference|39/48|3|6|48/48|45|
|control|policy-51200-evaluation|none|14/16|0|2|0/0|0|
|control|policy-51200-evaluation|reference|37/48|4|7|36/36|33|
|observation|policy-0-evaluation|none|15/16|0|1|0/0|0|
|observation|policy-0-evaluation|reference|45/48|3|0|39/39|36|
|observation|policy-100352-evaluation|none|13/16|0|3|0/0|0|
|observation|policy-100352-evaluation|reference|45/48|3|0|61/61|58|
|observation|policy-151552-evaluation|none|13/16|0|3|0/0|0|
|observation|policy-151552-evaluation|reference|35/48|6|7|70/70|67|
|observation|policy-200000-evaluation|none|16/16|0|0|0/0|0|
|observation|policy-200000-evaluation|reference|42/48|6|0|83/83|81|
|observation|policy-51200-evaluation|none|15/16|0|1|0/0|0|
|observation|policy-51200-evaluation|reference|41/48|3|4|59/63|57|
|reward|policy-0-evaluation|none|15/16|0|1|0/0|0|
|reward|policy-0-evaluation|reference|45/48|3|0|39/39|36|
|reward|policy-100352-evaluation|none|14/16|0|2|0/0|0|
|reward|policy-100352-evaluation|reference|36/48|11|1|22/22|20|
|reward|policy-151552-evaluation|none|14/16|0|2|0/0|0|
|reward|policy-151552-evaluation|reference|34/48|13|1|25/25|22|
|reward|policy-200000-evaluation|none|12/16|0|4|0/0|0|
|reward|policy-200000-evaluation|reference|36/48|7|5|26/26|24|
|reward|policy-51200-evaluation|none|16/16|0|0|0/0|0|
|reward|policy-51200-evaluation|reference|38/48|10|0|36/36|34|
