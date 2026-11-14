# 교통 정책 이어학습: 안전보상 독립 비교 및 조합 탐색

원본 교통정책20만 스텝 보존. 표의 스텝은 추가학습 스텝이며0은 출발 정책이다. Adam 상태는 새로 시작. 모든 조건 reference교통 학습, seed11. 평가seed39101~39103. 강제보정 없음.

|조건|상태|추가 스텝|
|---|---|---:|
|baseline|complete|200000|
|combo|complete|200000|
|gap|complete|200000|
|lane|complete|200000|

|조건|추가학습 모델|교통|완주|충돌|기타|위험시 감속요청/위험표본|실제감속|
|---|---|---|---:|---:|---:|---:|---:|
|baseline|policy-0-evaluation|none|13/16|0|3|0/0|0|
|baseline|policy-0-evaluation|reference|39/48|1|8|22/22|22|
|baseline|policy-100352-evaluation|none|9/16|0|7|0/0|0|
|baseline|policy-100352-evaluation|reference|40/48|3|5|10/13|9|
|baseline|policy-151552-evaluation|none|14/16|0|2|0/0|0|
|baseline|policy-151552-evaluation|reference|39/48|3|6|40/40|39|
|baseline|policy-200000-evaluation|none|15/16|0|1|0/0|0|
|baseline|policy-200000-evaluation|reference|38/48|5|5|64/64|64|
|baseline|policy-51200-evaluation|none|16/16|0|0|0/0|0|
|baseline|policy-51200-evaluation|reference|37/48|3|8|19/19|19|
|combo|policy-0-evaluation|none|13/16|0|3|0/0|0|
|combo|policy-0-evaluation|reference|37/48|2|9|41/41|41|
|combo|policy-100352-evaluation|none|13/16|0|3|0/0|0|
|combo|policy-100352-evaluation|reference|36/48|2|10|87/87|86|
|combo|policy-151552-evaluation|none|14/16|0|2|0/0|0|
|combo|policy-151552-evaluation|reference|40/48|6|2|37/37|36|
|combo|policy-200000-evaluation|none|13/16|0|3|0/0|0|
|combo|policy-200000-evaluation|reference|42/48|6|0|52/52|52|
|combo|policy-51200-evaluation|none|16/16|0|0|0/0|0|
|combo|policy-51200-evaluation|reference|38/48|6|4|54/54|53|
|gap|policy-0-evaluation|none|13/16|0|3|0/0|0|
|gap|policy-0-evaluation|reference|39/48|1|8|22/22|22|
|gap|policy-100352-evaluation|none|11/16|0|5|0/0|0|
|gap|policy-100352-evaluation|reference|25/48|1|22|36/36|36|
|gap|policy-151552-evaluation|none|13/16|0|3|0/0|0|
|gap|policy-151552-evaluation|reference|30/48|4|14|68/68|66|
|gap|policy-200000-evaluation|none|15/16|0|1|0/0|0|
|gap|policy-200000-evaluation|reference|40/48|5|3|46/46|46|
|gap|policy-51200-evaluation|none|8/16|0|8|0/0|0|
|gap|policy-51200-evaluation|reference|40/48|8|0|60/60|60|
|lane|policy-0-evaluation|none|13/16|0|3|0/0|0|
|lane|policy-0-evaluation|reference|39/48|1|8|22/22|22|
|lane|policy-100352-evaluation|none|13/16|0|3|0/0|0|
|lane|policy-100352-evaluation|reference|34/48|4|10|35/35|34|
|lane|policy-151552-evaluation|none|16/16|0|0|0/0|0|
|lane|policy-151552-evaluation|reference|42/48|1|5|17/17|17|
|lane|policy-200000-evaluation|none|16/16|0|0|0/0|0|
|lane|policy-200000-evaluation|reference|37/48|1|10|26/26|26|
|lane|policy-51200-evaluation|none|13/16|0|3|0/0|0|
|lane|policy-51200-evaluation|reference|38/48|0|10|41/41|41|
