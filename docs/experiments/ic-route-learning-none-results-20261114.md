# 무교통 경로학습 독립 비교

16경로, seed11, 각20만 스텝, 강제보정 없음. 5만 스텝 경계 이후 첫 PPO 업데이트에서 중간 저장. 입력63차원·초기가중치·PPO 동일. 학습 보상 성분은 각 run의 reward-components.jsonl 및 TensorBoard에 누적 기록.

|조건|상태|스텝|
|---|---|---:|
|baseline|complete|200000|
|compact|complete|200000|
|route_reward|complete|200000|
|scale5|evaluation||
|dwell|training|126178|
|guidance|training|77185|

|조건|모델|완주/16|충돌|기타|
|---|---|---:|---:|---:|
|baseline|policy-100352.pt|13/16|0|3|
|baseline|policy-151552.pt|16/16|0|0|
|baseline|policy-200000.pt|13/16|0|3|
|baseline|policy-51200.pt|14/16|0|2|
|compact|policy-100352.pt|16/16|0|0|
|compact|policy-151552.pt|16/16|0|0|
|compact|policy-200000.pt|14/16|0|2|
|compact|policy-51200.pt|16/16|0|0|
|route_reward|policy-100352.pt|11/16|0|5|
|route_reward|policy-151552.pt|16/16|0|0|
|route_reward|policy-200000.pt|16/16|0|0|
|route_reward|policy-51200.pt|11/16|0|5|
