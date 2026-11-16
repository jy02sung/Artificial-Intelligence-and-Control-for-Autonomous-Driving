# 상위 3개 모델: 수집 데이터와 63차원 관측 설명

기준일: 2026-11-16. 「퇴계원IC_상위3모델_데이터비교.pptx」의 표를 Markdown으로 옮기고 관측 인덱스 설명을 추가했다.

## 1. 비교 대상과 순위

동일 교통 48회(16경로 × 평가 seed 59101·59102·59103)를 완료한 비교군에서 **완주율 내림차순, 동률이면 충돌률 오름차순**으로 선정했다. 무교통 평가와 다른 seed의 48회·320회 결과는 섞지 않았다. 아래 스텝은 공통 부모 모델 이후 추가 학습량이다. 세 모델 모두 PPO이며 모방학습·강제 행동 보정은 사용하지 않았다.

|순위|모델|추가 학습 스텝|완주율|충돌률|시간초과|완주+충돌|
|---|---|---:|---:|---:|---:|---:|
|1|기존 보상 유지|151,552|97.92% (47/48)|0.00% (0/48)|1/48|97.92%|
|2|경로 선택·유지 보상 2배|200,000|97.92% (47/48)|2.08% (1/48)|0/48|100.00%|
|3|경로 보상 + 변경 기회 미사용 벌점|51,200|95.83% (46/48)|2.08% (1/48)|1/48|97.92%|

1위는 요청한 정렬 기준의 1위다. 시간초과 1건이 있어 완주+충돌 100% 목표를 만족하는 모델이라는 의미는 아니다. 세 모델의 **관측 구성과 로그 항목은 같고**, 보상 조건과 저장 시점이 다르다.

## 2. PPT 데이터 표

**o**: 저장됨. **o (일부)**: 첫 상태와 실패 직전 구간에 저장됨. **x**: 표에서 요구하는 해당 스텝 항목으로 저장되지 않음. 학습 중 계산하거나 메모리 버퍼에 사용하는 것과 파일로 보존하는 것은 구분한다.

예를 들어 평가 식별자는 `route`·`seed`, 시간은 초 단위 `t`, 종료 결과는 `outcome`으로 남는다. 이것을 원본 양식의 정수형 `episode`·정수 timestep·매 스텝 Boolean 종료 이벤트와 동일한 저장 필드로 간주하지 않았다. 보상은 누적 집계가 있지만 매 스텝 reward는 없다.

### 1위: 기존 보상 유지

|항목|형태|의미|포함 여부|
|---|---|---|---|
|episode|Integer|Episode 번호|x|
|t|Integer|Episode 내부 timestep|x|
|observation|Vector (63-dim)|시점 t에서 AV Policy가 실제로 입력받는 부분 관측 oₜ|o (일부)|
|action_raw|Vector (2-dim)|Policy가 출력한 원본 Action [가감속, 차선변경]|o (일부)|
|action.accel_raw|Float|환경에 적용된 가감속 Action 값|o (일부)|
|action.lane_change_raw|Float|환경에 적용된 차선변경 raw 값|o (일부)|
|action.lane_change|Integer|실제 명령으로 변환된 차선변경 값 {-1, 0, +1}|x|
|action.lane_change_applied|Boolean|차선변경 명령이 실제 적용되었는지 여부|x|
|reward|Float|시점 t에서 받은 Reward rₜ|x|
|terminated|Boolean|충돌 / 도착 등 환경 자체의 종료 여부|x|
|truncated|Boolean|최대 Episode Step 등 시간 제한에 의한 종료 여부|x|
|state|Dictionary|SUMO 내부의 privileged simulator state sₜ|x|
|detected_vehicles|List|Observation 시점에 감지된 주변 차량 정보|x|
|event.collided|Boolean|해당 step에서 충돌 발생 여부|x|
|event.arrived|Boolean|목적지 도착 여부|x|
|event.lane_before|Integer / String|Action 적용 전 AV 차선|o (일부)|
|event.lane_after|Integer / String|Action 적용 후 AV 차선|x|
|next_observation|Vector (63-dim)|다음 시점의 부분 관측 oₜ₊₁|x|
|next_state|Dictionary|다음 시점의 privileged simulator state sₜ₊₁|x|

팀별 추가 항목:

|항목|형태|의미|포함 여부|
|---|---|---|---|
|reward_components|Dictionary|경로·안전 보상별 누적값과 발생 횟수|o|
|following|Dictionary|차로 선택·실제 변경·위험 시 감속 집계|o|

### 2위: 경로 선택·유지 보상 2배

|항목|형태|의미|포함 여부|
|---|---|---|---|
|episode|Integer|Episode 번호|x|
|t|Integer|Episode 내부 timestep|x|
|observation|Vector (63-dim)|시점 t에서 AV Policy가 실제로 입력받는 부분 관측 oₜ|o (일부)|
|action_raw|Vector (2-dim)|Policy가 출력한 원본 Action [가감속, 차선변경]|o (일부)|
|action.accel_raw|Float|환경에 적용된 가감속 Action 값|o (일부)|
|action.lane_change_raw|Float|환경에 적용된 차선변경 raw 값|o (일부)|
|action.lane_change|Integer|실제 명령으로 변환된 차선변경 값 {-1, 0, +1}|x|
|action.lane_change_applied|Boolean|차선변경 명령이 실제 적용되었는지 여부|x|
|reward|Float|시점 t에서 받은 Reward rₜ|x|
|terminated|Boolean|충돌 / 도착 등 환경 자체의 종료 여부|x|
|truncated|Boolean|최대 Episode Step 등 시간 제한에 의한 종료 여부|x|
|state|Dictionary|SUMO 내부의 privileged simulator state sₜ|x|
|detected_vehicles|List|Observation 시점에 감지된 주변 차량 정보|x|
|event.collided|Boolean|해당 step에서 충돌 발생 여부|x|
|event.arrived|Boolean|목적지 도착 여부|x|
|event.lane_before|Integer / String|Action 적용 전 AV 차선|o (일부)|
|event.lane_after|Integer / String|Action 적용 후 AV 차선|x|
|next_observation|Vector (63-dim)|다음 시점의 부분 관측 oₜ₊₁|x|
|next_state|Dictionary|다음 시점의 privileged simulator state sₜ₊₁|x|

팀별 추가 항목:

|항목|형태|의미|포함 여부|
|---|---|---|---|
|reward_components|Dictionary|경로·안전 보상별 누적값과 발생 횟수|o|
|following|Dictionary|차로 선택·실제 변경·위험 시 감속 집계|o|

### 3위: 경로 보상 + 변경 기회 미사용 벌점

|항목|형태|의미|포함 여부|
|---|---|---|---|
|episode|Integer|Episode 번호|x|
|t|Integer|Episode 내부 timestep|x|
|observation|Vector (63-dim)|시점 t에서 AV Policy가 실제로 입력받는 부분 관측 oₜ|o (일부)|
|action_raw|Vector (2-dim)|Policy가 출력한 원본 Action [가감속, 차선변경]|o (일부)|
|action.accel_raw|Float|환경에 적용된 가감속 Action 값|o (일부)|
|action.lane_change_raw|Float|환경에 적용된 차선변경 raw 값|o (일부)|
|action.lane_change|Integer|실제 명령으로 변환된 차선변경 값 {-1, 0, +1}|x|
|action.lane_change_applied|Boolean|차선변경 명령이 실제 적용되었는지 여부|x|
|reward|Float|시점 t에서 받은 Reward rₜ|x|
|terminated|Boolean|충돌 / 도착 등 환경 자체의 종료 여부|x|
|truncated|Boolean|최대 Episode Step 등 시간 제한에 의한 종료 여부|x|
|state|Dictionary|SUMO 내부의 privileged simulator state sₜ|x|
|detected_vehicles|List|Observation 시점에 감지된 주변 차량 정보|x|
|event.collided|Boolean|해당 step에서 충돌 발생 여부|x|
|event.arrived|Boolean|목적지 도착 여부|x|
|event.lane_before|Integer / String|Action 적용 전 AV 차선|o (일부)|
|event.lane_after|Integer / String|Action 적용 후 AV 차선|x|
|next_observation|Vector (63-dim)|다음 시점의 부분 관측 oₜ₊₁|x|
|next_state|Dictionary|다음 시점의 privileged simulator state sₜ₊₁|x|

팀별 추가 항목:

|항목|형태|의미|포함 여부|
|---|---|---|---|
|reward_components|Dictionary|경로·안전 보상별 누적값과 발생 횟수|o|
|following|Dictionary|차로 선택·실제 변경·위험 시 감속 집계|o|

## 3. observation 63차원 전체 목록

아래는 **위 상위 세 모델의 실제 설정** 기준이며 인덱스는 Python과 동일하게 0부터 시작한다. 입력 길이가 같아도 다른 실험은 예비 칸의 의미가 달라질 수 있다.

- 기준 속도 `V = 27.78 m/s`, 차량 감지 거리 `W = 50 m`, 내비게이션 탐색 거리 `H = 300 m`.
- 차로 순서: 왼쪽 두 번째(+2), 왼쪽(+1), 현재(0), 오른쪽(−1), 오른쪽 두 번째(−2).
- 상대속도는 선행·후행 모두 **상대 차량 속도 − AV 속도**다. 후행 거리는 음수로 입력한다.
- 차량이 없을 때 선행 거리·속도는 `(1, 1)`, 후행은 `(−1, −1)`. 따라서 부재 표시는 실제 극값과 겹칠 수 있다.
- 존재하는 차량의 음수 gap은 0으로 처리한다. 기본 31차원 결과는 [−1, 1]로 제한한다.

|인덱스|관측 내용|입력값·정규화|상위 3개 모델|
|---:|---|---|---|
|0|AV 현재 속도|AV 속도 / V|사용|
|1|왼쪽 두 번째 차로 선행차 거리|gap / W; 없으면 1|사용|
|2|왼쪽 두 번째 차로 선행차 상대속도|clip((선행차 속도 − AV 속도) / V, −1, 1); 없으면 1|사용|
|3|왼쪽 두 번째 차로 후행차 거리|−gap / W; 없으면 −1|사용|
|4|왼쪽 두 번째 차로 후행차 상대속도|clip((후행차 속도 − AV 속도) / V, −1, 1); 없으면 −1|사용|
|5|왼쪽 차로 선행차 거리|gap / W; 없으면 1|사용|
|6|왼쪽 차로 선행차 상대속도|clip((선행차 속도 − AV 속도) / V, −1, 1); 없으면 1|사용|
|7|왼쪽 차로 후행차 거리|−gap / W; 없으면 −1|사용|
|8|왼쪽 차로 후행차 상대속도|clip((후행차 속도 − AV 속도) / V, −1, 1); 없으면 −1|사용|
|9|현재 차로 선행차 거리|gap / W; 없으면 1|사용|
|10|현재 차로 선행차 상대속도|clip((선행차 속도 − AV 속도) / V, −1, 1); 없으면 1|사용|
|11|현재 차로 후행차 거리|−gap / W; 없으면 −1|사용|
|12|현재 차로 후행차 상대속도|clip((후행차 속도 − AV 속도) / V, −1, 1); 없으면 −1|사용|
|13|오른쪽 차로 선행차 거리|gap / W; 없으면 1|사용|
|14|오른쪽 차로 선행차 상대속도|clip((선행차 속도 − AV 속도) / V, −1, 1); 없으면 1|사용|
|15|오른쪽 차로 후행차 거리|−gap / W; 없으면 −1|사용|
|16|오른쪽 차로 후행차 상대속도|clip((후행차 속도 − AV 속도) / V, −1, 1); 없으면 −1|사용|
|17|오른쪽 두 번째 차로 선행차 거리|gap / W; 없으면 1|사용|
|18|오른쪽 두 번째 차로 선행차 상대속도|clip((선행차 속도 − AV 속도) / V, −1, 1); 없으면 1|사용|
|19|오른쪽 두 번째 차로 후행차 거리|−gap / W; 없으면 −1|사용|
|20|오른쪽 두 번째 차로 후행차 상대속도|clip((후행차 속도 − AV 속도) / V, −1, 1); 없으면 −1|사용|
|21|왼쪽 두 번째 차로 연결성|차로 없음 −1; 시야 내 연결 유지 1; 끝나면 남은 거리 / W|사용|
|22|왼쪽 차로 연결성|차로 없음 −1; 시야 내 연결 유지 1; 끝나면 남은 거리 / W|사용|
|23|현재 차로 연결성|차로 없음 −1; 시야 내 연결 유지 1; 끝나면 남은 거리 / W|사용|
|24|오른쪽 차로 연결성|차로 없음 −1; 시야 내 연결 유지 1; 끝나면 남은 거리 / W|사용|
|25|오른쪽 두 번째 차로 연결성|차로 없음 −1; 시야 내 연결 유지 1; 끝나면 남은 거리 / W|사용|
|26|왼쪽 두 번째 차로 전방 밀도|clip(전방 차량 수 / (사용 가능 거리 / 7.5m), 0, 1); 차로 없음 1|사용|
|27|왼쪽 차로 전방 밀도|clip(전방 차량 수 / (사용 가능 거리 / 7.5m), 0, 1); 차로 없음 1|사용|
|28|현재 차로 전방 밀도|clip(전방 차량 수 / (사용 가능 거리 / 7.5m), 0, 1); 차로 없음 1|사용|
|29|오른쪽 차로 전방 밀도|clip(전방 차량 수 / (사용 가능 거리 / 7.5m), 0, 1); 차로 없음 1|사용|
|30|오른쪽 두 번째 차로 전방 밀도|clip(전방 차량 수 / (사용 가능 거리 / 7.5m), 0, 1); 차로 없음 1|사용|
|31|바로 다음 도로로 갈 수 있는 차로의 상대 방향|가장 가까운 연결 가능 차로와 현재 차로의 인덱스 차 / 3, [−1,1] 제한|사용|
|32|전방 차로 제약 지점까지 남은 거리|거리 / H; 탐색 범위 내 지점 없으면 −1|사용|
|33|전방 제약을 통과하기 위해 현재 준비할 차로의 상대 방향|실제 차로 연결을 역추적한 준비 차로와 현재 차로의 인덱스 차 / 3, [−1,1] 제한|사용|
|34|현재 차로의 제한속도|min(현재 차로 제한속도 / V, 1)|사용|
|35|현재 도로가 램프인지|도로 유형이 `_link`이면 1, 아니면 0|사용|
|36|지퍼 합류 교대 상태|관련 ZIP 대상이 비어 있어 항상 0|비활성·0 고정|
|37|안내 명확화 확장용 예비 칸 1|clarity=baseline이므로 0|비활성·0 고정|
|38|안내 명확화 확장용 예비 칸 2|clarity=baseline이므로 0|비활성·0 고정|
|39|안내 명확화 확장용 예비 칸 3|clarity=baseline이므로 0|비활성·0 고정|
|40|안내 명확화 확장용 예비 칸 4|clarity=baseline이므로 0|비활성·0 고정|
|41|안내 명확화 확장용 예비 칸 5|clarity=baseline이므로 0|비활성·0 고정|
|42|안내 명확화 확장용 예비 칸 6|clarity=baseline이므로 0|비활성·0 고정|
|43|안내 명확화 확장용 예비 칸 7|clarity=baseline이므로 0|비활성·0 고정|
|44|안내 명확화 확장용 예비 칸 8|clarity=baseline이므로 0|비활성·0 고정|
|45|안내 명확화 확장용 예비 칸 9|clarity=baseline이므로 0|비활성·0 고정|
|46|안내 명확화 확장용 예비 칸 10|clarity=baseline이므로 0|비활성·0 고정|
|47|관측 비교 실험용 예비 칸 1|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|48|관측 비교 실험용 예비 칸 2|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|49|관측 비교 실험용 예비 칸 3|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|50|관측 비교 실험용 예비 칸 4|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|51|관측 비교 실험용 예비 칸 5|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|52|관측 비교 실험용 예비 칸 6|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|53|관측 비교 실험용 예비 칸 7|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|54|관측 비교 실험용 예비 칸 8|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|55|관측 비교 실험용 예비 칸 9|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|56|관측 비교 실험용 예비 칸 10|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|57|관측 비교 실험용 예비 칸 11|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|58|관측 비교 실험용 예비 칸 12|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|59|관측 비교 실험용 예비 칸 13|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|60|관측 비교 실험용 예비 칸 14|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|61|관측 비교 실험용 예비 칸 15|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|
|62|관측 비교 실험용 예비 칸 16|condition=baseline; 상세 안내·접근 위험 관측을 켜지 않아 0|비활성·0 고정|

### 해석할 때 주의할 점

- **0~35번은 실제 관측 항목 36개**, 36~62번은 이 세 모델에서 0 고정인 27개 칸이다. 네트워크 입력 형상은 여전히 `(63,)`이다.
- 31·33번의 0은 올바른 차로에 있는 경우뿐 아니라 안내 대상 차로를 계산하지 못한 경우에도 나온다. 이 세 모델은 별도 안내 유효성 플래그를 켜지 않았다.
- 연결성은 다음 도로로의 링크 존재를 보는 값이다. 지정 목적지에 맞는 연결 차로 여부는 31번 내비게이션으로 제공한다.
- 밀도는 필요할 때 현재 차로의 첫 번째 후속 링크까지 이어서 계산한다. junction 내부는 밀도 0·연결성 1로 처리한다.
- 34번 제한속도는 `traci.lane.getMaxSpeed()` 기반이다. 31~33번은 정적 도로 연결과 지정 경로로 계산하며 강제 차선변경 명령이 아니다.
- 최근 합류 관측·모방 실험에서 사용하는 60~62번(합류 위험·자차/상대 예상시간)은 **이 문서의 상위 세 모델에서는 활성화되지 않았다**.

## 4. 확인 근거

- 저장된 첫 상태·실패 직전 관측 367개에서 벡터 길이 63 및 36~62번 0 고정을 확인했다. 저장 표본만의 확인이며 전체 주행 로그를 복원한 것은 아니다.
- 세 조건의 환경 생성 설정과 아래 관측 구현을 대조했다. 핵심 관측 소스 7개는 실험 manifest의 SHA256과 현재 파일이 일치했다.

- [env/sumo_env.py](../../env/sumo_env.py)
- [env/ic_experiment.py](../../env/ic_experiment.py)
- [env/navigation_observation.py](../../env/navigation_observation.py)
- [env/navigation_clarity.py](../../env/navigation_clarity.py)
- [env/observation_reward_trials.py](../../env/observation_reward_trials.py)
- [env/route_following_trials.py](../../env/route_following_trials.py)
- [env/traffic_route_recovery.py](../../env/traffic_route_recovery.py)
- [utils/run_traffic_safety_trials.py](../../utils/run_traffic_safety_trials.py)

원시 평가 파일은 용량 관리 정책에 따라 로컬 `results/`에 보관한다. 아래 해시는 문서에 사용한 파일을 식별한다.

|로컬 평가 파일|SHA256|
|---|---|
|`results/traffic_recovery_auto_20261116/round-02/runs/control/policy-151552-evaluation.json`|`125bc3619953161bd71f77cc5b0f324d5be42a52fd89bf99213534c490ed436d`|
|`results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000-evaluation.json`|`15e738e7f73f2483f668ee1e95f1742ef1368102d9fd2ea1d0d265c5f7e0da88`|
|`results/traffic_recovery_auto_20261116/round-02/runs/opportunity/policy-51200-evaluation.json`|`72afd0b442c5296dbfb7ed15346c15dbca3cf94360a7c09e80efc79014f2c73f`|

원본 변환 대상 PPTX SHA256: `add385e6a527f8999c02d91dc716a4f5605ceacb878077c957e3ba475da50cf4`.
