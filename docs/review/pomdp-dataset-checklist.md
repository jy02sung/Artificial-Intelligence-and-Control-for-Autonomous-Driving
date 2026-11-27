# POMDP 데이터셋 필수 포함 항목 및 팀 추가 항목 정리

> 기준 데이터: [Hugging Face `jy02sung/toegyewon-ic-dagger5-pomdp`](https://huggingface.co/datasets/jy02sung/toegyewon-ic-dagger5-pomdp) — 발표 모델 BC + 보강형 DAgger R5(`dagger5.pt`, SHA-256 `fb3a775f…28cd`)가 고교통 15,000대/시, 16개 경로, 안전게이트 OFF에서 주행한 기록  
> 수집 규모: 28,080회 주행(경로별 1,755회), 전이 10,003,923개 · 종료 사유 완주 27,123 / 충돌 450 / 경로 이탈 409 / 차로 끝 정지 71 / 시간초과 27  
> 저장 포맷: Parquet — `data/steps/`(자율주행차 전이) + `data/tracks/`(맵 전체 차량 궤적), 두 표를 [`reader.py`](https://huggingface.co/datasets/jy02sung/toegyewon-ic-dagger5-pomdp/blob/main/reader.py)의 `transitions()`로 연결해 전이 단위로 읽음

---

## 1. 필수 포함 항목 체크표

| 필수 포함 항목 | 형태 | 의미 | 포함 여부 (o / x) | 비고 |
| :--- | :--- | :--- | :---: | :--- |
| `episode` | Integer | Episode 번호 | **o** | `steps.episode` (int32) |
| `t` | Integer | Episode 내부 timestep | **o** | `steps.t` (int32), 0.2초 간격, 최대 1,000스텝 |
| `observation` | Vector (50-dim) | 시점 $t$에서 AV Policy가 실제로 입력받는 부분 관측 $o_t$ | **o** | `steps.observation` (float32 × 50) |
| `action_raw` | Vector (2-dim) | Policy가 출력한 원본 Action [가감속, 차선변경] | **o** | `steps.action_raw` (float32 × 2) |
| `action.accel_raw` | Float | 환경에 적용된 가감속 Action 값 | **o** | `steps.action.accel_raw`, 범위 [-1, 1] |
| `action.lane_change_raw` | Float | 환경에 적용된 차선변경 raw 값 | **o** | `steps.action.lane_change_raw` — 정책은 좌·유지·우 3클래스를 출력하며, 선택한 클래스를 수치로 기록 |
| `action.lane_change` | Integer | 실제 명령으로 변환된 차선변경 값 `{-1, 0, +1}` | **o** | `steps.action.lane_change` (int32) |
| `action.lane_change_applied` | Boolean | 차선변경 명령이 실제 적용되었는지 여부 | **o** | `steps.action.lane_change_applied` |
| `reward` | Float | 시점 $t$에서 받은 Reward $r_t$ | **o** | `steps.reward` (float32) — 환경이 계산해 기록한 값. 발표 모델은 BC + DAgger로 학습해 이 보상을 학습에 쓰지 않음 |
| `terminated` | Boolean | 충돌 / 도착 등 환경 자체의 종료 여부 | **o** | `steps.terminated`, 종료 사유는 `steps.outcome` |
| `truncated` | Boolean | 최대 Episode Step 등 시간 제한에 의한 종료 여부 | **o** | `steps.truncated` |
| `state` | Dictionary | SUMO 내부의 privileged simulator state $s_t$ | **o** | `steps.state`: `ego_alive`, `ego_x/y/speed/lane/lane_id/road_id/lane_pos` · 맵 전체 차량 `vehicles` 목록은 `data/tracks/`에 저장하고 `transitions()`가 `state["vehicles"]`로 복원 |
| `detected_vehicles` | List | Observation 시점에 감지된 주변 차량 정보 | **o** | `{id, x, y, rel_pos, speed, lane_offset}` 리스트 |
| `event.collided` | Boolean | 해당 step에서 충돌 발생 여부 | **o** | `steps.event.collided` |
| `event.arrived` | Boolean | 목적지 도착 여부 | **o** | `steps.event.arrived` |
| `event.lane_before` | Integer / String | Action 적용 전 AV 차선 | **o** | `Integer` (SUMO lane index, 오른쪽 끝 = 0) |
| `event.lane_after` | Integer / String | Action 적용 후 AV 차선 | **o** | `Integer` (SUMO lane index) |
| `next_observation` | Vector (50-dim) | 다음 시점의 부분 관측 $o_{t+1}$ | **o** | 종료 스텝에만 저장, 나머지는 `transitions()`가 다음 행의 `observation`으로 복원 |
| `next_state` | Dictionary | 다음 시점의 privileged simulator state $s_{t+1}$ | **o** | 종료 스텝에만 저장, 나머지는 `transitions()`가 다음 행의 `state`(차량 목록 포함)로 복원 |

---

## 2. 팀별 추가 항목 (발표 모델 데이터셋 기준)

| 팀별 추가 항목 | 형태 | 의미 |
| :--- | :--- | :--- |
| `observation` 확장 (`50-dim`) | Vector (50-dim) | 자차 속도·차로 제한속도·현재 차로 번호·전체 차로 수, 120m 안 현재 및 좌우 두 차로(5차로)의 앞뒤 차량 간격·상대속도·전방 밀도, 좌우 차체 겹침, 차로별 앞쪽 연속 길이·분기 연결·실선/교차로 내부 여부·도로 끝까지 거리, 최대 400m 앞 목적 도로 방향과 분기까지 거리, 차로별 분기 플래그(5차로 × 2) |
| `route` | String | 현재 에피소드 지정 경로 ID (`route_01` ~ `route_16`) |
| `seed` | Integer | 해당 에피소드의 SUMO 교통류 시드 번호 |
| `outcome` | String | 에피소드 종료 사유 `arrival` / `collision` / `off_route`(지정 경로 밖 도로로 진입) / `stall`(차로 끝 정지) / `timeout`(1,000스텝 도달) |
| `tracks.id` | String | 맵 전체 차량 ID (자율주행차는 `ego`) |
| `tracks.episode`, `tracks.start` | Integer | 궤적 구간이 속한 에피소드와 시작 timestep |
| `tracks.lane`, `tracks.lane_id`, `tracks.road_id` | Integer / String | 궤적 구간 동안의 차로 번호·SUMO 차로 ID·도로 ID |
| `tracks.x`, `tracks.y` | List[Integer] | 차량 좌표(m) — 1/1000 단위 정수의 2차 차분으로 압축, `transitions()`가 m 단위로 복원 |
| `tracks.speed` | List[Integer] | 차량 속도(m/s) — 같은 방식으로 압축 |
| `tracks.lane_pos` | List[Integer] | 차로 시작점부터의 위치(m) — 같은 방식으로 압축 |
| `progress.json` | Dictionary | 수집 블록·주행·전이 수, 종료 사유별 집계, 모델 가중치 SHA-256 |
