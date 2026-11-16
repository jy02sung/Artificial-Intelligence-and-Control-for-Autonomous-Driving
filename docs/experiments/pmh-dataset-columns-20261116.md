# pmh 데이터셋 칼럼 (최선 조건 zipper_filter, 2026-11-16)

## 수집 조건

- 정책: 현재 최선 조건 `zipper_filter`(지퍼 관측 + 지퍼 보상 + 압박 0.01 + 행동 안전 필터)의 30만 스텝 모델. seed 11/22/33/44/55 각각, 결정적 행동.
- 환경: 지퍼 지도 `maps/toegyewon_zipper`, 실측 교통 `low_exit25`(13,949대/시간), 안전 필터 켬, 평가 모드(학습 전용 보상 항 없음).
- 에피소드: seed당 16개(경로 route_01~route_16 각 1회), 총 80개, 38,773 transition. SUMO seed는 50001부터 에피소드마다 1씩 늘린다. 평가 seed(40001~)와 겹치지 않는다.
- 결과: 도착 71(88.8%), 충돌 6, 시간 초과 3. 평가 결과와 같은 수준이다.
- 수집 코드: `utils/pmh_collect_dataset.py`. 기록 형식은 팀 공용 `collect_pomdp_data.collect()`를 그대로 쓴다. 환경과 정책만 바꾸고 팀 항목을 `event.team`에 붙였다.
- 파일(커밋 안 함, `data/`는 gitignore): `data/pmh_zipper_filter/`
  - `pmh_zipper_filter_seed{seed}.jsonl.gz`: 전체 항목, 약 313MB(5개 합계). 읽을 때는 `collect_pomdp_data.iter_transitions()`를 쓴다. 이 함수가 next_observation/next_state를 채운다.
  - `pmh_zipper_filter_seed{seed}.npz`: 학습용 고정 크기 배열.
  - `manifest.json`: 모델 경로, transition 수, 검증 결과.
- 다시 만들기: `python -m utils.pmh_collect_dataset --seeds 11 22 33 44 55 --episodes-per-seed 16`

## 필수 포함 항목

| 필수 포함 항목 | 형태 | 의미 | 포함 여부 |
|---|---|---|:---:|
| episode | Integer | Episode 번호 | o |
| t | Integer | Episode 내부 timestep | o |
| observation | Vector (**47-dim**) | 시점 t에서 AV Policy가 실제로 입력받는 부분 관측 o_t | o (아래 주 1) |
| action_raw | Vector (2-dim) | Policy가 출력한 원본 Action [가감속, 차선변경] | o |
| action.accel_raw | Float | 환경에 적용된 가감속 Action 값 | o |
| action.lane_change_raw | Float | 환경에 적용된 차선변경 raw 값 | o |
| action.lane_change | Integer | 실제 명령으로 변환된 차선변경 값 {-1, 0, +1} | o |
| action.lane_change_applied | Boolean | 차선변경 명령이 실제 적용되었는지 여부 | o (주 2) |
| reward | Float | 시점 t에서 받은 Reward r_t | o |
| terminated | Boolean | 충돌 / 도착 등 환경 자체의 종료 여부 | o |
| truncated | Boolean | 최대 Episode Step 등 시간 제한에 의한 종료 여부 | o |
| state | Dictionary | SUMO 내부의 privileged simulator state s_t | o |
| detected_vehicles | List | Observation 시점에 감지된 주변 차량 정보 | o |
| event.collided | Boolean | 해당 step에서 충돌 발생 여부 | o |
| event.arrived | Boolean | 목적지 도착 여부 | o |
| event.lane_before | Integer / String | Action 적용 전 AV 차선 | o (Integer, 차선 번호) |
| event.lane_after | Integer / String | Action 적용 후 AV 차선 | o (Integer, 차선 번호) |
| next_observation | Vector (**47-dim**) | 다음 시점의 부분 관측 o_{t+1} | o (JSONL은 마지막 스텝만 저장, 주 3) |
| next_state | Dictionary | 다음 시점의 privileged simulator state s_{t+1} | o (JSONL은 마지막 스텝만 저장, 주 3) |

주

1. 표의 19차원은 원래 단일 도로 환경 기준이다. 퇴계원 IC 환경의 관측은 47차원이다.
   - 0~36: 원래 주변 차량 관측에 경로·IC 관측을 더한 칸.
   - 37~44: 이번에 추가한 합류 관측.
   - 45~46: 쓰지 않음(0).
2. 안전 필터를 켠 상태다. 그래서 `lane_change_applied=False`는 두 경우를 포함한다. 갈 차로가 없는 경우와 안전 게이트가 막은 경우다. 둘은 `event.team.lane_change_blocked`로 구분한다.
3. 용량 때문에 중간 스텝의 next_observation/next_state는 다음 레코드의 observation/state와 같아서 저장하지 않는다. `iter_transitions()`로 읽으면 모든 레코드에 채워진다. NPZ의 `next_observation`은 모든 스텝에 있다.

`state` 구성: ego_alive, ego_x, ego_y, ego_speed, ego_lane, ego_lane_id, ego_road_id, ego_lane_pos, vehicles(전체 차량의 id, x, y, speed, lane, lane_id, road_id, lane_pos).

`detected_vehicles` 원소: id, x, y, rel_pos(ego 기준 종방향 거리, 앞 +), speed, lane_offset(ego 차선 기준, +왼쪽).

## 팀별 추가 항목

| 팀별 추가 항목 | 형태 | 의미 |
|---|---|---|
| event.route | String | 이번 에피소드 경로 (route_01~route_16) |
| event.sumo_seed | Integer | SUMO 교통 seed |
| event.ramp | Boolean | ego가 램프 구간에 있음 |
| event.ramp_excess_mps | Float | 램프 제한속도 초과량 (m/s) |
| event.wrong_exit | Boolean | 경로와 다른 출구로 나감 |
| event.route_lane_ready | Boolean / null | 분기 100m 전에 목표 차로에 있었는지 (분기마다 1회) |
| event.exit_lane_change_distance_m | Float / null | 진출 차로 변경 시점의 분기까지 거리 |
| event.zipper_count, event.zipper_compliant | Integer, Boolean / null | 기존 IC 지퍼 판정 |
| event.reward_components | Dictionary | 보상 분해 (base, route, zipper, ramp) |
| event.team.zipper_obs | Vector (8-dim) | 합류 관측 (observation[37:45]): 합류 유무, 거리, 내 차로 합류 여부, 경쟁 차량 유무, 도착 시간차, 속도 차, 거리 차, 합류 후 빈 공간 |
| event.team.merge_pass | Boolean | 이번 스텝에 지퍼 합류 링크에 진입 |
| event.team.merge_simultaneous | Boolean | 경쟁 차량과 동시에 진입 |
| event.team.filter_speed_capped | Boolean | 안전 필터가 속도 명령을 낮춤 |
| event.team.lane_change_blocked | Boolean | 차로 변경 게이트가 명령을 막음 |
| event.team.policy_seed | Integer | 행동한 모델의 학습 seed |

수집 데이터 전체에서 필터의 속도 제한은 14,713 스텝(38%), 차로 변경 차단은 2,448 스텝, 합류 진입은 64회였다.
