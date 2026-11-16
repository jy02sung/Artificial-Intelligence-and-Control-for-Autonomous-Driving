# 퇴계원IC AV 데이터 칼럼 정의

작성일 2026-11-16. 대상 모델은 `gap-lane-22`, 기준 환경은 팀이 배포한 `hv_high15000_entryprep`이다. 제공된 칼럼 설명 PPT의 필수 19개 항목과 우리 모델의 끼어들기 관측, 실제 평가 파일의 칼럼을 정리했다.

**데이터 단위와 현재 저장 상태**

| 데이터 | 한 행의 단위 | 현재 상태 |
|---|---|---|
| 스텝별 전이 데이터 | 한 에피소드의 한 제어 스텝 | 기존 수집기에 필수 19개 필드의 기록 로직이 있다. 최종 모델의 전체 스텝 데이터 수집에는 해당 환경과 정책 설정을 연결해야 한다. |
| 팀 추가 끼어들기 정보 | 행동 실행 직전의 한 관측 | 최종 모델의 관측 인덱스 37~40에서 계산한다. 아래에 제안한 별도 칼럼명으로 저장하려면 수집기에 기록 항목을 추가해야 한다. |
| 개별 평가 결과 | 한 모델·교통 시드·경로의 주행 한 번 | 최종 모델의 1,600행 CSV가 모델 패키지에 저장돼 있다. |
| 경로별 성적 | 한 경로의 100회 평가 집계 | 16행 CSV가 저장소에 포함돼 있다. |

아래 표에서 **O는 해당 필드의 존재 또는 저장 완료**, **X는 해당 필드로 저장한 자료의 부재**를 뜻한다. 각 표의 열 제목이 확인 대상을 명시한다. 스텝별 데이터의 정책 입력은 57차원이다. 제공 양식의 19차원 표기는 이 모델에 맞춰 57차원으로 작성한다.

## 1. 필수 스텝별 칼럼 19개

한 행은 시점 t의 관측과 상태, 행동, 행동 실행 후의 보상·종료 판정, 다음 관측과 상태를 나타낸다. 현재 시뮬레이션의 한 제어 스텝은 0.2초다.

| 칼럼명 | 형태·단위 | 의미 | 기존 수집기 필드 정의 | 최종 평가 CSV에 해당 필드 저장 |
|---|---|---|---|---|
| `episode` | Integer | 에피소드 번호. 기존 수집기는 0부터 부여한다. | O | X |
| `t` | Integer | 에피소드 내부 스텝 번호. 행동 실행 직전 기준이며 0부터 시작한다. | O | X |
| `observation` | Vector, Float 57개 | AV 정책이 행동 결정에 사용한 현재 부분 관측 | O | X |
| `action_raw` | Vector, Float 2개 | 정책에서 전달한 `[가감속, 차선변경]` 행동 | O | X |
| `action.accel_raw` | Float, −1~1 | 환경이 받은 정규화 가감속 명령. 양수는 가속, 음수는 감속이다. | O | X |
| `action.lane_change_raw` | Float, −1~1 | 환경이 받은 차선변경 원시 명령 | O | X |
| `action.lane_change` | Integer, −1·0·+1 | 변환된 명령. 오른쪽 −1, 유지 0, 왼쪽 +1 | O | X |
| `action.lane_change_applied` | Boolean | 환경이 해당 차로 명령을 적용했는지 여부 | O | X |
| `reward` | Float | 해당 스텝의 행동 실행 후 받은 보상 | O | X |
| `terminated` | Boolean | 충돌·도착·차량 제거 등 환경의 종료 사건 여부 | O | X |
| `truncated` | Boolean | 최대 스텝 수 등 시간 제한 도달 여부 | O | X |
| `state` | Dictionary | 행동 실행 전 SUMO 내부 상태 | O | X |
| `detected_vehicles` | List of Dictionary | 행동 실행 직전 수집한 감지 차량 목록 | O | X |
| `event.collided` | Boolean | 해당 스텝에서 AV가 충돌했는지 여부 | O | X |
| `event.arrived` | Boolean | 해당 스텝에서 도착했는지 여부. 최종 모델 환경은 지정 경로의 마지막 도로에서 도착했는지도 확인한다. | O | X |
| `event.lane_before` | Integer / null | 행동 실행 전 차로 인덱스 | O | X |
| `event.lane_after` | Integer / null | 행동 실행 후 차로 인덱스. AV가 제거되면 null이다. | O | X |
| `next_observation` | Vector, Float 57개 | 행동 실행 후 다음 시점의 부분 관측 | O | X |
| `next_state` | Dictionary | 행동 실행 후 다음 시점의 SUMO 내부 상태 | O | X |

기존 수집기의 필드 지원은 `collect_pomdp_data.py`의 기록 로직을 확인한 결과다. 최종 모델에 적용할 때는 `EntryEnv(condition="gap_lane")`와 해당 모델의 manifest에 기록된 정책 설정을 연결해야 한다. 기존 수집 명령의 기본 환경과 정책 설정은 별도로 정의돼 있다.

`action.lane_change_applied=True`에는 차로 유지 명령을 적용한 경우도 포함된다. 실제 차로 이동은 같은 도로 구간에서 `lane_before`와 `lane_after`가 달라졌는지로 판정한다. 차로 인덱스는 SUMO 기준으로 오른쪽부터 0, 1, 2 순서다.

가감속 raw 값의 단위는 무차원이다. 실제 가속 명령은 양수일 때 `accel_raw × max_accel`, 음수일 때 `accel_raw × max_decel`로 계산한다. 기존 수집기의 PPO·BC 행동은 전달 전에 −1~1 범위로 제한된다.

JSONL 수집기는 중간 스텝의 `next_observation`·`next_state`를 다음 행의 현재 값에서 복원하며, 종료 스텝에는 두 값을 직접 기록한다. `iter_transitions()`가 복원한다. NPZ에는 고정 크기 관측·행동 배열 등이 저장되고, 전체 중첩 상태는 JSONL에 저장된다. [수집 코드](../../collect_pomdp_data.py)

## 2. 팀 추가 끼어들기 칼럼 4개

이 네 항목은 제공 양식에 개별 칼럼으로 제시되지 않은 우리 모델의 추가 정보다. 모델은 현재 주변 차량의 간격·속도와 경로 정보를 이용해 값을 계산한다. 아래 이름은 관측 값을 별도 칼럼으로 기록할 때 사용할 이름이다.

| 추가 칼럼명 | 형태·범위 | 의미 | 관측 위치 | 런타임 계산 | 별도 칼럼 저장 |
|---|---|---|---|---|---|
| `merge_direction` | Integer, −1·0·+1 | 필요한 차로 이동 방향. 오른쪽 −1, 해당 상황 없음 0, 왼쪽 +1 | `observation[37]` | O | X |
| `merge_gap_status` | Integer, −1·0·+1 | 목표 차로의 앞뒤 간격 기준 충족 여부. 부족 −1, 해당 상황 없음 0, 충족 +1 | `observation[38]` | O | X |
| `merge_speed_error_norm` | Float, −1~1 | 접근 목표 속도와 현재 속도의 차이를 최대 감속도로 나눈 뒤 범위를 제한한 값 | `observation[39]` | O | X |
| `merge_required_now` | Boolean, 0·1 | 현재 경로 연결에 필요한 즉시 차로 변경 조건이 있는지 여부 | `observation[40]` | O | X |

접근 속도 입력은 다음과 같이 계산한다. 별도 칼럼은 이 관측 값을 그대로 사용한다.

```text
merge_speed_error_norm = clip((접근 목표 속도 − 현재 AV 속도) / 최대 감속도, −1, 1)
```

음수는 목표 속도가 현재 속도보다 낮다는 뜻이고, 양수는 높다는 뜻이다. 이 값은 코드에서 정규화 입력으로 사용한다. 속도와 감속도로 나누는 식의 차원은 시간에 해당하므로 이를 실제 가속도 값으로 해석하지 않는다.

간격 판정은 아래 기준을 사용한다. 거리 단위는 m, 속도 단위는 m/s다.

```text
앞 필요 간격 = 3 + 0.35 × AV 속도 + 1.25 × max(0, AV 속도 − 앞차 속도)
뒤 필요 간격 = 3 + 0.35 × 뒤차 속도 + 1.25 × max(0, 뒤차 속도 − AV 속도)
```

앞뒤 실제 간격이 기준을 초과하면 충족으로 판정한다. 목표 차로의 50m 관측 범위에서 차량이 없는 상태도 해당 방향이 비어 있는 것으로 처리한다. `merge_gap_status=1`은 이 계산 기준의 충족을 뜻하며, 실행 후의 실제 충돌 여부는 `event.collided`가 기록한다.

현재 차로 방향이나 전방 준비 방향이 있고, 인접 차로가 존재하며, 출발 후 약 22m 이상 진행한 상황에서 이 입력을 구성한다. 해당 상황이 없으면 네 값이 모두 0이다. `merge_required_now=0`은 전방 차로 변경 준비 상황에도 나타날 수 있으므로, 다른 세 값과 함께 해석한다. [끼어들기 계산](../../env/gap_reward.py), [관측 생성](../../env/entryprep_av.py)

## 3. 관측 벡터 57차원의 구성

인덱스는 0부터 시작한다. 기본 주변 차로 순서는 왼쪽 두 번째, 왼쪽 첫 번째, 현재, 오른쪽 첫 번째, 오른쪽 두 번째다. 정규화 기준 `W`는 50m, `vmax`는 환경의 AV 최고속도다.

| 인덱스 | 개수 | 내용·인코딩 |
|---|---|---|
| `0` | 1 | 현재 AV 속도 / vmax |
| `1~20` | 20 | 다섯 차로 각각의 앞차 거리, 앞차 상대속도, 뒤차 거리, 뒤차 상대속도 |
| `21~25` | 5 | 다섯 차로의 연결성. 차로 없음 −1, 시야 안에서 계속 연결 1, 중간 종료는 남은 거리 비율 |
| `26~30` | 5 | 다섯 차로의 전방 차량 밀도. 없는 차로는 1로 표시 |
| `31` | 1 | 현재 도로에서 다음 경로로 연결되는 차로까지의 방향·차이 / 3, −1~1 범위 제한 |
| `32` | 1 | 전방 차로 제약까지 거리 / 300m. 해당 제약을 찾지 못하면 −1 |
| `33` | 1 | 전방 차로 제약에 대비하기 위한 차로 방향·차이 / 3, −1~1 범위 제한 |
| `34` | 1 | 현재 차로 제한속도 / vmax |
| `35` | 1 | 램프 구간 여부 |
| `36` | 1 | 공통 IC 환경의 지퍼 합류 맥락 값. 현재 EntryEnv는 해당 전용 구간 목록을 비워 이 값이 0이다. |
| `37~40` | 4 | 팀 추가 끼어들기 정보 네 개 |
| `41~46` | 6 | 현재 모델에서 0으로 채우는 예약 입력 |
| `47` | 1 | 확장 전방 위험 정보의 존재 여부 |
| `48` | 1 | 확장 전방 거리 / 120m. 앞차 또는 합류 충돌 예상 지점까지 거리 |
| `49` | 1 | 확장 전방 상대속도 / vmax. 합류 예상 지점은 정지 장애물 형태로 인코딩 |
| `50` | 1 | SUMO 경로 안내의 적합 차로까지 차이 / 4, −1~1 범위 제한 |
| `51` | 1 | 출발 후 누적 주행거리 / 100m. 100m 이상은 1로 포화 |
| `52` | 1 | 현재 차로 끝까지 남은 거리 / 300m, 상한 1. 내부 교차로에서는 0 |
| `53~56` | 4 | 왼쪽 앞·뒤, 오른쪽 앞·뒤 차량의 간격 부족 정도. 0~1 |
| **합계** | **57** | 최종 정책 입력 |

`1~20`의 차로 블록 번호를 k=0~4로 두면 다음과 같다.

| 인덱스 식 | 의미 | 차량이 없을 때 값 |
|---|---|---|
| `1+4k` | 앞차 간격 / W | 1 |
| `2+4k` | (앞차 속도 − AV 속도) / vmax, −1~1 범위 제한 | 1 |
| `3+4k` | −뒤차 간격 / W | −1 |
| `4+4k` | (뒤차 속도 − AV 속도) / vmax, −1~1 범위 제한 | −1 |

차량 없음 표시는 관측 범위의 경계값과 겹칠 수 있다. 이 값을 실제 차량의 거리·속도로 복원할 때는 해당 차량의 존재 여부도 확인해야 한다.

확장 전방 관측은 앞차를 최대 120m까지 조회한다. 다른 진입로 차량에 대해서는 80m 이내 합류 지점의 도달 시점 겹침을 계산한다. `47~49`에는 그중 현재 위험으로 선택한 정보를 넣는다. `53~56`의 간격 부족 정도는 별도 위험도 계산이며, 2절의 끼어들기 간격 판정과 각각의 계산식을 사용한다. [관측 구현](../../env/entryprep_av.py), [경로 안내 구현](../../env/navigation_observation.py)

## 4. 중첩 상태와 감지 차량의 하위 칼럼

`state`와 `next_state`는 같은 구조다. 전자는 행동 전, 후자는 행동 후 상태를 나타낸다.

| 하위 칼럼 | 형태·단위 | 의미 |
|---|---|---|
| `state.ego_alive` | Boolean | AV가 시뮬레이션에 존재하는지 여부 |
| `state.ego_x`, `state.ego_y` | Float, m | SUMO 지도 좌표계에서 AV의 위치 |
| `state.ego_speed` | Float, m/s | AV의 실제 속도 |
| `state.ego_lane` | Integer | AV 차로 인덱스 |
| `state.ego_lane_id` | String | AV 차로 ID |
| `state.ego_road_id` | String | AV 도로 구간 ID |
| `state.ego_lane_pos` | Float, m | 현재 차로 시작점에서 AV까지의 종방향 위치 |
| `state.vehicles` | List of Dictionary | 내부 상태 수집 시 조회한 시뮬레이션 차량 목록. AV도 포함될 수 있다. |
| `state.vehicles[].id` | String | 차량 ID |
| `state.vehicles[].x`, `state.vehicles[].y` | Float, m | 차량의 지도 좌표 |
| `state.vehicles[].speed` | Float, m/s | 차량 속도 |
| `state.vehicles[].lane` | Integer | 차로 인덱스 |
| `state.vehicles[].lane_id` | String | 차로 ID |
| `state.vehicles[].road_id` | String | 도로 구간 ID |
| `state.vehicles[].lane_pos` | Float, m | 현재 차로 내 종방향 위치 |

AV가 제거된 시점에는 내부 상태 함수가 `ego_alive=False`만 반환한다. 나머지 상태 키는 생략될 수 있다. 정책은 `observation`을 입력으로 받으며, 전체 `state`는 데이터 분석·검증을 위한 별도 자료다.

| 감지 차량 칼럼 | 형태·단위 | 의미 |
|---|---|---|
| `detected_vehicles[].id` | String | 감지 차량 ID |
| `detected_vehicles[].x`, `detected_vehicles[].y` | Float, m | 감지 차량 지도 좌표 |
| `detected_vehicles[].rel_pos` | Float, m | AV 기준 종방향 위치 차이. 앞은 양수, 뒤는 음수 |
| `detected_vehicles[].speed` | Float, m/s | 감지 차량 속도 |
| `detected_vehicles[].lane_offset` | Integer | AV 차로 기준 상대 차로. 왼쪽은 양수, 오른쪽은 음수 |

현재 감지 목록 함수는 같은 도로의 좌우 두 차로 범위와 종방향 ±50m 안의 배경 차량을 수집한다. 내부 교차로 또는 AV가 없는 상태에서는 빈 목록을 반환한다. `rel_pos`는 차량 위치 차이로 정의되며, 차체 길이를 반영한 앞뒤 간격과 정의가 다르다. 최종 모델이 추가로 조회하는 120m 전방·80m 합류 정보를 이 목록으로 함께 보존하려면 감지 목록 수집 범위도 확장해야 한다. [상태·감지 목록 구현](../../env/sumo_env.py)

## 5. 실제 평가 CSV의 칼럼 13개

아래 모든 칼럼의 현재 저장 여부는 O이다. 한 파일 안에서 `(seed, route)` 조합이 각 주행을 식별한다. 다른 모델의 결과와 합칠 때는 모델 실행 폴더 또는 정책 해시도 식별 정보로 함께 사용한다.

| 칼럼명 | 형태·단위 | 의미 | 저장 여부 |
|---|---|---|---|
| `seed` | Integer | 교통 난수 시드 | O |
| `route` | String | `route_01`~`route_16`의 경로 ID | O |
| `outcome` | String | 최종 종료 결과 | O |
| `seconds` | Float, 초 | AV 삽입 후 종료까지의 제어 시간. 삽입 실패 시 0으로 기록된다. | O |
| `edge` | String | 종료 직전 도로 구간 ID | O |
| `insert_wait_seconds` | Float, 초 | AV가 실제로 삽입될 때까지 기다린 시간. 삽입 실패 시 대기 제한값을 기록한다. | O |
| `route_overrides` | Integer, 회 | 환경이 보고한 강제 경로 보정 횟수 | O |
| `collision_partner` | String | 충돌 상대 차량 ID. 여러 ID는 쉼표로 연결하며 해당 기록이 없으면 빈 값이다. | O |
| `last_speed_mps` | Float, m/s | 마지막 행동 실행 직전의 AV 속도 | O |
| `last_accel` | Float, −1~1 | 마지막 정책 가감속 명령 | O |
| `last_lane` | Integer, −1·0·+1 | 마지막 차선변경 명령 | O |
| `lane_changes` | Integer, 회 | 같은 도로에서 실제로 차로가 바뀐 횟수 | O |
| `reversals_within_2s` | Integer, 회 | 같은 도로에서 차로 변경 후 2초 이내에 반대 방향으로 다시 변경한 횟수 | O |

삽입 실패 시 주행 중에만 얻을 수 있는 위치·최종 행동 값은 빈 값으로 남을 수 있다. `route_overrides`는 경로 보정 계수이며 다른 모든 제어 개입의 유무를 이 값만으로 판정할 수는 없다.

| `outcome` 값 | 의미 |
|---|---|
| `arrival` | 지정 경로 마지막 도로에서 도착 이벤트 확인 |
| `collision` | AV 충돌 이벤트 확인 |
| `timeout` | 주행 시간 제한 도달 |
| `teleport` | AV 순간이동 시작 이벤트 확인 |
| `insertion_failure` | AV 삽입 대기 제한 도달 |
| `wrong_exit` | 목적지 도착 조건을 충족하지 못한 기타 환경 종료 |

실제로 저장된 행의 예시다.

```csv
seed,route,outcome,seconds,edge,insert_wait_seconds,route_overrides,collision_partner,last_speed_mps,last_accel,last_lane,lane_changes,reversals_within_2s
33001,route_02,arrival,136.4,844484877#1,0.0,0,,19.569,0.01414,0,2,0
```

최종 평가의 교통 시드는 33001~33100이다. 1,600행에서 완주 1,599회, 충돌 0회, 시간초과 1회가 기록됐다. 실제 차로 변경은 총 799회, 2초 이내 반대 방향 재변경은 0회이며, 삽입 대기시간은 0~25.4초였다. `lane_changes`와 `reversals_within_2s`는 제공 PPT에 없는 실제 저장 데이터다.

원자료는 [모델 패키지](../../results/entryprep_av/av-models-22.zip) 안의 `results/entryprep_av/gap-lane-22/eval-snapshot-33001-33100-all-900s.csv`에 있다. [평가 기록 코드](../../utils/entryprep_av.py)

## 6. 경로별 성적 CSV의 칼럼 11개

모든 칼럼의 저장 여부는 O이며, 한 행은 해당 경로의 100회 평가 집계다.

| 칼럼명 | 형태·단위 | 의미 |
|---|---|---|
| `route` | String | 경로 ID |
| `runs` | Integer, 회 | 해당 경로의 전체 평가 횟수 |
| `arrival` | Integer, 회 | 완주 횟수 |
| `collision` | Integer, 회 | AV 충돌 횟수 |
| `timeout` | Integer, 회 | 시간초과 횟수 |
| `teleport` | Integer, 회 | 순간이동 종료 횟수 |
| `insertion_failure` | Integer, 회 | 삽입 실패 횟수 |
| `wrong_exit` | Integer, 회 | 목적지 도착 조건을 충족하지 못한 기타 종료 횟수 |
| `arrival_pct` | Float, % | `100 × arrival / runs` |
| `collision_pct` | Float, % | `100 × collision / runs` |
| `rates_pass` | Boolean | 완주율 95% 이상이며 AV 충돌률 5% 미만인지 여부 |

시간초과·삽입 실패 등 모든 종료 유형을 `runs`에 포함한다. 현재 05번은 99/100회 완주, 나머지 15개 경로는 각각 100/100회 완주했다. [경로별 결과 CSV](ic-entryprep-av-gap-lane-rates.csv), [집계 코드](../../utils/entryprep_score.py)

## 7. 실험 조건과 재현 정보

주행 결과의 해석에는 모델 및 평가 설정을 함께 보관해야 한다. 다음 항목은 모델 패키지의 JSON 파일에 저장돼 있다.

| 파일 | 주요 키 | 의미 |
|---|---|---|
| `manifest.json` | `condition`, `model_seed`, `observations`, `params` | 환경 조건, 모델 학습 시드, 관측 차원, PPO 설정 |
| `manifest.json` | `traffic_seeds`, `routes`, `warmup_seconds` | 학습 교통 시드, 학습 경로, HV 선행 주행 시간 |
| `manifest.json` | `inputs`, `source`, `base_commit` | 기준 교통 파일·코드 해시와 학습 당시 기준 커밋 |
| `manifest.json` | `gap_demonstrations`, `assisted_initialization`, `initial_policy_sha256` | 시범 학습 사용 여부와 초기 정책 식별 정보 |
| `manifest.json` | `speedMode`, `laneChangeMode`, `lane_change_safety` | 기록된 AV 제어·안전 필터 설정 |
| `complete.json` | `policy_sha256` | 학습 완료 정책 파일의 해시 |
| 평가 폴더의 `protocol.json` | `policy`, `inputs`, `software`, `sumo`, `backend` | 실제 평가에서 사용한 정책·입력·코드·SUMO·연결 방식 |
| 평가 폴더의 `protocol.json` | `protocol`, `drive_limit`, `routes` | 교통 초기화 방식, 실제 주행 제한, 평가 경로 |

최종 1,600회 평가의 초기화 방식은 `snapshot`, 주행 제한은 900초다. 학습 manifest의 기본 평가 제한과 특정 평가의 실제 제한은 각각의 파일에 기록되므로, 개별 결과를 해석할 때는 해당 `protocol.json`을 확인한다. 학습 교통 시드 12001~12008과 최종 평가 교통 시드 33001~33100은 분리돼 있다.

현재 모델은 규칙 기반 시범 학습과 사람이 설계한 끼어들기 관측 정보를 활용했다. 이 학습 방법과 입력 구성을 실험 결과와 함께 설명한다. [모델 개발·평가 보고서](ic-entryprep-av-personal.md)
