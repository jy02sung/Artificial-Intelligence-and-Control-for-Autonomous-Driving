# 발표 영상용 스크립트 시나리오(`demo_scripted`) 구현 및 모델 이식 가이드

> 작성일: 2026-11-25  
> 브랜치 / 기준 코드: `mcw2` · [utils/mcw_demo.py](../../utils/mcw_demo.py), [utils/mcw_train.py](../../utils/mcw_train.py)  
> 대상: **BC + DAgger + PPO 등 신규 학습 모델(>90% 완주율 달성 모델)에 발표용 추월·눈치싸움·가려짐·교차로 시나리오 및 GUI 시각화를 이식하려는 팀원**

---

## 1. 왜 만들었는가? (일반 평가 트래픽 vs 발표용 스크립트 시나리오)

일반 랜덤 교통류(`high_real_ratio` 등)에서는 에이전트가 출발 직후 빈 차선으로 조기 이탈해 **“추월·눈치싸움·가려짐(occlusion) 회피 없이 운 좋게 완주한 것처럼 보이는”** 구간이 많습니다.  
발표 영상(30초+ 실주행 요구)에서 자율주행 의사결정 능력이 한눈에 보이도록 **`ScriptedDemoDirector`**가 에고 차량 위치(`edge`, `pos`, `lane`)에 연동해 **저속 트럭, 가림용 버스+숨은 차량, 측면 추월/병행 차량, 실시간 컷인 차량, 다단 플라툰, X자 교차로 합류 차량**을 동적으로 배치·제어합니다.

---

## 2. 시나리오 구성 요소 전체 아키텍처 ([utils/mcw_demo.py](../../utils/mcw_demo.py))

1. **진입부 직선 구간 연장 (`DEMO_OVERTAKE_EXTENSIONS`, `build_demo_overtake_network`)**
   - 출발 직후 바로 분기·감속으로 들어가지 않고, 충분한 추월·눈치싸움 서사가 전개되도록 4개 진입 엣지를 연장합니다.
   - `52873356` (`+560m`), `86198728` (`+560m`), `544295537` (`+500m`), `781184662#1` (`+450m`) → 주 접근부 길이 `850~1,050m+` 확보.
2. **배경 교통류 + 차종별 외형/속도 분포 (`configure_scripted_flows`, `VEHICLE_TYPE_COLORS`)**
   - 기본 혼잡 교통량(`DEMO_FLOW_COUNTS`)의 **45% 수준(최소 `120 vph`)**만 남겨 도로가 비어 보이지 않으면서도 스크립트 액터가 랜덤 정체에 묻히지 않도록 조절합니다.
   - `<flow color=...>`를 제거하고 `<vType>`별 실사 외형(`real world` 테마) 및 색상을 부여합니다:
     - `ego_type` **빨강** (`passenger/sedan`)
     - `truck_slow` (`maxSpeed=14.5 m/s`, **주황**) / `truck_idm` (`노랑`)
     - `bus_slow` (`maxSpeed=15.5 m/s`) / `bus_idm` (**파랑**)
     - `delivery_idm` (**청록** — GUI 시 불투명 박스밴 50% 진한 청록 / 개방형 50% 연하늘)
     - `*_violator` (`lcCooperative=0`, 비양보 차량 **주황**) / 일반 승용(`car_*`, **연회색**)
3. **고차로(1차로/중간차로) 강제 출발 (`SCRIPTED_DEPART_LANES`)**
   - `route_05`: `"2"` (4차로 중좌측)
   - `route_13`: `"4"` (5차로 최좌측, 한국 기준 1차로)
   - `route_08`: `"2"` (3차로 최좌측, 한국 기준 1차로)
4. **깔끔한 GUI 오버레이 & 상황 설명 캡션 (`DemoOcclusionOverlay.render`, `ScriptedDemoDirector.get_status_caption`)**
   - 시야 원(`mcw_fov_ring`), 차량 주위 링·화살표, 속도 게이지 바 등 번잡한 도형은 제거해 화면을 깔끔하게 유지합니다.
   - 트럭·버스·불투명 박스밴 뒤 반투명 그림자 원뿔(`mcw_occ_cone_*`)과 가려진 차량의 얇은 마젠타 마커(`mcw_occ_tgt_*`)만 유지해 사각지대 원인을 보여줍니다.
   - 현재 시나리오 의도(예: `"추월 · 전방 저속 트럭 추월 시도"`, `"눈치싸움 · 버스 뒤 사각지대 경계 후 추월"`, `"X자 교차로 · 좌측 합류 차량과 엇갈리며 통과"`)를 ego 우측 여백에 작은 배지(`mcw_caption_poi`)와 콘솔 로그로 띄워줍니다.
5. **차선 변경 진동 억제 가드 (`lc_reverse_cooldown=12`, [utils/mcw_train.py](../../utils/mcw_train.py))**
   - 영상에서 `L0 ↔ L1` 와리가리(ping-pong) 현상을 막기 위해, 차선 변경 후 `12스텝(2.4초)` 내 역방향 복귀 및 `4스텝(0.8초)` 내 연속 변경을 억제합니다.
   - 단, 분기 막다른 차로(`dist < 90m`) 또는 전방 위험(`lead < 6.5m` 혹은 `TTC < 0.8s`) 시에는 즉시 해제되어 회피 기동을 허용합니다.

---

## 3. 상위 3개 쇼케이스 루트별 차량 배치·타이밍·상호작용 상세

> 아래 모든 액터는 `ScriptedDemoDirector.step(traci, step_count, ego_id)`에서 조건이 맞으면 `1회(_fired)`만 스폰됩니다.  
> 기준 실측 횟수는 최신 스크립트 검증(`--seed 40001..40203`, 총 9에피소드 9/9 완주 · 충돌 0) 로그 기반입니다.

### 3.1 `route_05` — **“트럭 추월 → 실시간 ego 차선 컷인 회피 → 4대 호송열 합류 → X자 교차”**
- **출발**: 엣지 `52873356` (4차로), `L2` 출발
- **시나리오 타임라인**:

| 단계 (`_fired`) | 발동 조건 | 스폰 액터 (`vid`) | 차선 / 위치 / 속도 | 의도된 상호작용 |
|---|---|---|---|---|
| `r05_init` | `edge == "52873356"` (step 0) | `act05_bus_r` (`bus_idm`)<br>`act05_hid_car` (`car_idm`)<br>`act05_pass_l` (`car_krauss_violator`) | `L1` `pos+18` `16.5m/s`<br>`L1` `pos+65` `14.5m/s`<br>`L3` `pos+8` `27.5m/s` | 우측 버스 그림자 뒤 가려진 승용차 배치 + 좌측 고속 추월 차량 통과 |
| `r05_truck` | `edge == "52873356"` & `pos >= 35m` | `act05_truck` (`truck_slow`) | `L2` `pos+100` `10.0→8.8m/s` | **본선 주행 중 저속 트럭 등장** (즉시 회피 대신 감속·추월 유도) |
| `r05_cutin_spawn` | `edge == "52873356"` & `pos >= 310m` | `act05_cutin` (`car_eidm_violator`) | 인접 차선 `pos+42` `ego_v+1.5` (`14.5~20m/s`) | 스폰 **5스텝 뒤 실시간 ego 현재 차선(`getLaneIndex`)으로 컷인** + `14.5m/s` 감속 |
| `r05_convoy` | `edge == "52873356"` & `pos >= 520m` | `act05_conv1` (`bus_idm`)<br>`act05_conv2` (`car_idm`)<br>`act05_conv3` (`delivery_idm`)<br>`act05_conv4` (`car_eidm`) | `L0` `pos+170` `16.0m/s`<br>`L0` `pos+132` `16.0m/s`<br>`L0` `pos+88` `16.0m/s`<br>`L1` `pos+108` `16.5m/s` | 출구 우측 차선(`L0`) 3대+1대 호송열 사이의 ~38~44m 틈을 노려 **눈치싸움 합류** |
| `r05_xcross` | `edge == "170059991"` & `pos >= 10m` | `act05_weave` (`truck_idm`)<br>`act05_x1` (`delivery_idm`)<br>`act05_x2` (`car_idm_violator`) | `170059991` `L1` `62m` `8.0m/s`<br>`170059995` `L0` `34m` `12.5m/s`<br>`170059995` `L0` `80m` `13.0m/s` | **X자 교차 합류(`170059994`)**: 좌측 합류부(`170059995`) 차량들과 엇갈리며 진출 (`844484876`) |

- **관찰된 행동 빈도 (`3 seeds`)**: 트럭 추월 **1회** (`act05_truck`), 전체 차선 변경 **5~7회** (진입 횡단 2회 + 직조/X-교차 구간 3~5회).

---

### 3.2 `route_13` — **“5차로 최좌측 출발 → 버스·숨은 차 눈치싸움 → 5차로 다단 직조 → X자 교차”**
- **출발**: 엣지 `86198728` (5차로), `L4`(1차로) 출발
- **시나리오 타임라인**:

| 단계 (`_fired`) | 발동 조건 | 스폰 액터 (`vid`) | 차선 / 위치 / 속도 | 의도된 상호작용 |
|---|---|---|---|---|
| `r13_init` | `edge == "86198728"` (step 0) | `act13_side` (`car_krauss_violator`)<br>`act13_p3b` (`bus_slow`)<br>`act13_hid` (`car_idm`) | `L3` `pos+22` `27.5m/s`<br>`L3` `pos+72` `16.5m/s`<br>`L2` `pos+112` `15.0m/s` | 초반 우측 차로 장벽 + `L3` 저속 버스 뒤 `L2` 가려진 차량(`act13_hid`) 경계 유도 |
| `r13_truck` | `edge == "86198728"` & `pos >= 30m` | `act13_truck` (`truck_slow`) | `L4` `pos+98` `10.2→9.2m/s` | 1차로 전방 저속 트럭 등장 → 감속 후 우측 틈새 진입·추월 |
| `r13_weave` | `edge == "86198728"` & `pos >= 300m` | `act13_m2a` / `m2b`<br>`act13_m1a` / `m1b`<br>`act13_r0a` (`route_16`) / `r0b` (`route_13`) | `L2` `pos+120, +72` (`15m/s`)<br>`L1` `pos+98, +52` (`15~15.5m/s`)<br>`L0` `pos+175, +135` (`14.5, 18m/s`) | 5차로 전체에 ~45m 포켓을 둔 다단 직조(weave) 플라툰 → 좌우 눈치싸움 |
| `r13_xcross` | `edge == "170059996"` | `act13_x1` (`delivery_idm`)<br>`act13_x2` (`car_idm`) | `170059991` `L2` `35m, 85m` (`16, 17m/s`)<br>*(사전에 `170059994_2`, `170059991_2` 랜덤 정체 정리)* | `route_05` 방향으로 X-교차하는 `act13_x1/x2`와 짧은 `170059994`(51.1m)에서 교차 후 우측 이탈(`1063527748`) |

- **관찰된 행동 빈도 (`3 seeds`)**: 트럭 추월 **1회** (`act13_truck`), 전체 차선 변경 **7~9회** (초반 4개 차선 단계 횡단 중 `act13_p3b`/`act13_hid` 뒤 `v=7.2~7.5m/s` 감속·눈치싸움 포함 + 중반 직조 추월 2회 + X-교차 이탈 2~3회).

---

### 3.3 `route_08` — **“저속 트럭 우측 추월 눈치싸움 → 램프 대기열 우회 → 지퍼(Zipper) 합류”**
- **출발**: 엣지 `544295537` (3차로), `L2`(1차로) 출발
- **시나리오 타임라인**:

| 단계 (`_fired`) | 발동 조건 | 스폰 액터 (`vid`) | 차선 / 위치 / 속도 | 의도된 상호작용 |
|---|---|---|---|---|
| `r08_init` | `edge == "544295537"` (step 0) | `act08_bus` (`bus_idm`, `keep_lane=False`)<br>`act08_hid` (`car_idm`) | `L1` `pos+14` `19.5m/s`<br>`L1` `pos+80` `17.5m/s` | 옆 차선 버스 통과 및 전방 차량 가려짐(`OCC_CHECK`) 연출 |
| `r08_truck` | `edge == "544295537"` & `pos >= 40m` | `act08_truck` (`truck_slow`) | `L2` `pos+96` `10.2→7.6m/s`<br>*(L1 `pos+115~240` 배경 정체 정리)* | 1차로에서 매우 느린 트럭 뒤 감속(`1.8~5.5m/s`) → 우측 `L1` 틈 확인 후 추월 |
| `r08_merge` | `edge == "544295537"` & `pos >= 280m` | `act08_q1` / `q2` (`route_07`, `L0`)<br>`act08_mid` (`delivery_idm`, `L1`) | `L0` `pos+185, +145` (`11.5m/s`)<br>`L1` `pos+150` (`15.5m/s`, `keep_lane=False`) | 우측 출구 대기열과 중간 차선 배송차 사이의 합류 경쟁 |
| `r08_zipper` | `edge == "544295537"` & `pos >= 650m` | `act08_z1` (`car_eidm`)<br>`act08_z2` (`delivery_idm`) | `169728849` `L0` `240m, 185m` (`15m/s`) | `844484877` 지퍼 합류 구간에서 우측 합류 차량과 교대 합류 |

- **관찰된 행동 빈도 (`3 seeds`)**: 트럭 추월 **1회** (`40003`, `40103` — `seed=40001` 실행 시 3개 루트 모두 트럭 추월 포함), 전체 차선 변경 **2~4회**.

---

## 4. SUMO 스크립트 작성/튜닝 시 핵심 해결 포인트 (이식 시 주의)

다른 브랜치·환경에 `ScriptedDemoDirector`를 가져다 쓸 때 아래 5가지를 지키지 않으면 SUMO 삽입 실패나 교차로 정지가 발생합니다.

1. **`_resolve_subroute()`로 중간 엣지 sub-route 생성**: `traci.vehicle.add(vid, route_id, departPos=...)`는 `route_id`의 **첫 번째 엣지**에 차량을 넣으려 하므로, 중간 엣지(`170059991`, `170059995`, `169728849`) 스폰 시에는 `edges[edges.index(edge_id):]`로 잘라 등록한 `__sub_<route>_<edge>` 경로를 써야 합니다.
2. **차체 길이 + 안전 차간거리 확보 후 `_clear_corridor()`**: `12m` 버스 바로 뒤에 `20m` 미만 간격으로 다른 차량을 두면 SUMO가 0스텝 삽입을 미뤄(delayed insertion) ego가 지나간 뒤 뒤늦게 스폰됩니다.
3. **`departSpeed` 상한 클램핑 + 차로 끝 처리**: `max(1.0, vType.maxSpeed * 0.98)` 초과 또는 `pos > lane_len - 55m`에서 고정 속도를 넣으면 삽입 오류가 나므로 `"desired"`로 대체합니다.
4. **교차부 이후 액터는 `setLaneChangeMode(..., 514)` 또는 `keep_lane=False`**: `setLaneChangeMode(vid, 0)`으로 묶어 두면 합류부(`170059994`, `844484877`)에서 전략적 차선 변경을 못 해 속도가 `0`으로 멈춰 데드락을 유발합니다.
5. **실시간 컷인 `setLaneChangeMode(cid, 512)`**: 컷인 시점에 `ego`의 실제 현재 차선을 다시 조회(`getLaneIndex`)하고, `512` 모드로 충돌 안전 체크를 켠 채 `changeLane` + `slowDown`을 호출해야 측면 충돌 없이 자연스러운 끼어들기가 나옵니다.

---

## 5. 신규 모델(BC + DAgger + PPO)에 쉽게 이식하는 방법

### 방법 A. `47차원` `ClarityEnv` / `utils/mcw_train.py` 호환 체크포인트인 경우 (수정 없이 바로 실행)
새 모델이 `algorithms/ppo.py`(`PPO`) 형식이고 관측 47차원·행동 2차원(`[accel, lane_change]`)이면 **`--resume` 경로만 바꿔서 바로 실행**할 수 있습니다.

**Windows `cmd` 기준:**
```cmd
cd /d C:\Users\문시원\Artificial-Intelligence-and-Control-for-Autonomous-Driving && C:\Users\문시원\AppData\Local\Programs\Python\Python39\python.exe -m utils.mcw_train --stage demo_scripted --resume <새_모델_체크포인트.pt> --eval-only --eval-episodes 1 --eval-routes route_05 route_13 route_08 --no-safety --visibility 120.0 --obs-noise --agenda2 --occlusion --gui --gui-delay 60 --gui-zoom 2000 --seed 40001
```

*(macOS/Linux `bash`에서는 `python` 경로와 `--agenda2`/`--occlusion` 여부를 본인 모델 학습 환경에 맞게 선택)*

> **체크 포인트**: 신규 모델이 `--occlusion`이나 `120m + noise`, `agenda2` 없이(예: 기존 31차원/47차원 기본 관측) 학습되었다면, 먼저 모델이 학습된 동일한 환경 플래그(`--no-occlusion`, `--no-agenda2`, `--no-obs-noise` 등)를 맞춰 테스트하고 성능 변화를 확인하세요.

---

### 방법 B. 다른 Env 클래스 / 다른 관측 차원(예: 31차원, 57차원, 63차원)에 시나리오만 이식할 경우 (4단계)

`utils/mcw_demo.py`는 **특정 정책 구조나 관측 차원에 의존하지 않고 순수 `traci`만 사용**하므로, 본인 환경 파일(`reset()`, `step()`)에 아래 4곳만 연결하면 됩니다.

1. **시나리오 준비 (`prepare_scenario` 참고)**
   - 맵 확장: `build_demo_overtake_network(base_net, dest_net)`
   - 교통류·차종 색상: `configure_scripted_flows(rou_root)` + `apply_real_vehicle_colors_and_shapes(rou_root)`
   - GUI 설정: `write_demo_gui_settings(dest_dir, delay_ms=60.0)` + `attach_gui_settings_to_sumocfg(cfg_path, "highway.gui.xml")`
2. **`reset()` 훅 연결**
   - `self.ego_depart_lane = SCRIPTED_DEPART_LANES.get(route, self.ego_depart_lane)` 적용
   - `self._director.reset(route)` 호출 → `super().reset(...)` 완료 직후 `self._director.step(traci, 0, self.ego_id)` 호출
3. **`step()` 훅 연결**
   - `obs, reward, done, cut, info = super().step(action)` 직후 `if not done: self._director.step(traci, self.step_count, self.ego_id)` 호출
   - *(권장)* 영상에서 차선 와리가리가 보이면 `utils/mcw_train.py` 728~756행의 `lc_reverse_cooldown` 로직(10~15줄)을 함께 복사
4. **GUI 사각지대·상황 캡션 오버레이 연결 (선택)**
   - `gui=True`일 때 스텝마다 `caption = self._director.get_status_caption(traci, self.step_count, self.ego_id, occ_count=len(occ_targets), lc_cmd=lc_cmd, lead_gap=lead_gap)` 호출 후 `self._overlay.render(traci, sx, sy, visibility, occ_boxes, occ_targets, ex=ex, ey=ey, caption=caption)` 연결 (`ex`, `ey`, `caption` 생략 가능)
