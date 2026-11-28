# 퇴계일황 · 퇴계원 IC 자율주행 코드 제출본

**최종 제출 모델은 `models/dagger5.pt`입니다.** 이 모델은 SUMO에서 퇴계원 IC의 지정 경로를 따라 주행합니다. 배경 교통량을 시간당 15,000대로 설정한 평가에서 **576회 중 561회 완주(97.4%), 5회 충돌(0.9%)**을 기록했습니다.

먼저 전문가의 가감속과 차로 선택을 모방하도록 행동 복제(BC)로 학습했습니다. 이후 모델이 직접 주행하며 마주친 상황에 같은 전문가가 정답 행동을 붙이고, 이를 기존 데이터에 더해 다시 학습하는 과정(DAgger)을 5회 반복했습니다.

[팀 GitHub 저장소](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving) · [프로젝트 페이지](https://jy02sung.github.io/Artificial-Intelligence-and-Control-for-Autonomous-Driving/)

[과제함 제출 파일 다운로드 · 퇴계일황_code.zip](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving/raw/refs/heads/main/퇴계일황_code.zip)

## 최종 모델 바로 실행하기 (Windows PowerShell)

1. Python 3.10(64비트)과 Eclipse SUMO 1.27.1을 설치합니다. 설치 시 `SUMO_HOME`을 SUMO 설치 폴더로 지정하고 그 아래 `bin`을 `PATH`에 추가합니다. 검증에는 Python 3.10.2와 CPU PyTorch 2.14.0을 사용했습니다. [SUMO 공식 다운로드](https://sumo.dlr.de/releases/1.27.1/)
2. 제출 ZIP을 풀었다면 **`run.py`와 `requirements.txt`가 있는 폴더**에서 PowerShell을 엽니다. GitHub 저장소를 복제했다면 **`submission` 폴더**에서 엽니다.
3. 처음 실행할 때는 아래 네 줄을 순서대로 입력합니다. 설치를 마친 뒤에는 마지막 줄만 실행하면 제출 모델이 한 번 주행합니다.

```powershell
sumo --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

기본 설정에서는 차량이 8번 경로의 0차로에서 출발합니다(시드 `104320008`). 아래 출력에서 **`steps=437 arrival`은 437스텝 만에 지정된 목적지에 도착했다는 뜻**입니다. 마지막 줄은 종료 사유별 주행 횟수입니다.

```text
dagger5 route=8 lane=0 seed=104320008 steps=437 arrival
{"arrival": 1, "collision": 0, "stall": 0, "timeout": 0, "off_route": 0}
```

### 다른 실행 방법

아래 명령은 위에서 설치한 같은 폴더에서 실행합니다.

```powershell
# 최종 모델의 13번 진출을 SUMO 화면에서 보기
.\.venv\Scripts\python.exe run.py --gui --route 13 --lane 1 --seed 104320013

# 비교용 순수 PPO 모델로 16번 본선 유지 주행
.\.venv\Scripts\python.exe run.py --model pure_ppo --route 16 --lane 0 --seed 104320016

# 최종 모델로 발표와 같은 고교통 576회를 모두 평가
.\.venv\Scripts\python.exe run.py --model dagger5 --holdout
```

`--holdout`은 학습에 쓰지 않은 세 평가 묶음(64회 + 256회 + 256회)으로 16개 경로를 주행합니다. 각 주행은 완주, 충돌, 차로 끝 정지, 시간초과, 경로 이탈 중 하나로 집계됩니다.

모델은 50개 관측값으로 가감속과 차로 명령을 결정하며, 평가 중 같은 관측에는 같은 행동을 선택합니다. 분기에서는 현재 차로가 연결된 도로로 실제 진입할 수 있습니다. 고가 교차부 20쌍에는 높이 차이를 반영한 시야 보정을 적용했고, 안전게이트는 꺼 두었습니다. 순수 PPO 모델은 학습할 때와 같은 분기 방향 보정을 사용합니다.

## 발표에 사용한 가중치

| 모델 | 실행 옵션 `--model` | 평가 횟수 | 완주 | 충돌 |
|---|---|---:|---:|---:|
| **최종 제출 · BC + 보강형 DAgger R5** | **`dagger5`** | **576** | **561 (97.4%)** | **5 (0.9%)** |
| BC + 기본 DAgger R5 | `uniform_dagger5` | 576 | 541 (93.9%) | 15 (2.6%) |
| 보강형 DAgger R5 + KL-PPO | `dagger5_ppo` | 576 | 552 (95.8%) | 9 (1.6%) |
| 순수 PPO · 학습 시드 33, GAE λ=0.95 | `pure_ppo` | 576 | 498 (86.5%) | 25 (4.3%) |
| BC 50에포크 실험 · 검증으로 선택한 48에포크 | `bc50` | 128 | 107 (83.6%) | 17 (13.3%) |
| BC 100에포크 실험 · 검증으로 선택한 94에포크 | `bc100` | 128 | 107 (83.6%) | 13 (10.2%) |

위 네 모델은 동일한 576회 주행으로 비교했습니다. 아래 두 BC 모델은 학습 횟수(50에포크와 100에포크)의 영향을 살펴보기 위해 별도의 동일한 128회 주행으로 비교했습니다. 가중치 파일은 `models/<실행 모델명>.pt`에 있으며, 각 파일의 원본 경로와 SHA-256 해시는 `models/manifest.json`에 기록했습니다.

## 학습 재현

```powershell
.\.venv\Scripts\python.exe train_submission.py dagger5 --workers 4
```

1. **전문가 시범 수집.** `utils/imitation.py`의 `collect_demos()`가 16개 경로에서 전문가를 주행시킵니다. 매 스텝 `utils/expert.py`의 `control()`이 가감속과 차로 명령을 정하고, `utils/storage.py`가 50개 관측값과 그때의 전문가 행동을 주행별 `.npz`에 저장합니다. 학습용으로는 완주한 시범 중 합류를 기다리며 1 m/s 미만으로 10초 이상 연속 주행한 사례를 제외했습니다. 실제 수집한 168회 중 이 기준으로 8회를 제외해 160회를 사용했고, 검증용은 33회 중 시간초과 1회를 제외해 32회를 사용했습니다.
2. **행동 복제(BC).** `utils/imitation.py`의 `fit()`이 학습용 시범 160회로 신경망(은닉층 256×256, ReLU)을 학습합니다. 모델은 관측값을 보고 전문가의 가감속과 차로 명령을 예측하며, 별도 시범 32회로 검증해 `bc.pt`를 선택합니다.
3. **보강형 DAgger.** 이제 BC 모델이 직접 운전하고, 같은 전문가가 모델이 마주친 각 상황에 정답 행동을 붙입니다. `imitation.py`의 `pipeline()`은 매회 144회 주행(16개 경로 × 8회 + 취약 경로 4개 × 4회)에서 얻은 데이터를 기존 시범에 누적해 모델을 다시 학습합니다. 학생 모델의 충돌·경로 이탈 등 실패 주행도 학습 데이터에 포함합니다. 차로변경·실패 직전·모델이 틀린 장면은 더 자주 뽑습니다.
4. **모델 선정.** 5회까지의 평가 결과에 따라 최대 8회까지 학습합니다. 결과와 가중치는 `results/jys3_final_relu_dagger/`에 저장되며, 학습이 끝나면 선정된 파일 경로가 출력됩니다.

5회차 가중치로 주행하는 예시는 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe run.py --model dagger5 --checkpoint results/jys3_final_relu_dagger/dagger5.pt
```

비교 모델을 학습하려면 `train_submission.py` 뒤의 `dagger5`를 아래 모델명으로 바꿉니다.

| 모델명 | 학습 과정 |
|---|---|
| `bc50`, `bc100` | 같은 전문가 시범에서 BC를 50 또는 100에포크 학습하고 검증 점수가 가장 높은 가중치를 선택 |
| `uniform_dagger5` | BC 모델로 시작해 매회 경로당 9회씩 주행하고, 누적 데이터에서 학습 장면을 균등하게 뽑는 과정을 5회 반복 |
| `dagger5_ppo` | 동봉된 제출 모델에서 가치 함수 워밍업 20,000스텝 후 원본 정책에 대한 KL 벌점으로 PPO를 32,768스텝 학습 |
| `pure_ppo` | 시드 33 무작위 초기화에서 Q1 250,000스텝, 가치 함수 워밍업 10,000스텝, M5-FT 125,000스텝, 충돌 벌점 −14 조건에서 100,000스텝 순으로 학습 |

`scenarios/hv_high15000_entryprep/`에 지도·교통 수요·SUMO 설정, `env/`에 시뮬레이터 환경, `algorithms/`와 `utils/`에 정책·학습 구현이 있습니다. 신규 학습 산출물은 `results/`에 생성됩니다.

전문가의 운전 규칙은 `utils/expert.py`, BC·DAgger 학습 과정은 `utils/imitation.py`, 평가에 사용한 SUMO 환경 설정은 `utils/simulator.py`에서 확인할 수 있습니다.
