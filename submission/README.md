# 퇴계일황 · 퇴계원 IC 자율주행 코드 제출본

**최종 제출 모델은 `models/dagger5.pt`입니다.** 전문가 시범으로 행동 복제(BC)를 학습한 뒤, 학생이 실제로 주행하면서 수집한 상태에 전문가 행동을 붙여 보강형 DAgger를 5회 적용했습니다. 발표에서 비교한 고교통 576회 중 **561회 완주(97.4%), 5회 충돌(0.9%)**을 기록했습니다.

[팀 GitHub 저장소](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving) · [프로젝트 페이지](https://jy02sung.github.io/Artificial-Intelligence-and-Control-for-Autonomous-Driving/)

[과제함 제출 파일 다운로드 · 퇴계일황_code.zip](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving/raw/refs/heads/main/퇴계일황_code.zip)

## 최종 모델 바로 실행하기 (Windows PowerShell)

1. Python 3.10(64비트)과 Eclipse SUMO 1.27.1을 설치합니다. 설치 시 `SUMO_HOME`을 SUMO 설치 폴더로 지정하고 그 아래 `bin`을 `PATH`에 추가합니다. 검증에는 Python 3.10.2와 CPU PyTorch 2.14.0을 사용했습니다. [SUMO 공식 다운로드](https://sumo.dlr.de/releases/1.27.1/)
2. ZIP을 풀고 `run.py`가 보이는 폴더에서 PowerShell을 엽니다. Git으로 받은 경우에는 저장소의 `submission` 폴더에서 엽니다.
3. 아래 명령을 순서대로 실행합니다. 마지막 줄이 제출 가중치로 한 번 주행하는 명령입니다.

```powershell
sumo --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

기본 주행은 8번 경로, 출발 0차로, 시드 104320008을 사용합니다. 정상 실행 시 마지막 두 줄은 다음과 같습니다.

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

평가는 결정론적 정책으로 실시합니다. 16개 경로에서 고교통 수요 15,000대/시, 물리적 분기, 고가 교차부 20쌍의 시야 보정, 정책 입력 50차원, 안전게이트 해제를 적용합니다. `--holdout`은 64회와 256회 두 묶음을 합친 576회 전체를 주행하고 완주, 충돌, 차로 끝 정지, 시간초과, 경로 이탈을 집계합니다. 순수 PPO에는 학습 때 사용한 분기 방향 관측 보정을 적용합니다.

## 발표에 사용한 가중치

| 구분 | 실행 모델명 | 평가 주행 | 완주 | 충돌 |
|---|---|---:|---:|---:|
| **최종 제출 · BC + 보강형 DAgger R5** | **`dagger5`** | **576** | **561 (97.4%)** | **5 (0.9%)** |
| BC + 기본 DAgger R5 | `uniform_dagger5` | 576 | 541 (93.9%) | 15 (2.6%) |
| 보강형 DAgger R5 + KL-PPO | `dagger5_ppo` | 576 | 552 (95.8%) | 9 (1.6%) |
| 순수 PPO · 학습 시드 33, GAE λ=0.95 | `pure_ppo` | 576 | 498 (86.5%) | 25 (4.3%) |
| BC 50에포크 실험 · 검증으로 선택한 48에포크 | `bc50` | 128 | 107 (83.6%) | 17 (13.3%) |
| BC 100에포크 실험 · 검증으로 선택한 94에포크 | `bc100` | 128 | 107 (83.6%) | 13 (10.2%) |

576회는 세 홀드아웃을 합산한 발표 성적이고, BC 두 행은 같은 128회 선정 패널에서 에포크 예산의 영향을 비교한 성적입니다. 각 파일명은 실행 모델명에 `.pt`를 붙인 것입니다. 원본 실험 경로와 SHA-256은 `models/manifest.json`에 있습니다.

## 학습 재현

```powershell
.\.venv\Scripts\python.exe train_submission.py dagger5 --workers 4
```

위 명령은 최종 전문가로 16개 경로에서 학습 시범 160회와 검증 시범 32회를 수집한 뒤, 256×256 ReLU 신경망의 BC와 보강형 DAgger를 학습합니다. DAgger는 매회 균등 주행 128회와 취약 경로 16회를 추가하고, 실패 직전·차로변경·오답 장면을 더 자주 학습합니다. 5회까지의 선정 패널 성적에 따라 최대 8회까지 진행하며 결과와 가중치는 `results/jys3_final_relu_dagger/`에 저장합니다. 재학습한 가중치로 한 번 주행하려면 다음처럼 경로를 지정합니다.

```powershell
.\.venv\Scripts\python.exe run.py --model dagger5 --checkpoint results/jys3_final_relu_dagger/dagger5.pt
```

비교 모델도 같은 진입점에서 모델명을 지정합니다.

| 모델명 | 학습 과정 |
|---|---|
| `bc50`, `bc100` | 같은 전문가 시범에서 BC를 50 또는 100에포크 학습하고 검증 점수가 가장 높은 가중치를 선택 |
| `uniform_dagger5` | BC에서 시작해 경로당 9회씩 학생 주행을 수집하고 균일 미니배치로 5회 갱신 |
| `dagger5_ppo` | 동봉된 제출 모델에서 가치 함수 워밍업 20,000스텝 후 원본 정책에 대한 KL 벌점으로 PPO를 32,768스텝 학습 |
| `pure_ppo` | 시드 33 무작위 초기화에서 Q1 250,000스텝, 가치 함수 워밍업 10,000스텝, M5-FT 125,000스텝, 충돌 벌점 −14 조건에서 100,000스텝 순으로 학습 |

`scenarios/hv_high15000_entryprep/`에 지도·교통 수요·SUMO 설정, `env/`에 시뮬레이터 환경, `algorithms/`와 `utils/`에 정책·학습 구현이 있습니다. 신규 학습 산출물은 `results/`에 생성됩니다.

코드를 살펴볼 때는 전문가 제어기 `utils/expert.py`, BC·DAgger 학습 `utils/imitation.py`, SUMO 환경 생성 `utils/simulator.py` 순으로 보면 됩니다.
