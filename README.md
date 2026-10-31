# Artificial-Intelligence-and-Control-for-Autonomous-Driving

SUMO 기반 자율주행 인공지능 및 제어 팀 프로젝트입니다. `jys` 브랜치에는 V의 **퇴계원IC 도로 환경**을 추가했습니다. 기존 수업 고속도로 예제와 학습 코드는 그대로 보존합니다.

## 퇴계원IC 실행 (jys)

- [도로 환경 안내·가정·다음 단계](scenarios/toegyewon/README.md)
- Windows: `scenarios/toegyewon/open_sumo.cmd` 더블클릭 후 재생
- [교통량 및 도로 기본 설정](scenarios/toegyewon/parameters.json)
- 빨간 ego 1대 + 일반 승용차·버스. 현재 ego는 도로 검증용 IDM이며 학습된 PPO는 후속 연결 대상입니다.
- [검증 결과](scenarios/toegyewon/validation.json)
- [기본 제어기 데이터 수집 안내](scenarios/toegyewon/COLLECTION.md) — 두 경로·교통량 3단계·4종 제어기 혼합, 모델 학습 없음
- [4종 제어기 동일 조건 비교·BC 데이터](scenarios/toegyewon/CONTROLLER_COMPARISON.md) — ego 제어기별 30회, IDM 기준 수치 비교와 학습 분할

## 프로젝트 바로가기

- [프로젝트 웹페이지](https://jy02sung.github.io/Artificial-Intelligence-and-Control-for-Autonomous-Driving/)
- [Canva 발표자료 편집](https://canva.link/s8q5mppy42hkaay) — 제목·목차·주요 섹션 초안, 도로 및 결과 미정

발표자료는 Canva에서 관리하며 저장소에는 링크만 제공합니다. Canva 접근·편집 권한은 Canva의 공유 설정을 따릅니다. 웹페이지는 `docs/index.html`에서 수정하며 GitHub Pages로 게시합니다.

## 팀 문서

- [문서 목록](docs/README.md)
- [프로젝트 최종 목적 및 완료 체크리스트](docs/project-goals-checklist.md) — 수업 목표와 팀 권장 작업 구분
- [기존 학습 데이터와 추가 권장 데이터](docs/data-inventory-and-proposals.md) — 도로 미정, 현재 구현과 향후 제안 구분

## 구성

- SUMO 도로·교통류 생성 및 GUI 확인
- Gymnasium / TraCI 환경: 관측 31차원, 가감속·차선 변경 행동
- PPO 강화학습, BC 모방학습, 데이터 수집, 모델 평가 및 TensorBoard
- 도로·차량·관측·행동·보상·학습 파라미터와 기존 수업 자료
- 원본 샘플 데이터 `data/dataset.npz`

이전 개인 실험의 모델·로그 및 별도 과제용 회전교차로는 포함하지 않습니다.

## 설치

Python 3.10 이상과 [Eclipse SUMO](https://eclipse.dev/sumo/)를 설치합니다. SUMO의 bin 폴더를 PATH에 추가하고 SUMO_HOME을 설치 폴더로 설정합니다. 기본 확인 환경은 Python 3.10 / SUMO 1.27.1입니다.

```bash
git clone https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving
python -m venv .venv
```

가상환경 활성화: Windows 명령 프롬프트에서는 `.venv\Scripts\activate.bat`, macOS/Linux에서는 `source .venv/bin/activate`를 실행합니다.

```bash
python -m pip install -r requirements.txt
sumo --version
python view_road.py --nogui --seconds 10
```

SUMO 실행 파일은 Python 패키지와 별도로 설치해야 합니다. GPU 학습 시 장치에 맞는 PyTorch 빌드를 설치하세요.

## 실행

```bash
# 도로 확인
python view_road.py
# PPO 학습
python train.py
# 학습 그래프: http://localhost:6006
python -m tensorboard.main --logdir results
# 모델 평가: 실제 생성된 run 경로로 교체
python test.py results/run_YYYYMMDD_HHMMSS/model.pt --episodes 100 --nogui
# 모델 GUI 재생
python test.py results/run_YYYYMMDD_HHMMSS/model.pt
# 데이터 파이프라인 확인용 임의 정책 (전문가 시범 정책이 아님)
python collect_pomdp_data.py --policy random --episodes 10 --name demo
# BC 실행 확인
python train_bc.py --data data/demo.npz
```

좋은 BC 정책에는 양질의 시범 데이터가 필요합니다. 기존 `data/dataset.npz`는 원본 샘플로 보존했으며 현재 관측·행동 차원과 일치하는지 확인 후 사용하세요.

## 코드와 파라미터

| 파일 | 역할 |
| --- | --- |
| `env/road_config.py` | 도로 길이·차로 수·속도, ego, 일반 차량 모델과 교통량 |
| `env/mdp_config.py` | 시간 설정, 관측·행동, 기본 보상 |
| `env/road_builder.py` | SUMO 도로·경로·설정 파일 생성 |
| `env/sumo_env.py` | reset/step, TraCI 제어, 관측·보상·종료 판정 |
| `env/sumo/` | 기본 고속도로 XML 및 sumocfg |
| `train.py` | PPO 하이퍼파라미터, 보상 덮어쓰기, 학습 |
| `train_bc.py` / `collect_pomdp_data.py` | BC 학습 / 주행 데이터 수집 |
| `algorithms/` | PPO 및 BC 구현 |
| `utils/` | 신경망, 버퍼, 로깅, 평가 |
| `test.py` / `view_road.py` | 모델 평가 / 도로 확인 |
| `project_materials/` | 기존 수업 자료 |

현재 로컬 작업본 설정을 유지합니다: `lr=0.0003`, 목표 `200000`스텝, `n_steps=2048`, `n_epochs=10`, `minibatch_size=64`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, 은닉층 `(256,256)`, `seed=0`. 전체 값은 코드가 기준입니다. rollout 단위로 학습하므로 실제 스텝은 목표보다 조금 많을 수 있습니다.

기본 도로는 12km, 편도 3차로이며 일반 차량은 Krauss·IDM·EIDM·ACC 혼합입니다. `train.py`의 `REWARD_OVERRIDES`가 기본 보상을 덮어씁니다.

**실행 시 SUMO 파일이 다시 생성됩니다.** NETEDIT에서 XML만 수정하면 덮어써질 수 있습니다. 실제 교차로 확장 시 도로 생성뿐 아니라 경로, 관측, 보상, 종료 판정도 검토해야 합니다. `test.py`는 현재 설정을 사용하므로 과거 모델 평가 시 해당 run 설정과 일치시켜야 합니다.

## 팀 협업

1. 담당 작업 브랜치 생성: `git switch -c feature/작업명`
2. 수정 후 짧은 실행으로 확인하고 브랜치를 push합니다.
3. main 대상 Pull Request로 변경 내용과 검증 결과를 공유하고 병합합니다.
4. 실험 조건·결과 요약을 남기고 대용량 데이터·모델은 별도로 공유합니다.

results, 생성 데이터, 가상환경, 캐시 및 비밀 설정 파일은 .gitignore로 제외합니다. 원본 샘플 데이터만 포함합니다. 팀원 쓰기 권한은 GitHub Settings → Collaborators에서 계정을 초대하여 부여합니다.

## 출처

- [수업 원본 저장소](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving)
- 기준 커밋: `d958d059b8a9a60c94dd99ae8b16f10575a00287` 및 로컬 수정 사항
- 준비 날짜: 2026-10-27
- 원본 README 보존본: `project_materials/UPSTREAM_README.md` (내부 상대 링크는 원본 위치 기준)
- [프로젝트 준비 수업 자료](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving/blob/main/project_materials/5_Project_Preparation.md)

원본 코드·자료의 저작권은 원저작자에게 있습니다. 복사한 원본에 별도 LICENSE 파일이 없어 임의의 오픈소스 라이선스를 추가하지 않았습니다.

### Windows 콘솔 인코딩

`UnicodeEncodeError`가 발생하면 UTF-8 모드로 실행하세요: `python -X utf8 view_road.py` 또는 `python -X utf8 train.py`. PowerShell에서는 현재 세션에 `$env:PYTHONUTF8="1"`을 설정해도 됩니다.

### 초기 검증 (2026-10-27)

Python 구문 검사, SUMO 10초 headless 실행, 환경 reset/10 step, PPO 초기화를 확인했습니다. 관측 `(31,)`, 행동 `(2,)`입니다. 새 저장소에서 전체 학습을 다시 실행한 것은 아닙니다.
