# pmh 분기 준비 보상 단독 비교

상태: 보상 단위 검사 및 libsumo 1.27.1에서 두 조건 × 2048스텝 학습/16경로 평가 통합 검증 완료. 본 실험 결과는 완료 후 추가한다.

기준 코드: cb753c04e8e3e3c00266bd2a528379624c40d601.
회의 모델 파일이 없어 사용자가 새 학습 비교를 선택했다. 회의 모델의 개선율을 재현하는 실험이 아니다.

- 대조군: 기존 보상. 수정군: 분기 300m 이내, 목적지 연결 차로까지 필요한 차로 변경 횟수가 한 단계 줄어들면 +0.1.
- 같은 edge/분기/목표 차로 집합/진척 단계에는 에피소드당 한 번만 지급. 정지 유지·멀어지는 이동에는 추가 보상 없음. 종료·충돌 스텝에는 추가 지급 없음.
- 두 군 모두 47차원 baseline 관측, 기존 300m 안내, 퇴계원 단독 지도, 저수요, warmup 300, best/max 출발, 기존 행동·안전 설정 유지.
- seed 11/22/33, 각 200000 스텝. 동일 seed는 동일 초기 state_dict를 사용한다. PPO 설정은 기존 계획의 hparams 사용.
- 평가: 보상 수정 없는 공통 환경, 무교통/저수요 각각 16경로 × 20개 seed(40001..40020). 완주·충돌·기타 실패와 경로별 원인을 기록한다.
- 지도·교통량·관측·TTC·안전거리 보상을 동시에 변경하지 않는다.
- 초기 가중치, 모델·지도 해시, 설정, 환경 버전, 진행 상황, 최종 평가를 results 아래 보존한다.

실행 전 짧은 검증:

```bash
python -m unittest discover -s tests -p test_prepare_reward.py
IC_LIBSUMO=1 python -m utils.pmh_prepare_trial --steps 2048 --seeds 11 --count 1 --workers 2 --output results/pmh_prepare300_smoke
```

통합 검증 후 본 실험:

```bash
IC_LIBSUMO=1 python -m utils.pmh_prepare_trial --workers 3
```

기존 결과 디렉터리는 덮어쓰지 않는다. 미완료 실행은 최종 결과로 보고하지 않는다.
판정은 동일 seed 대조군 대비 완주 증가, 충돌 비증가, 경로 선택 실패 감소 여부다.
새 학습의 짧은 예산에서 두 군 모두 실패하면 분기 보상이 무효라고 단정하지 않는다.

## 실행 환경

기존 `sumo-rl` 환경의 Python/PyTorch를 사용한다. 본 비교는 `libsumo`, `traci`, `sumolib`, `sumo-data` 모두 1.27.1로 맞춘다. `IC_LIBSUMO=0`이면 기존 SUMO 바이너리와 TraCI 소켓 방식을 사용할 수 있다. 두 군의 백엔드는 반드시 동일하게 둔다.

수업 기본 저장소는 실행 구조 참고 자료로만 사용한다. 수정 대상은 팀 저장소의 pmh이며, PPO·관측·교통·기존 보상은 pmh 기준을 유지한다. 보상은 기존 REWARD_OVERRIDES가 적용된 기본 step 반환값에 추가한다. 모델, CSV, TensorBoard, 실제 적용 설정과 소스 스냅샷을 보존한다.
