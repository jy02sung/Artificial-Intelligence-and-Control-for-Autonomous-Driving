# 시간초과 경계의 GAE 누출 진단과 단독 비교

기존 collect_rollout은 시간초과 보상에 gamma*V(final_obs)를 더하지만 buffer.done에는 terminated만 저장한다. 이후 reset한 다음 에피소드의 가치와 advantage가 이전 에피소드에 연결된다. 두 에피소드의 작은 재현에서 기대 return7.93이14.761로 계산됐다. 재현 원본은 로컬 results/ppo_timeout_boundary_audit_20261116/reproduction.json.

현재 실행 코드와 기준 모델을 보존한다. 별도 mixin에서 동일 rollout을 사용하되 buffer의 경계만 terminated or truncated로 바꾼다. 보상·관측·초기 가중치·학습률·seed·평가 조건은 기존 control 추가학습과 동일하다. 최종 관측 가치로 bootstrap한 후 다음 에피소드 연결은 끊는다. terminal과 rollout 중간 계속 진행은 기존과 같아야 한다.

시간초과가 발생한 학습에서의 계산 오류이며 모든 충돌이나 퇴화의 원인이라고 단정하지 않는다. 특히 시간초과가 없는 rollout은 이 수정으로 달라지지 않는다. 학습은4500스텝 제한, 평가는54000스텝 제한인 기존 조건을 유지하여 수정 효과만 비교한다.

검증: 에피소드 중간 timeout, rollout 마지막 timeout, 실제 terminal 경계 테스트. smoke32스텝+4평가 후 기준 모델에서20만 추가학습,5/10/15/20만 중간 평가. 개발 평가 교통48회와 무교통16경로, 후보는 새 seed 독립검증. 강제 보정 없음.

## 실행 판단 갱신

경계 테스트3개와 smoke32스텝·4평가는 통과했다. 하지만 실제17개 조건의 TensorBoard episode/length를 검사한 시점에4500스텝 도달 에피소드는 모두0이었다. control의691에피소드 중 최장1087스텝이다. 환경에서 truncated는max_steps 도달 시에만 설정된다. 따라서 현재 기록된 퇴화를 이 버그 탓으로 설명하지 않으며,20만 학습은 시작하지 않는다. 수정은 별도 mixin에 보존하고 공용 PPO 및 실행 중 조건은 변경하지 않았다. 향후 timeout이 있는 학습에는 수정된 수집기를 사용해야 한다.
