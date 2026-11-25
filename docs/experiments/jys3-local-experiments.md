# jys3 실험 자료

[환경·모방학습·초기 PPO 결과](../../results/jys3_ppo_comparison/README.md) · [보상 A/B](../../results/jys3_ppo_comparison/reward_ab/) · [순수 PPO F 실험](../../results/jys3_ppo_comparison/pure_f/) · [시간초과 진단](../../results/jys3_ppo_comparison/pure_f/timeout_distances/) · [종료 보상 비율](../../results/jys3_ppo_comparison/pure_f/terminal_ratio/)

## 시작 가중치별 PPO 전후 비교

학습 시드 11, 고교통, 같은 64개 주행 조건에서 시작 가중치와 60만 스텝 가중치를 평가했다. 각 셀은 학습 전 → 후 건수다.

| 시작 모델 | 완주 | 충돌 | 일반 시간초과 및 정지 종료 |
|---|---:|---:|---:|
| 무작위 | 11 → 18 | 26 → 7 | 27 → 39 |
| BC | 25 → 24 | 39 → 7 | 0 → 33 |
| DAgger | 37 → 2 | 13 → 15 | 14 → 47 |
| BC + DAgger | 46 → 9 | 18 → 4 | 0 → 51 |

BC + DAgger가 처음에 완주했던 46건 중 PPO 후에도 완주한 사례는 9건이었다. 29건은 일반 시간초과, 4건은 잘못된 차로 끝 정지, 4건은 충돌로 바뀌었다. [모델별 학습 전후 주행 기록 및 가중치](../../results/jys3_ppo_comparison/four_600k/)

같은 전문가 검증 관측에서 BC + DAgger의 출력을 비교했다.

| 행동 | PPO 전 | 32,768스텝 | 600,000스텝 |
|---|---:|---:|---:|
| 전문가가 가속한 3,240개 상태에서 모델이 감속한 비율 | 16.0% | 66.6% | 79.1% |
| 전문가가 차로를 바꾼 29개 상태에서 모델이 같은 방향을 명령한 횟수 | 25 | 10 | 10 |

[행동 집계 원자료](../../results/jys3_ppo_comparison/four_600k/expert_probe.json) · [시간제한 재생 결과](../../results/jys3_ppo_comparison/four_600k/horizon_replay.json)

## 종료 보상 4조건 × 3시드

5만 스텝마다 평가한다. 중간 평가는 screen.json, 별도 시드 최종 검증은 final.json에 기록한다. [실행 조건과 체크포인트별 결과](../../results/jys3_ppo_comparison/pure_f/terminal_ratio/long_600k_4/)
이번 업로드는 2026-11-25 10:51 한국시간 기준의 실험 자료다.
