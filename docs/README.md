# 팀 프로젝트 문서

현재 상태: 맵 수정과 기본 AV 실패 원인 진단 완료. **순수 PPO 내비게이션 관측 재학습은 설계 단계**입니다. 규칙 보정48/48은 PPO 학습 성능이 아닙니다.

| 문서 | 용도 |
|---|---|
| [현재 실험 설계](ic-navigation-learning-plan.md) | 강제 보정 제외, 수업 기준에 맞춘 관측 비교·재학습 계획 |
| [최신 원인 분석](ic-toegyewon-lane-cause-results.md) | 기본 PPO32/48, 실패 원인과 진단 개입의 구분 |
| [맵 수정 기록](ic-map-geometry-review.md) | 수정·정정 순서와 현재 분기 구조 검증 |
| [수업 평가 기준](../project_materials/6_Project_Evaluation.md) | 원본 수업 요구사항 |
| [데이터 보관 정책](../data/README.md) | GitHub와 로컬에 보관할 파일 구분 |
| [과거 문서 보관함](archive/README.md) | 이전 설계·실패 실험·맵 기록 |

원본 수업 자료는 project_materials에 유지합니다. 소형 실험 설정과 설계 JSON은 Git에서 관리하고, 개별 사례·추적·학습 데이터는 로컬에 보존합니다.
