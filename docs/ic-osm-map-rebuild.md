# 첨부 OSM 기반 퇴계원·사노 IC 맵 재구성

2026-11-07. 사용자가 제공한 map.osm을 기준으로 다시 구성했다. 기존 맵은 교육용 단순화 모델이므로 실제 도로와 동일하다고 볼 수 없다. 학습·데이터 수집은 중단 상태다.

## 무엇을 수정했는가

| 항목 | 기존 수작업 맵 | 새 OSM 기반 맵 |
|---|---|---|
| 도로 형상 | 일부 수작업 곡선, 위치 이동·길이 단순화 | 원본 위경도 중심선을 투영하여 변환 |
| 남향 본선 진입 | 6차로 | OSM 86198728의 5차로 |
| 북향 본선 진입 | 5차로 | OSM 52873356의 4차로 |
| 북향 진출 램프 | 1차로에서 3차로로 확장 | OSM 37927512의 3차로 |
| 외곽 램프 진입 | 남향 고속도로에서 직접 분기 | 서측 도로 노드 1808903166에서 진입 |
| 외곽 램프 종점 | 독립 종료점 | 노드 1000493465에서 남향 본선에 합류 |
| 주변 도로 | 선택적으로 생략 | 첨부 파일 경계에 걸치는 승용차·버스 도로 포함 |
| 표시 | 수작업 상판·차선 장식 | SUMO 기본 차선 표시; 실제 노면 표선 판독 결과는 아님 |

## 검증 결과와 한계

- 822개 외부 도로 구간, 1,201개 차로를 생성했다. 구간 수는 물리적인 도로 개수가 아니다.
- 차로 수가 명시된 일방통행 115개 구간이 OSM 숫자와 일치한다.
- 대표 6개 경로의 승용차 연결을 정적으로 확인했다. 이는 실제 주행·교통용량 검증이 아니다.
- 원본 중심선 좌표 대조에서 8개 구간의 끝점이 약 1.59~3.20m 달랐다. netconvert의 기본 geometry.avoid-overlap=true가 교차부 끝점을 조정할 수 있다. 예외 좌표를 validation.json에 그대로 기록했으며, 전체 정밀 일치 검증은 review_required이다.
- 671개 구간은 차로 태그가 없어 SUMO 도로 유형 기본값을 사용했다. 대부분 주변 도로를 포함하며, 이를 실제 차로 수로 확정해서는 안 된다.
- 일부 교차부 형상 확장·자동 신호 생성 경고가 남는다. conversion-warnings.txt를 보존했다. 특히 본선 합류부 1000493465, 12994833390의 생성 형상은 추가 정밀 검토 대상이다.
- 원본에 구성원이 누락된 회전 제한 관계 5개는 적용 불가로 기록했다. 완전한 제한 관계만 유지한다.
- OSM layer는 실측 높이가 아니다. 임의 높이를 부여하지 않았고, 공유 노드가 없는 입체교차를 평면 교차로 연결하지 않았다. 3D 높이·실선/점선·신호 주기는 실측 재현하지 않았다.

## 파일과 실행

- [원본 OSM](../maps/toegyewon_osm/source.osm), [SUMO 맵](../maps/toegyewon_osm/network.net.xml), [열기 설정](../maps/toegyewon_osm/review.sumocfg)
- [변환 감사 기록](../maps/toegyewon_osm/import-audit.json), [정적 검증](../maps/toegyewon_osm/validation.json), [변환 경고](../maps/toegyewon_osm/conversion-warnings.txt)

```bash
python -m utils.rebuild_osm_map
python -m utils.validate_osm_map
python view_road.py --nogui
python view_road.py
```

기존 맵은 `python view_road.py --legacy`로 재현한다. 기존 env/road_config.py와 학습 환경은 과거 실험 재현용으로 보존했다. 새 맵은 도로 ID가 달라 기존 22개 경로·관측·합류 판정·교통 설정을 그대로 사용할 수 없다. 실제 연결과 차선 형상을 확인하고 경로를 이식한 후에만 별도 검증을 재개한다. 이번 변경으로 PPO 학습이나 데이터 수집을 시작하지 않았다.

## 출처

사용자 제공 OpenStreetMap 추출물. © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright). 원본 SHA-256: d49c7f2c84e3ddf3adb66fd34982e0d983cf386bf518d4edff70ec6e1f3a3ea6. 원본에 없는 현실 정보는 별도 지도·현장 검증이 필요하다.
