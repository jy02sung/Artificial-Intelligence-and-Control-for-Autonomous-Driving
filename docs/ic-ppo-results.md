# 퇴계원IC PPO 실험 결과

평가를 마친 조건만 표시한다. 1만 스텝은 기능 점검이며, 본 실험은 조건별 20만 스텝이다.

|조건|seed|학습 스텝|평가 에피소드|완주|충돌|시간 초과|기준 대비 완주/충돌|학습 시간(분)|
|---|---:|---:|---:|---:|---:|---:|---|---:|
|BLOCK-11-10000|11|10,000|22|13.6%|72.7%|13.6%|+0.0%p / +50.0%p|0.7|
|BLOCK-11-200000|11|200,000|110|30.9%|28.2%|40.9%|-19.1%p / -3.6%p|12.1|
|GAP-11-10000|11|10,000|22|0.0%|18.2%|81.8%|-13.6%p / -4.5%p|0.7|
|PREP-11-10000|11|10,000|22|9.1%|81.8%|9.1%|-4.5%p / +59.1%p|1.0|
|R0-11-10000|11|10,000|22|13.6%|22.7%|63.6%|+0.0%p / +0.0%p|0.9|
|R0-11-200000|11|200,000|110|50.0%|31.8%|18.2%|+0.0%p / +0.0%p|11.9|
|RAMP-11-10000|11|10,000|22|18.2%|45.5%|36.4%|+4.5%p / +22.7%p|0.9|
|ZIP-11-10000|11|10,000|22|13.6%|22.7%|63.6%|+0.0%p / +0.0%p|0.7|

지퍼 준수와 준비 성공률은 해당 사건만 분모로 사용한다. 해당 사건이 없으면 ‘관측 없음’으로 표시한다.

|조건|100m 준비|지퍼 준수|램프 과속 시간 비율|최대 초과(km/h)|완주 평균(초)|
|---|---:|---:|---:|---:|---:|
|BLOCK-11-10000|7/7 (100.0%)|관측 없음|63/84 (75.0%)|14.3|35.4|
|BLOCK-11-200000|35/39 (89.7%)|2/6 (33.3%)|1093/9678 (11.3%)|53.5|73.1|
|GAP-11-10000|6/6 (100.0%)|관측 없음|0/13413 (0.0%)|0.0|완주 없음|
|PREP-11-10000|7/7 (100.0%)|관측 없음|3/3 (100.0%)|20.8|31.4|
|R0-11-10000|6/6 (100.0%)|관측 없음|0/21655 (0.0%)|0.0|92.5|
|R0-11-200000|40/40 (100.0%)|0/4 (0.0%)|1270/7395 (17.2%)|60.0|114.0|
|RAMP-11-10000|9/11 (81.8%)|0/1 (0.0%)|57/1118 (5.1%)|56.8|51.0|
|ZIP-11-10000|6/6 (100.0%)|관측 없음|0/21655 (0.0%)|0.0|92.5|

<details><summary>BLOCK-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|0.0%|100.0%|0.0%|
|east_to_north|1|0.0%|100.0%|0.0%|
|east_to_south|1|0.0%|100.0%|0.0%|
|north_through|1|100.0%|0.0%|0.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|100.0%|0.0%|
|sano_north_through|1|0.0%|100.0%|0.0%|
|sano_north_to_west|1|0.0%|100.0%|0.0%|
|sano_south_through|1|100.0%|0.0%|0.0%|
|sano_south_to_east|1|0.0%|100.0%|0.0%|
|sano_south_to_west|1|0.0%|0.0%|100.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|100.0%|0.0%|0.0%|
|south_to_east|1|0.0%|100.0%|0.0%|
|south_to_local|1|0.0%|100.0%|0.0%|
|south_to_west|1|0.0%|100.0%|0.0%|
|west_through|1|0.0%|100.0%|0.0%|
|west_to_local|1|0.0%|100.0%|0.0%|

</details>

<details><summary>BLOCK-11-200000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|5|60.0%|40.0%|0.0%|
|east_to_north|5|100.0%|0.0%|0.0%|
|east_to_south|5|100.0%|0.0%|0.0%|
|north_through|5|20.0%|40.0%|40.0%|
|north_to_east|5|0.0%|0.0%|100.0%|
|north_to_local|5|20.0%|40.0%|40.0%|
|north_to_west|5|0.0%|0.0%|100.0%|
|outer_to_south|5|100.0%|0.0%|0.0%|
|sano_east_to_south|5|0.0%|100.0%|0.0%|
|sano_north_through|5|0.0%|100.0%|0.0%|
|sano_north_to_west|5|0.0%|20.0%|80.0%|
|sano_south_through|5|100.0%|0.0%|0.0%|
|sano_south_to_east|5|0.0%|0.0%|100.0%|
|sano_south_to_west|5|20.0%|0.0%|80.0%|
|sano_west_to_north|5|0.0%|60.0%|40.0%|
|sano_west_to_south|5|0.0%|60.0%|40.0%|
|south_through|5|80.0%|20.0%|0.0%|
|south_to_east|5|0.0%|0.0%|100.0%|
|south_to_local|5|0.0%|20.0%|80.0%|
|south_to_west|5|0.0%|0.0%|100.0%|
|west_through|5|80.0%|20.0%|0.0%|
|west_to_local|5|0.0%|100.0%|0.0%|

</details>

<details><summary>GAP-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|0.0%|0.0%|100.0%|
|east_to_north|1|0.0%|0.0%|100.0%|
|east_to_south|1|0.0%|0.0%|100.0%|
|north_through|1|0.0%|0.0%|100.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|0.0%|100.0%|
|sano_north_through|1|0.0%|0.0%|100.0%|
|sano_north_to_west|1|0.0%|0.0%|100.0%|
|sano_south_through|1|0.0%|0.0%|100.0%|
|sano_south_to_east|1|0.0%|0.0%|100.0%|
|sano_south_to_west|1|0.0%|0.0%|100.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|0.0%|0.0%|100.0%|
|south_to_east|1|0.0%|0.0%|100.0%|
|south_to_local|1|0.0%|0.0%|100.0%|
|south_to_west|1|0.0%|0.0%|100.0%|
|west_through|1|0.0%|0.0%|100.0%|
|west_to_local|1|0.0%|0.0%|100.0%|

</details>

<details><summary>PREP-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|0.0%|100.0%|0.0%|
|east_to_north|1|0.0%|100.0%|0.0%|
|east_to_south|1|0.0%|100.0%|0.0%|
|north_through|1|100.0%|0.0%|0.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|100.0%|0.0%|
|sano_north_through|1|0.0%|100.0%|0.0%|
|sano_north_to_west|1|0.0%|100.0%|0.0%|
|sano_south_through|1|0.0%|100.0%|0.0%|
|sano_south_to_east|1|0.0%|100.0%|0.0%|
|sano_south_to_west|1|0.0%|100.0%|0.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|100.0%|0.0%|0.0%|
|south_to_east|1|0.0%|100.0%|0.0%|
|south_to_local|1|0.0%|100.0%|0.0%|
|south_to_west|1|0.0%|100.0%|0.0%|
|west_through|1|0.0%|100.0%|0.0%|
|west_to_local|1|0.0%|100.0%|0.0%|

</details>

<details><summary>R0-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|100.0%|0.0%|0.0%|
|east_to_north|1|0.0%|0.0%|100.0%|
|east_to_south|1|0.0%|0.0%|100.0%|
|north_through|1|0.0%|100.0%|0.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|0.0%|100.0%|
|sano_north_through|1|100.0%|0.0%|0.0%|
|sano_north_to_west|1|0.0%|0.0%|100.0%|
|sano_south_through|1|0.0%|0.0%|100.0%|
|sano_south_to_east|1|0.0%|0.0%|100.0%|
|sano_south_to_west|1|0.0%|0.0%|100.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|0.0%|0.0%|100.0%|
|south_to_east|1|0.0%|0.0%|100.0%|
|south_to_local|1|0.0%|0.0%|100.0%|
|south_to_west|1|0.0%|0.0%|100.0%|
|west_through|1|100.0%|0.0%|0.0%|
|west_to_local|1|0.0%|0.0%|100.0%|

</details>

<details><summary>R0-11-200000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|5|40.0%|60.0%|0.0%|
|east_to_north|5|100.0%|0.0%|0.0%|
|east_to_south|5|100.0%|0.0%|0.0%|
|north_through|5|60.0%|40.0%|0.0%|
|north_to_east|5|80.0%|20.0%|0.0%|
|north_to_local|5|80.0%|20.0%|0.0%|
|north_to_west|5|40.0%|20.0%|40.0%|
|outer_to_south|5|100.0%|0.0%|0.0%|
|sano_east_to_south|5|80.0%|20.0%|0.0%|
|sano_north_through|5|40.0%|60.0%|0.0%|
|sano_north_to_west|5|40.0%|60.0%|0.0%|
|sano_south_through|5|80.0%|20.0%|0.0%|
|sano_south_to_east|5|100.0%|0.0%|0.0%|
|sano_south_to_west|5|60.0%|40.0%|0.0%|
|sano_west_to_north|5|0.0%|40.0%|60.0%|
|sano_west_to_south|5|0.0%|40.0%|60.0%|
|south_through|5|0.0%|40.0%|60.0%|
|south_to_east|5|0.0%|60.0%|40.0%|
|south_to_local|5|0.0%|60.0%|40.0%|
|south_to_west|5|40.0%|60.0%|0.0%|
|west_through|5|60.0%|40.0%|0.0%|
|west_to_local|5|0.0%|0.0%|100.0%|

</details>

<details><summary>RAMP-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|0.0%|100.0%|0.0%|
|east_to_north|1|0.0%|0.0%|100.0%|
|east_to_south|1|0.0%|0.0%|100.0%|
|north_through|1|100.0%|0.0%|0.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|0.0%|100.0%|
|sano_north_through|1|0.0%|100.0%|0.0%|
|sano_north_to_west|1|0.0%|0.0%|100.0%|
|sano_south_through|1|100.0%|0.0%|0.0%|
|sano_south_to_east|1|0.0%|0.0%|100.0%|
|sano_south_to_west|1|0.0%|100.0%|0.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|100.0%|0.0%|0.0%|
|south_to_east|1|0.0%|100.0%|0.0%|
|south_to_local|1|0.0%|100.0%|0.0%|
|south_to_west|1|0.0%|100.0%|0.0%|
|west_through|1|100.0%|0.0%|0.0%|
|west_to_local|1|0.0%|0.0%|100.0%|

</details>

<details><summary>ZIP-11-10000 경로별 평가</summary>

|경로|횟수|완주|충돌|시간 초과|
|---|---:|---:|---:|---:|
|east_through|1|100.0%|0.0%|0.0%|
|east_to_north|1|0.0%|0.0%|100.0%|
|east_to_south|1|0.0%|0.0%|100.0%|
|north_through|1|0.0%|100.0%|0.0%|
|north_to_east|1|0.0%|100.0%|0.0%|
|north_to_local|1|0.0%|100.0%|0.0%|
|north_to_west|1|0.0%|100.0%|0.0%|
|outer_to_south|1|0.0%|100.0%|0.0%|
|sano_east_to_south|1|0.0%|0.0%|100.0%|
|sano_north_through|1|100.0%|0.0%|0.0%|
|sano_north_to_west|1|0.0%|0.0%|100.0%|
|sano_south_through|1|0.0%|0.0%|100.0%|
|sano_south_to_east|1|0.0%|0.0%|100.0%|
|sano_south_to_west|1|0.0%|0.0%|100.0%|
|sano_west_to_north|1|0.0%|0.0%|100.0%|
|sano_west_to_south|1|0.0%|0.0%|100.0%|
|south_through|1|0.0%|0.0%|100.0%|
|south_to_east|1|0.0%|0.0%|100.0%|
|south_to_local|1|0.0%|0.0%|100.0%|
|south_to_west|1|0.0%|0.0%|100.0%|
|west_through|1|100.0%|0.0%|0.0%|
|west_to_local|1|0.0%|0.0%|100.0%|

</details>
