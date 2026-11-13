# 개인 브랜치에서 실험하는 방법

**각자 최신 main에서 개인 브랜치를 만들고, 테스트할 내용을 직접 정해 실험합니다. 담당 항목을 중앙에서 배정하는 방식이 아닙니다.**

## 1. 최신 main에서 시작하기

처음 받는 사람은 먼저 저장소를 복제합니다.

```bash
git clone https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving
```

이미 저장소가 있다면 해당 폴더에서 시작합니다. 기존 수정사항은 먼저 기존 브랜치에 커밋하거나 임시 보관한 뒤 진행합니다.

```bash
git fetch origin
git switch -c codex/본인이름-실험주제 origin/main
git rev-parse HEAD
```

예: `codex/jys-safe-distance`. 마지막 명령의 커밋 번호를 기록하면 어느 버전의 main에서 시작했는지 알 수 있습니다. 같은 이름의 브랜치가 이미 있다면 다른 이름을 사용합니다.

## 2. 개인 브랜치에서 수정하고 실험하기

- main은 직접 수정하지 않고, 본인 브랜치에서 코드·설정을 변경합니다.
- 실험 주제는 자유롭게 정하되, 변경 전 조건도 평가하고 가능하면 한 번에 한 요소씩 비교합니다.
- 결과에는 **시작 커밋, 바꾼 설정, 학습 스텝·seed, 교통 조건, 완주율·충돌률·기타 실패율**을 간단히 적습니다.
- 설치 방법은 [메인 README](../README.md#설치)를 참고합니다. 학습 모델 파일은 Git에 없으므로 기존 모델을 이어 학습하려면 별도로 전달받아야 합니다.

## 3. GitHub에 올리고 링크 공유하기

```bash
git status
git add 수정한파일경로 결과요약문서경로
git commit -m "실험 내용과 결과 요약"
git push -u origin HEAD
```

`git add`의 경로는 실제 수정한 코드·문서 경로로 바꿉니다. 결과 요약은 `docs/experiments/본인이름-실험주제.md`에 작성하면 됩니다.

**GitHub의 개인 브랜치 또는 결과 문서 링크를 카톡에 공유합니다.** 모델·데이터·TensorBoard 로그 등 큰 파일은 로컬에 보관하고, 필요할 때 별도 공유합니다. 기존 `.gitignore`를 유지합니다.

## 4. main에 반영할 때

실험이 끝났다고 자동으로 main에 합치지 않습니다. 팀에서 결과를 비교한 뒤 반영할 변경만 **개인 브랜치 → main Pull Request(PR)**로 올려 검토·병합합니다. 실패한 실험도 결과를 남기면 됩니다.

> 최신 main 받기 → 개인 브랜치 생성 → 자유롭게 수정·실험 → 코드와 결과 요약 push → 링크 공유 → 필요한 변경만 PR
