"""Report complete matched suites only; never pool repeated checkpoint evaluations."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'control': '기존 조건 추가학습', 'observation': '교차 차량 관측',
          'reward': '교차 위험 보상', 'combined': '교차 관측+보상',
          'future': '다음 분기 안내', 'lateral': '옆 차로 위험 관측'}


def main():
    suites = {}
    for family in ('junction_conflict_trials_20261116', 'lane_followup_20261116', 'preserved_input_trial_20261116'):
        root = ROOT / 'results' / family
        for folder in ('early-evaluation', 'runs'):
            for path in sorted((root / folder).rglob('policy-*-evaluation.json')):
                rows = json.loads(path.read_text(encoding='utf8'))
                assert len(rows) == 64, path
                assert len({(r['traffic'], r['route'], r['seed']) for r in rows}) == 64
                assert all(r['route_overrides'] == 0 for r in rows)
                key = (family, rows[0]['condition'], int(rows[0]['checkpoint'].split('-')[1].split('.')[0]))
                assert all(r['condition'] == key[1] and r['checkpoint'] == rows[0]['checkpoint'] for r in rows)
                signature = sorted((r['traffic'], r['route'], r['seed'], r['outcome'], r.get('seconds')) for r in rows)
                if key in suites:
                    assert suites[key][2] == signature, ('replay differs', key)
                suites[key] = (path, rows, signature)
    lines = ['# 목표 근접 기준 모델의 후속 비교', '',
             '동일 출발 모델 round-02/route/policy-200000.pt, 학습 seed11. 강제 보정 없음. 추가 학습 스텝 기준.',
             '평가 seed79101~79103: 교통48회, 무교통16회. 완료된 전체 평가만 표시하며 중복 재평가는 합산하지 않는다.', '',
             '|조건|추가 스텝|무교통 완주|교통 완주|교통 충돌|교통 기타|완주+충돌|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for (family, condition, step), (path, rows, _) in sorted(suites.items()):
        traffic = Counter(r['outcome'] for r in rows if r['traffic'] == 'reference')
        none = [r for r in rows if r['traffic'] == 'none']
        assert len(none) == 16 and sum(traffic.values()) == 48
        a, c = traffic['arrival'], traffic['collision']
        label = '다음 분기 안내·행동 보존 초기화' if family.startswith('preserved_input') else LABELS[condition]
        lines.append(f'|{label}|{step:,}|{sum(r["outcome"] == "arrival" for r in none)}/16|{a}/48 ({a/48:.2%})|{c}/48 ({c/48:.2%})|{48-a-c}/48|{(a+c)/48:.2%}|')
    lines += ['', '## 판단 기준', '',
              '- 기준 모델의 별도320회 검증은 완주287·충돌29·시간초과4였다. 이 표의48회와 표본을 섞지 않는다.',
              '- 기존 모델을 보존한다. 더 오래 학습한 모델이나 새 기능을 넣은 모델을 자동으로 우수하다고 취급하지 않는다.',
              '- 개발 평가에서 개선된 후보만 새로운 seed로 검증하고, 충돌률5% 미만 및 기타 실패0 여부를 별도로 확인한다.', '',
              '## 평가 파일', '']
    lines += [f'- `{path.relative_to(ROOT).as_posix()}`' for path, _, _ in suites.values()]
    output = ROOT / 'docs/experiments/ic-live-comparison-20261116.md'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(f'{len(suites)} completed suites -> {output}')


if __name__ == '__main__':
    main()
