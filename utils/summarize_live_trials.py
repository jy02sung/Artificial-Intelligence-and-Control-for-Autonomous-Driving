"""Report complete matched suites and separately verified screened partial counts."""
import json
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'control': '기존 조건 추가학습', 'observation': '교차 차량 관측',
          'reward': '교차 위험 보상', 'combined': '교차 관측+보상',
          'future': '다음 분기 안내', 'lateral': '옆 차로 위험 관측'}


def condition_label(family, condition):
    if family.startswith('junction_lookahead'):
        label = {'near50': '합류 검색50m·행동 보존 초기화', 'ahead150': '합류 검색150m·물리 감지50m'}[condition]
    elif family.startswith('safe_route'):
        label = {'safe_reward': '안전한 변경에 경로 보상', 'lateral_zero': '옆 차로 위험·행동 보존 초기화', 'combined': '안전한 경로 보상+옆 차로 위험'}[condition]
    elif family.startswith('conservative_update'):
        label = '사전 위험·안전 보상 조합·학습률1/10'
    elif family.startswith('safety_strength'):
        label = {'collision20': '충돌 벌점 강화', 'lane4': '위험 차선변경 벌점 강화', 'anticipatory': '사전 위험 관측+안전 벌점 강화'}[condition]
    else:
        label = '다음 분기 안내·행동 보존 초기화' if family.startswith('preserved_input') else LABELS[condition]
    return label


def read_screened(root, note):
    meta = json.loads(note.read_text(encoding='utf8'))
    partial = note.with_name(meta['checkpoint'].replace('.pt', '-partial.json'))
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == meta['partial_sha'], note
    rows = json.loads(partial.read_text(encoding='utf8'))
    assert rows and len(rows) == len({(r['traffic'], r['route'], r['seed']) for r in rows})
    condition = rows[0]['condition']
    assert all(r['condition'] == condition and r['checkpoint'] == meta['checkpoint'] and r['route_overrides'] == 0 for r in rows)
    checkpoint = root / 'runs' / condition / meta['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == meta['checkpoint_sha'], note
    expected = {(f'route_{i:02d}', mode, seed) for i in range(1,17) for mode in ('none','reference') for seed in ((79101,) if mode=='none' else (79101,79102,79103))}
    assert {(r['route'],r['traffic'],r['seed']) for r in rows} <= expected
    counts = Counter(r['outcome'] for r in rows if r['traffic']=='reference')
    assert dict(counts) == meta['counts'] and sum(counts.values()) == meta['observed_traffic']
    assert meta['planned_traffic'] == 48
    assert counts['collision'] >= 3 or counts['timeout']+counts['wrong_exit'] > 0
    return dict(condition=condition,step=int(Path(meta['checkpoint']).stem.split('-')[1]),observed=sum(counts.values()),counts=counts,path=note)


def main():
    suites = {}
    revisions = {}
    screened = {}
    for family in ('junction_conflict_trials_20261116', 'lane_followup_20261116', 'preserved_input_trial_20261116', 'safe_route_reward_trials_20261116', 'junction_lookahead_trials_20261116', 'safety_strength_trials_20261116', 'conservative_update_trial_20261116'):
        root = ROOT / 'results' / family
        for note in root.rglob('policy-*-screen-rejected.json'):
            item = read_screened(root, note)
            screened[(family,item['condition'],item['step'])] = item
        proof = root / 'verified-source-revision.json'
        if proof.exists():
            revisions[family] = json.loads(proof.read_text(encoding='utf8'))['verified_commit']
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
             '평가 seed79101~79103: 교통48회, 무교통16회. 첫 표는 완료된 전체 평가만 표시하며 중복 재평가는 합산하지 않는다. 부분 평가는 아래 별도 표에 건수만 표시한다.', '',
             '|조건|추가 스텝|무교통 완주|교통 완주|교통 충돌|교통 기타|완주+충돌|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for (family, condition, step), (path, rows, _) in sorted(suites.items()):
        traffic = Counter(r['outcome'] for r in rows if r['traffic'] == 'reference')
        none = [r for r in rows if r['traffic'] == 'none']
        assert len(none) == 16 and sum(traffic.values()) == 48
        a, c = traffic['arrival'], traffic['collision']
        label = condition_label(family, condition)
        lines.append(f'|{label}|{step:,}|{sum(r["outcome"] == "arrival" for r in none)}/16|{a}/48 ({a/48:.2%})|{c}/48 ({c/48:.2%})|{48-a-c}/48|{(a+c)/48:.2%}|')
    lines += ['', '## 목표 미달로 보류한 부분 평가', '',
              '아래는 완료된 전체 평가가 아니다. 관측된 사례 수만 표시하며 전체 완주율·충돌률로 환산하지 않는다. 원시 결과와 모델 해시를 확인했다.', '',
              '|조건|추가 스텝|교통 관측 사례|완주 건수|충돌 건수|기타 건수|',
              '|---|---:|---:|---:|---:|---:|']
    for (family,condition,step), item in sorted(screened.items()):
        if (family,condition,step) in suites:
            continue
        counts=item['counts'];n=item['observed']
        lines.append(f"|{condition_label(family,condition)}|{step:,}|{n}/48|{counts['arrival']}|{counts['collision']}|{n-counts['arrival']-counts['collision']}|")
    lines += ['', '## 판단 기준', '',
              '- 기준 모델의 별도320회 검증은 완주287·충돌29·시간초과4였다. 이 표의48회와 표본을 섞지 않는다.',
              '- 기존 모델을 보존한다. 더 오래 학습한 모델이나 새 기능을 넣은 모델을 자동으로 우수하다고 취급하지 않는다.',
              '- 개발 평가에서 개선된 후보만 새로운 seed로 검증하고, 충돌률5% 미만 및 기타 실패0 여부를 별도로 확인한다.', '',
              '## 평가 파일', '']
    lines += [f'- `{path.relative_to(ROOT).as_posix()}`' for path, _, _ in suites.values()]
    lines += [f'- `{item["path"].relative_to(ROOT).as_posix()}` (부분 평가)' for item in screened.values()]
    lines += ['', '## 검증된 소스 버전', '',
              '실험 파일의 원본 해시는 manifest와 대조하고, 줄바꿈을 정규화한 소스 내용은 아래 커밋과 일치함을 확인했다. 상세 증거는 각 결과 폴더의 verified-source-revision.json에 보관한다.', '']
    lines += [f'- `{family}`: `{commit}`' for family, commit in revisions.items()]
    output = ROOT / 'docs/experiments/ic-live-comparison-20261116.md'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(f'{len(suites)} completed suites -> {output}')


if __name__ == '__main__':
    main()
