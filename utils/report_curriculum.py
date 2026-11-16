"""Report completed and partial curriculum benchmarks without conflating them."""
import collections,json
from pathlib import Path
from utils.curriculum_data import OUT,ROOT

def report():
    lines=['# Curriculum·행동 유지 비교 실험', '', '자동 집계: results/curriculum_keep_20261116. 부분 평가는 최종 완주율로 해석하지 않는다. 강제 개입 없이 mean/argmax 정책을 평가한다.', '', '## 평가', '', '| 모델 | 교통 | 평가 진행 | 완주 | 충돌 | 시간초과 | 경로 실패 | 완주+충돌 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    folders=sorted((OUT/'evaluations').glob('*'))
    for folder in folders:
        if folder.name.endswith('-smoke') or not (folder/'partial.json').exists():continue
        rows=json.loads((folder/'partial.json').read_text(encoding='utf8'))
        for mode,total in [('none',16),('reference',48)]:
            subset=[r for r in rows if r['traffic']==mode];counts=collections.Counter(r['outcome'] for r in subset);n=len(subset)
            values=[str(counts[k]) for k in ['arrival','collision','timeout','wrong_exit']]
            ratio=f"{100*(counts['arrival']+counts['collision'])/n:.2f}%" if n==total else '미완료'
            label='기준 모델' if folder.name=='A-seed11-0' else folder.name
            lines.append('| '+' | '.join([label,mode,f'{n}/{total}',*values,ratio])+' |')
    lines+=['', '## 학습', '', '| 조건 | 추가 환경스텝 | 전체/위험 학습스텝 | 위험 관측 표본 | 감속 명령/실제 감속 표본 | 최종 rollout KL |', '|---|---:|---|---:|---|---:|']
    for name in ['A-seed11','B-seed11','C-seed11']:
        path=OUT/'runs'/name/'training.jsonl'
        if not path.exists():continue
        row=json.loads(path.read_text(encoding='utf8').splitlines()[-1]);m=row['mixture'];e=row['exposure']
        lines.append(f"| {name} | {row['steps']} | {m.get('full',0)}/{m.get('risk',0)} | {e.get('risk_states',0)} | {e.get('risk_brake_attempts',0)}/{e.get('risk_actual_decelerations',0)} | {row['metrics']['rollout_kl_final']:.5f} |")
    lines+=['', '학습 중 위험 계측은 5환경스텝마다의 표본 수다. 감속 비율은 서로 다른 정책이 방문한 상태가 달라 직접적인 성능 우열 지표가 아니다. 학습 스텝은 기준 모델 이후 추가량이다.', '', 'A: curriculum+기존 행동 유지+actor LR 감소. B: A+선택적 감속·양보 모방. 두 조건의 관측·보상은 동일하다. 안전 차로변경 모방 자료는 아직 확보되지 않았다.', '', '기존 확인 기준: 무교통15/16, 교통45완주·3충돌·0기타/48. 이번 기준 재평가가 이를 재현하는지 먼저 확인한다. 최종 목표는 새 독립320회에서 충돌5%미만·기타실패0이며, 현재 개발 평가로 목표 달성을 선언하지 않는다.', '']
    dest=ROOT/'docs/experiments/ic-curriculum-keep-results-20261116.md'
    dest.write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines))

if __name__=='__main__':report()
