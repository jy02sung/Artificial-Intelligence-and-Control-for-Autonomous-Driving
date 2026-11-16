"""Summarize completed early evaluations without pooling repeated evaluations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'results/junction_conflict_trials_20261116/early-evaluation'
lines=['# 교차 위험 실험: 완료된 중간 평가','', '동일 평가seed79101~79103. 모델당 교통48회, 무교통16회. 반복 실행 결과를 합산하지 않는다. 학습은 진행 중이다.','', '|조건|추가 스텝|무교통 완주|교통 완주|교통 충돌|교통 기타|','|---|---:|---:|---:|---:|---:|']
seen=set()
for p in sorted(base.rglob('*-evaluation.json')):
 rows=json.loads(p.read_text(encoding='utf8'))
 assert len(rows)==64
 c=rows[0]['condition'];cp=rows[0]['checkpoint'];key=(c,cp)
 if key in seen:continue
 seen.add(key)
 none=[r for r in rows if r['traffic']=='none'];traffic=[r for r in rows if r['traffic']=='reference']
 a=sum(r['outcome']=='arrival' for r in traffic);b=sum(r['outcome']=='collision' for r in traffic)
 lines.append(f"|{c}|{cp}|{sum(r['outcome']=='arrival' for r in none)}/16|{a}/48|{b}/48|{48-a-b}/48|")
(ROOT/'docs/experiments/ic-junction-early-results-20261116.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
