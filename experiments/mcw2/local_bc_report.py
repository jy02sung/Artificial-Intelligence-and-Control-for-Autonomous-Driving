"""Create the local Korean experiment note from paired frozen-model evaluation files."""
from pathlib import Path
from collections import Counter
import statistics, hashlib, json, math
import xml.etree.ElementTree as ET
D = Path(__file__).resolve().parents[2] / 'results/local_native_bc_ppo'
def read(name):
    return json.loads((D / (name + '.json')).read_bytes())
def rows(name, stage):
    d=read(name+'_'+stage)
    assert hashlib.sha256((D/(name+'.pt')).read_bytes()).hexdigest()==d['sha256']
    items=d['rows']; n={'screen':5,'dev':25,'final':100,'safety_holdout':100}[stage]
    assert len(items)==len({r['seed'] for r in items})==16*n
    assert Counter(r['route'] for r in items)=={f'route_{i:02}':n for i in range(1,17)}
    assert set(r['outcome'] for r in items)<={'arrival','collision','timeout'}
    return items
def rate(k,n):
    return f'{k}/{n} ({100*k/n:.2f}%)'
def wilson(k,n):
    z=1.96; p=k/n; u=1+z*z/n; m=(p+z*z/(2*n))/u
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/u
    return f'{100*(m-h):.2f}–{100*(m+h):.2f}%'
def counts(items):
    c=Counter(r['outcome'] for r in items)
    return [c['arrival'],c['collision'],c['timeout']]
def main():
    sels=sorted(D.glob('selection_*.json'),key=lambda s:int(s.stem.split('_')[-1]));assert len(sels)>=3
    variants=[json.loads(s.read_bytes()) for s in sels]
    report=['# SUMO 시범으로 초기화한 BC와 PPO 추가 학습', '',
        '2026-11-23 로컬 검증. 고교통 16개 경로 전체에서 완주 95% 이상, 충돌 5% 미만을 함께 충족하는지 확인했다. 최신 요청의 충돌 5% 이하와 정확히 5%인 경우도 구별한다.', '',
        '## 평가 방법','',
        '- 시간당 유입 설정 15,000대, 시야 120m, 대형차 가림과 거리·속도 노이즈, 0.2초 간격, 평가 제한 600초. 모든 평가에서 자차의 자동 속도·차선 제어와 추가 안전 필터를 껐다. 각 정책은 학습한 가속 평균과 최댓값 차선 범주를 실행했다.',
        '- 학생은 시야 내 다섯 차선의 교통 상황과 경로와 무관한 물리적 연결성, 자차 차선·도로 정보, 주행 경로상의 현재·다음 도로와 400m 이내 다음 분기에서 나갈 도로를 받았다. 목표 차선을 알려주는 입력은 제외했다. 도로 식별자를 사용했으므로 결과의 범위는 이 도로망이다.',
        '- SUMO 기본 차량이 낸 다음 스텝의 속도 변화와 같은 도로 안에서의 실제 차선 이동을 시범 행동으로 기록했다. 도로가 바뀐 스텝은 차선 학습에서 제외했다. 무작위 초기화 후 BC로 두 행동을 학습했고 이후 업데이트에는 학생 자신의 주행과 PPO만 사용했다.',
        '- BC와 PPO 체크포인트 선별에는 같은 80개 시드를 썼다. 각 실험군에서 완주가 가장 많고 동률이면 충돌이 가장 적은 BC와 PPO 후보를 각각 고른 뒤, 독립된 400개 시드(경로당 25개)에서 짝지어 비교했다. 1,600회(경로당 100개) 최종 검증에는 두 개발 기준을 모두 달성한 후보만 진입한다.',
        '- 기존 mcw2의 장면, 충돌·도착·속도·근접·정체 보상을 사용했다. 차선 사전 준비 보상은 껐다. 정체 보상에는 원래 코드의 분기 끝 차선 조건이 남아 있다. 두 번째 실험은 훈련 중에만 옆 차와의 짧은 간격에 추가 감점을 적용했다. 세 번째 실험은 도착 보상을 +5에서 +16, 충돌 보상을 −15에서 −45로 바꾸고 짧은 간격의 차선 변경 시도 감점 계수를 0.8에서 2.0으로 높였다. 또 가치 추정 오차가 BC에서 학습한 공통층을 직접 바꾸지 않도록 했다. 이 변경을 한꺼번에 적용해 개별 효과는 분리할 수 없다.','',
        '## 시범의 차이','']
    d=read('demo_summary')['outcomes']; r=read('recovery_summary')['outcomes'];coverage=read('coverage')
    report+=[f"기본 시범은 {d['arrival']}/768회 완주했다. 추가 시범 768회 중 완주는 {r['arrival']}회, 충돌은 {r['collision']}회였다. 추가분에서는 임의로 속도를 낮추거나 인접 차선으로 이동한 후 SUMO가 제어를 되찾게 했다. 개입 스텝은 버리고 완주한 시범만 BC에 사용했다.",
        '수업 BC 예제는 수집 정책이 출력한 원시 행동을 저장한다. 이 실험은 SUMO 기본 차량을 교사로 골라 관측된 속도·차선 변화에서 행동을 환산했다. 교사의 내부 명령과 실제 움직임 사이에는 차이가 있을 수 있다. [수업 BC 자료](https://github.com/jy02sung/Artificial-Intelligence-and-Control-for-Autonomous-Driving/blob/jys2/project_materials/3_Model_Training_BC.md)']
    for src,title in [('demos','기본 시범'),('recovery_demos','복귀 시범')]:
        s=coverage[src]
        report+=[f"- {title} 중 속도 3m/s 미만·관측 앞 간격 60m 초과 장면은 {s['rows']:,}스텝 중 {s['slow_open']:,}스텝({s['episodes_with_slow_open']}개 주행)이었다. 이 중 {s['positive']:,}스텝에서 교사가 가속했다."]
    report+=['','## 독립 400회 결과','',
        'bc8은 기본 시범만 쓴 BC다. kin10과 kin2는 기본·복귀 시범과 관측된 차간 간격·접근속도를 사용한다. 이름의 숫자는 BC의 차선 학습 배치에서 실제 차선 변경 시범이 차지한 백분율이다. 모델명의 p 뒤에는 PPO 학습 시드와 업데이트 횟수가 온다. 업데이트 1회는 2,048스텝이다.', '',
        '| 모델 | 완주 | 충돌 | 시간초과 | 충돌률의 참고용 95% Wilson 구간 |','|---|---:|---:|---:|---:|']
    model_names=[]; heldout=None
    for sel in variants:
        for name in [sel['parent'],sel['candidate']]:
            items=rows(name,'dev'); seen=[(r['route'],r['seed']) for r in items]
            if heldout is None:heldout=seen
            assert heldout==seen
            if name in model_names:continue
            model_names.append(name); a,c,t=counts(items);n=len(items)
            report.append(f'| [{name}]({name}_dev.json) | {rate(a,n)} | {rate(c,n)} | {rate(t,n)} | {wilson(c,n)} |')
    safe_name='kin2_p46301_12'
    if safe_name not in model_names:
        items=rows(safe_name,'dev');assert [(r['route'],r['seed']) for r in items]==heldout
        model_names.append(safe_name)
        a,c,t=counts(items);n=len(items)
        report.append(f'| [{safe_name} (충돌 우선 확인)]({safe_name}_dev.json) | {rate(a,n)} | {rate(c,n)} | {rate(t,n)} | {wilson(c,n)} |')
    report+=['','충돌률만 낮출 때의 대가도 보기 위해 선별 평가 충돌 1/80회였던 세 번째 실험의 12회 업데이트 모델을 같은 400회에 추가했다. 이 행은 완주 우선 선별 결과와 별도로 읽는다.']
    report+=['','### 동일한 시드에서 완주 여부가 달라진 횟수','']
    for sel in variants:
        p,b=sel['parent'],sel['candidate']
        before,after=rows(p,'dev'),rows(b,'dev')
        trans=Counter((s['outcome'],t['outcome']) for s,t in zip(before,after))
        gained=sum(v for (s,t),v in trans.items() if t=='arrival' and s!='arrival')
        lost=sum(v for (s,t),v in trans.items() if s=='arrival' and t!='arrival')
        tr=read(b+'_train')
        assert tr['parent_sha256']==hashlib.sha256((D/(p+'.pt')).read_bytes()).hexdigest()
        report.append(f"- {p} → {b} ({tr['steps']:,} PPO 스텝, 학습 시드 {tr['seed']}). BC 실패 중 {gained}회가 완주가 되었고 BC 완주 중 {lost}회가 실패로 바뀌었다.")
    report+=['','### 경로별 완주 / 충돌 / 시간초과','','| 경로 | '+' | '.join(model_names)+' |','|---|'+'---:|'*len(model_names)]
    for i in range(1,17):
        rt=f'route_{i:02}'; line=[f'{i:02}']
        for name in model_names:
            items=[r for r in rows(name,'dev') if r['route']==rt]
            line.append(' / '.join(map(str,counts(items))))
        report.append('| '+' | '.join(line)+' |')
    report+=['','## 선별 단계와 학습량','','각 행은 같은 별도 80개 시드이다. 각 실험군에서 선택한 후보의 성능을 인용할 때는 위의 400회 결과를 사용한다.','',
        '| 모델 | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|']
    for path in sorted(D.glob('*_screen.json')):
        nm=path.name.removesuffix('_screen.json'); a,c,t=counts(rows(nm,'screen'));report.append(f'| [{nm}]({path.name}) | {a} | {c} | {t} |')
    report+=['','| BC 초기 모델 | 학습 스텝 표본 수 | 같은 도로 내 차선 이동 표본 | BC 최적화 횟수 |','|---|---:|---:|---:|']
    for fit in sorted(D.glob('*_fit.json')):
        d=json.loads(fit.read_bytes());report.append(f"| [{fit.name.removesuffix('_fit.json')}]({fit.name}) | {d['train_rows']:,} | {d['positive_training_examples']:,} | {d['updates']:,} |")
    for sel in variants:
        b=sel['candidate']; prefix=b.rsplit('_',1)[0]; logs=[json.loads(f.read_bytes()) for f in D.glob(prefix+'_*_train.json')]
        report.append(f"- {prefix} 학습은 최대 {max(t['steps'] for t in logs):,} PPO 스텝까지 수행했다. 모든 저장 시점과 선별 점수는 위 표에서 확인할 수 있다.")
    if (D/'lateral_summary.json').exists():
        extra=read('lateral_summary')['outcomes']
        report+=['',f"추가로 02·05·11·13·14번에서 128회씩 총 640회 인접 차선에서 복귀하는 시범을 수집했다. 교사 차량은 {extra['arrival']}회 완주하고 {extra['collision']}회 충돌했다. kinlat3의 400회 평가 진입 기준은 80회 중 완주 74회 이상·충돌 4회 이하 또는 완주 73회 이상·충돌 2회 이하였다."]
        if (D/'selection_48177.json').exists():
            report+=['이 후보의 PPO는 세 번째 실험의 보상을 사용하고 학습률을 0.000008로 낮췄다.']
    report+=['','## 실패 재현과 해석','',
        '기본 BC의 01번 선별 시드 1187975352에서 차량은 마지막 다섯 스텝 동안 정지했고 앞 간격 관측은 모두 상한인 120m였다. 다섯 번 모두 가속 출력이 음수였고 현재 차선을 유지했다. 이 사례에서 충돌 없는 시간초과가 실제 정지와 일치했다. [재현 기록](bc8_diagnosis.json)']
    trace=read('kin10_screen_crashes')
    side=[r for r in trace if any(c['type']=='side' for c in r['contacts'])]
    near=sum(bool(f['cmd']) and min(f['target_front'],f['target_rear'])<1 for r in side for f in [r['frames'][-1]])
    report+=['',f'복귀 시범을 학습한 kin10의 선별 충돌 {len(trace)}회를 같은 시드에서 재현했다. SUMO가 측면 접촉으로 분류한 {len(side)}회 중 {near}회는 충돌 직전 목표 차선의 앞뒤 중 한 곳의 관측 간격이 1m 미만인데도 그쪽으로 변경을 요청했다. [충돌 직전 행동과 접촉 기록](kin10_screen_crashes.json)']
    for final in sorted(D.glob('*_final.json')):
        name=final.name.removesuffix('_final.json');report+=['',f"독립 최종 1,600회 {name} 결과(완주 / 충돌 / 시간초과)는 {' / '.join(map(str,counts(rows(name,'final'))))}이다."]
    holdout=rows(safe_name,'safety_holdout')
    a,c,t=counts(holdout); scout=rows(safe_name,'screen')
    assert not {rec['seed'] for rec in holdout}&{r['seed'] for r in scout+rows(safe_name,'dev')}
    report[2:2]=[
        '이 실험의 BC→PPO 모델 중 팀의 두 성능 목표를 함께 달성한 모델은 없었다. 독립 개발 400회에서 가장 많이 완주한 PPO는 369회 완주(92.25%)·31회 충돌(7.75%)이었다.',
        f'충돌을 줄인 PPO 모델을 별도의 1,600회에서 다시 확인한 결과는 완주 {rate(a,1600)}·충돌 {rate(c,1600)}·시간초과 {rate(t,1600)}였다. 두 기준을 달성한 발표 모델로 채택할 근거는 부족하다.', '']
    report+=['','## 충돌률만 따로 확인한 새 시드 1,600회','',
        f'400회에서 충돌 17회가 나온 {safe_name}을 동결했다. 새 시드에서 16개 경로를 100회씩 주행한 결과는 완주 {rate(a,1600)}, 충돌 {rate(c,1600)}, 시간초과 {rate(t,1600)}였다. 충돌률의 참고용 95% Wilson 구간은 {wilson(c,1600)}이다. [경로·시드별 종료 기록]({safe_name}_safety_holdout.json)',
        '', '이 평가는 충돌률 자체의 재현을 확인하기 위해 추가했다. 모델이나 체크포인트를 고르는 데 사용한 80회·400회와 시드가 겹치지 않는다.', '',
        '| 경로 (각 100회) | 완주 | 충돌 | 시간초과 |','|---|---:|---:|---:|']
    for i in range(1,17):
        a,c,t=counts([r for r in holdout if r['route']==f'route_{i:02}'])
        report.append(f'| {i:02} | {a} | {c} | {t} |')
    stop=read(safe_name+'_dev_diagnosis')
    scenario=D.parents[1]/'scenarios'/'hv_high15000_entryprep'/'highway.rou.xml'
    paths={r.get('id'):r.get('edges').split() for r in ET.parse(scenario).findall('route')}
    links=ET.parse(scenario.with_name('network.net.xml')).findall('connection')
    report+=['','### 시간초과 재현','',
        '400회 개발 평가에서 시간초과가 발생한 각 경로의 첫 사례를 같은 시드로 다시 주행했다. 표는 종료 직전 한 스텝에서의 센서와 행동이다. 간격 120m는 관측 상한이다. [마지막 다섯 스텝 기록]('+safe_name+'_dev_diagnosis.json)','',
        '| 경로 | 도로·차선 | 속도(m/s) | 앞 간격(m) | 도로 구간 끝까지(m) | 가속 출력 | 차선 명령 | 다음 도로에 연결되는 차선 번호 |','|---|---|---:|---:|---:|---:|---:|---|']
    for case in stop:
        lane,_,v,act,g,left=case['last'][-1]
        edge,idx=lane.rsplit('_',1);path=paths[case['route']]
        nxt=path[path.index(edge)+1]
        targets=sorted({ln.get('fromLane') for ln in links if ln.get('from')==edge and ln.get('to')==nxt})
        assert idx not in targets and all(last[2]==0 and last[3][1]==0 and abs(last[5])<0.1 for last in case['last'])
        report.append(f"| {case['route']} | {lane} | {v:.1f} | {g:.1f} | {left:.1f} | {act[0]:.2f} | {int(act[1])} | {', '.join(targets)} |")
    report+=['','7개 경로에서 재현한 각 1회는 모두 다음 경로로 연결되지 않는 차선 끝에서 멈췄다. 정책은 마지막 다섯 스텝에 차선 유지를 선택했다. 연결 차선은 종료 후 도로망에서 대조했다. 전체 시간초과 93회 중 이 7회를 재현했다.']
    report+=['','모든 시범·훈련·평가 시드는 용도별로 구간을 분리했다. 400회 결과는 실험 방향과 후보를 비교한 개발 자료다. 단일 학습 시드에서 얻은 각 모델과 이 도로망에서의 결과로 해석한다.','',
        '## 로컬 파일','',
        '- [시범 수집·정책 입력·평가](../../experiments/mcw2/local_bc_ppo.py) · [복귀 시범 수집](../../experiments/mcw2/local_bc_recovery.py) · [BC와 PPO 학습](../../experiments/mcw2/local_bc_fit.py) · [후보 선별](../../experiments/mcw2/local_bc_run.py)',
        '- [완주가 가장 많았던 PPO 체크포인트](kin10_p21823_32.pt) · [충돌률을 별도 확인한 PPO 체크포인트](kin2_p46301_12.pt) · [그 모델의 BC 초기 체크포인트](kin2.pt)']
    (D/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':main()
