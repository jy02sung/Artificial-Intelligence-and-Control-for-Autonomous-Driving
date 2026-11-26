"""R1-R5 without targeted routes or any state oversampling; match the 144-drive budget."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from utils import jys3_final_relu_dagger as ex

ROOT=ex.ROOT.parent/"jys3_uniform_dagger"
read,save=ex.p.read,ex.p.save


def paired(rows, reference):
    def ids(items):
        return [(r["seed"],r["route"],r["lane"],r["initial_sha256"]) for r in items]
    assert ids(rows)==ids(reference)
    return dict(Counter(a["outcome"]+" -> "+b["outcome"] for a,b in zip(reference,rows)))


def report(result):
    labels={"hold64":"64회", "final256":"91M 시드 256회","novel256":"104.32M 시드 256회"}
    text=["# 같은 BC에서 시작한 기본 DAgger와 보강형 DAgger",
          "", "수업의 데이터 누적 절차만으로 보강형 R5의 주행 성능을 재현할 수 있는지 비교했다.",
          "", "## 수업 근거와 비교 조건", "",
          "BMIL 강의자료 03_imitation_learning.pdf 4쪽은 BC와 DAgger를 구분한다. 18쪽은 전문가 시범과 학생 주행의 관측 분포가 달라지는 이유를 설명하고, 20쪽은 초기 BC, 학생 주행, 전문가 라벨링, 데이터 누적, 재학습을 순서대로 제시한다. 해당 슬라이드가 인용한 [DAgger 원 논문](https://proceedings.mlr.press/v15/ross11a.html).",
          "", "| 항목 | 기본 DAgger | 기존 보강형 |", "|---|---|---|",
          "| 라운드별 주행 | 16경로 × 9회 | 16경로 × 8회 및 취약 경로 4개 × 4회 |",
          "| 학습 배치 | 누적 상태에서 균등하게 최대 256개 | 기본 256개와 차로변경·실패 직전·현재 오답 상태를 최대 32개씩 추가 |",
          "| 지도 손실 | 기본 표본 전체에 가감속 MSE와 차로 분류 손실 | 분류는 전체 배치, 가감속은 기본 표본과 실패 직전 표본 |",
          "",f"두 조건 모두 같은 BC 체크포인트(SHA-256 `{result['start_sha256']}`), 전문가, BC 시범 160회, 검증 시범 32회, 수정 분기·고가 가림 환경 및 고교통 15,000대/시를 사용했다. R1~R2의 Adam 학습률은 0.0002, R3~R5는 0.0001이며 매번 12에포크 중 같은 오프라인 점수로 체크포인트를 골랐다. 학생 수집은 각 720회이고, 학생의 주행 시간에 따라 누적 상태 수는 달라진다. 평가의 경로·시드·출발 차로와 초기 관측 해시를 짝별로 검증했다.",
          "", "## 라운드별 128회 주행", "",
          "각 셀은 기본형 → 보강형 건수다. 공통 출발 BC는 107회 완주, 17회 충돌, 4회 이탈이었다.",
          "", "| 라운드 | 완주 | 충돌 | 차로 끝 정지 | 일반 시간초과 | 실제 이탈 | 10초 이상 대기 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k,r in result["rounds"].items():
        text.append("| R"+str(k)+" | "+" | ".join(f"{r['uniform'][s]} → {r['enhanced'][s]}" for s in (*ex.OUTCOMES,"slow10"))+" |")
    text+=["","## R5 고정 가중치의 홀드아웃 평가","",
           "| 패널 | 모델 | 완주 | 충돌 | 끝 정지 | 시간초과 | 실제 이탈 | 10초 이상 대기 | 완주율 | 충돌률 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,r in result["panels"].items():
        for model,label in (("uniform","기본"),("enhanced","보강")):
            c=r[model]
            n=sum(c[s] for s in ex.OUTCOMES)
            text.append("| "+labels[key]+" | "+label+" | "+" | ".join(str(c[s]) for s in (*ex.OUTCOMES,"slow10"))+f" | {c['arrival']/n:.2%} | {c['collision']/n:.2%} |")
    text+=["","두 256회 패널은 이전 보강형 실험에서도 사용했다. 이 비교에서는 모델별 R5를 고정한 뒤 평가했다. 단일 초기화·학습 시드에서 전체 보강을 함께 제거한 결과이므로 각 보강 항목 단독 효과는 이 비교의 측정 대상 밖이다.",
           "", "[실행 코드](../../utils/jys3_uniform_dagger.py) · [결과와 시드별 전환 집계](../../results/jys3_uniform_dagger/summary.json) · [기본 R5 가중치](../../results/jys3_uniform_dagger/dagger5.pt)"]
    for key,label in (("final256","지정"),("novel256","추가")):
        c=result["panels"][key]
        text.append(f"\n{label} 256회에서 기본형 완주 {c['uniform']['arrival']}건, 보강형 완주 {c['enhanced']['arrival']}건이었다. 기본형의 끝 정지는 {c['uniform']['stall']}건, 일반 시간초과는 {c['uniform']['timeout']}건이었다.")
    (ex.DOCS/"jys3-uniform-dagger.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def main():
    ROOT.mkdir(exist_ok=True)
    ex.torch.set_num_threads(2)
    previous=ex.checkpoint(0)
    start_hash=hashlib.sha256(previous.read_bytes()).hexdigest()
    assert read(ex.ROOT/"fit0.json")["sha256"]==start_hash
    selected=read(ex.ROOT/"expert.json")["short_reserve"]
    seeds=set()
    def reserve(values):
        assert len(values)==len(set(values)) and seeds.isdisjoint(values)
        seeds.update(values)
    def data(name):
        rows=read(ex.ROOT/name/"episodes.json")
        reserve([r["seed"] for r in rows])
        return ex.dataset(rows,name)
    x,y=data("train")
    vx,vy=data("valid")
    rounds={}
    panels=(("hold64",81260100,4,"hold64_5"),("final256",91000000,16,"final256"),
            ("novel256",104320000,16,"novel256"))
    for base,n in [(95380000,8)]+[(b,n) for _,b,n,_ in panels]:
        reserve(ex.planned(base,n))
    result=dict(start_sha256=start_hash,rounds=rounds,panels={})
    with ProcessPoolExecutor(10) as pool:
        for k in range(1,6):
            base=78000000+100000*k
            reserve(ex.planned(base,9))
            folder=ROOT/f"R{k}"
            rows=ex.schedule(pool,folder,base,9,selected,previous,collect=True)
            assert len(rows)==144 and [r["seed"] for r in rows]==ex.planned(base,9)
            ref=read(ex.ROOT/f"R{k}"/"uniform"/"episodes.json")
            paired(rows[:128],ref)
            if k==1:
                assert all((r["steps"],r["outcome"])==(s["steps"],s["outcome"]) for r,s in zip(rows,ref))
            xx,yy=ex.dataset(rows,folder)
            x,y=ex.torch.cat((x,xx)),ex.torch.cat((y,yy))
            ex.fit(k,x,y,vx,vy,[],uniform_dir=ROOT,initial=previous)
            previous=ROOT/f"dagger{k}.pt"
            screen=ex.schedule(pool,ROOT/f"screen{k}",95380000,8,selected,previous)
            ref=read(ex.ROOT/f"screen{k}"/"episodes.json")
            rounds[k]=dict(uniform=ex.tally(screen),enhanced=ex.tally(ref),
                           transitions=paired(screen,ref),states=len(x))
            save(ROOT/"rounds.json",rounds)
        for name,base,n,reference in panels:
            rows=ex.schedule(pool,ROOT/name,base,n,selected,previous)
            ref=read(ex.ROOT/reference/"episodes.json")
            result["panels"][name]=dict(uniform=ex.tally(rows),enhanced=ex.tally(ref),
                                        transitions=paired(rows,ref))
    save(ROOT/"summary.json",result)
    report(result)


if __name__=="__main__":
    main()
