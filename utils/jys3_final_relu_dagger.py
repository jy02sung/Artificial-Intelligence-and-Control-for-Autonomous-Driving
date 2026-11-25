"""Select the final expert, then train and validate its ReLU hybrid imitation policy."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib

from utils import jys3_rolling_margin_eval as margin
import numpy as np
import torch
import torch.nn.functional as F
from utils.networks import HybridActorCritic
from utils.experiment_c_train import atomic_torch

p = margin.panel
ROOT = p.ROOT.parent / "jys3_final_relu_dagger"
DOCS = ROOT.parent.parent / "docs" / "experiments"
FOCUS = (68173405, 94600413, 94601313, 94600014, 94600214)
ROUTES = tuple(range(1,17))
OUTCOMES = ("arrival","collision","stall","timeout","off_route")


def candidate_batch(jobs):
    with p.roll.baseline.simulator() as env:
        return [p.drive(env, job, revised=True, short_reserve=True) for job in jobs]


def phase1(pool):
    decision_file = ROOT / "expert.json"
    if decision_file.exists():
        return p.read(decision_file)
    jobs = p.cases()
    rows = [r for part in pool.map(candidate_batch, [jobs[i:i+8] for i in range(0,512,8)]) for r in part]
    before = {r["seed"]:r for r in p.read(margin.ROOT / "episodes.json")}
    after = {r["seed"]:r for r in rows}
    assert after.keys() == before.keys() and len(rows) == len(after) == 512
    for seed, row in after.items():
        assert all(row[k] == before[seed][k] for k in ("route","lane","group"))
    stats = p.summarize(rows)
    old = p.summarize(list(before.values()))
    changes = [[s, margin.slow_steps(before[s], margin.CONNECTOR), margin.slow_steps(after[s], margin.CONNECTOR)] for s in FOCUS]
    # If the total number is unchanged, require clear relief in the five named cases.
    focus_better = all(b <= a for _,a,b in changes) and sum(a-b >= 50 for _,a,b in changes) >= 3
    selected = stats["outcomes"] == {"arrival":512} and (stats["slow_10s"] < old["slow_10s"] or focus_better)
    decision = dict(short_reserve=selected, candidate=stats, baseline=old, focus_steps=changes)
    p.save(ROOT / "expert_panel.json",rows)
    p.save(decision_file,decision)
    return decision


def expert_report(decision):
    text = ["# 최종 시범 전문가 선정", "",
            "같은 고교통 512회에서 저속 도로의 후단 여유 거리만 24m에서 12m로 줄였다. 시드·경로·출발 차로를 짝지었고 재시도 없이 평가했다.",
            "", "| 조건 | 완주 | 충돌 | 끝 정지 | 시간초과 | 실제 이탈 | 10초 이상 대기 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for label,key in (("기존 24m","baseline"),("저속 12m","candidate")):
        stats=decision[key]
        text.append("| "+label+" | "+" | ".join(str(stats["outcomes"].get(k,0)) for k in OUTCOMES)+f" | {stats['slow_10s']} |")
    text += ["", "| 문제 연결로 시드 | 기존 대기(초) | 후보 대기(초) |","|---|---:|---:|"]
    for seed,b,a in decision["focus_steps"]:
        text.append(f"| {seed} | {b*.2:.1f} | {a*.2:.1f} |")
    chosen="저속 12m 후보" if decision["short_reserve"] else "기존 839e2d5의 24m 전문가"
    text += ["",f"최종 전문가로 {chosen}를 선정했다. 두 조건 모두 50개 관측과 같은 차로변경 안전 간격을 사용했다.",
             "완주 512건을 전제로 10초 이상 대기 건수가 줄거나, 문제 5건 모두 악화 없이 3건 이상에서 10초 이상 줄면 후보를 선택하도록 실행 전에 정했다."]
    (DOCS/"jys3-expert-final.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def network(checkpoint=None):
    model=HybridActorCritic(50,hidden_sizes=(256,256),activation="relu")
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["policy"])
    return model.eval()


def checkpoint(k):
    return ROOT / ("bc.pt" if k==0 else f"dagger{k}.pt")


def run_episodes(task):
    name, cases, selected, model_path, collect, demo = task
    torch.set_num_threads(1)
    model = network(model_path) if model_path else None
    folder=ROOT/name
    folder.mkdir(exist_ok=True,parents=True)
    rows=[]
    with p.roll.baseline.simulator() as env, torch.inference_mode():
        assert env.sim.physical_forks and not env.sim.lc_safety_enabled and env.sim.occlusion
        assert np.isclose(env.sim.step_length,.2)
        for route,rep,base in cases:
            path=f"route_{route:02d}"
            lane=min(rep%2,env.counts[env.routes[path][0]]-1)
            env.ego_depart_lane=str(lane)
            seed=base+rep*100+route
            while True:
                obs,_=env.reset(seed=seed,options={"route":path})
                digest=hashlib.sha256(obs.tobytes()).hexdigest()
                slow=longest=0
                previous=None
                xs,ys=[],[]
                for step in range(1,1001):
                    assert obs.shape==(52,) and not obs[50:].any()
                    state=obs[:50]
                    label,wait,direction,_=p.recovery.control(state,env,"queue_tail",revised=True,short_reserve=selected)
                    key=(*env.snapshot[:2],direction) if wait and env.snapshot[7]<1. else None
                    slow=(slow+1 if key==previous else 1) if key else 0
                    longest=max(longest,slow)
                    previous=key
                    if collect:
                        xs.append(state.copy())
                        ys.append(label)
                    action=label if model is None else model.det_action(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                    obs,_,done,cut,info=env.step(action)
                    if done or cut:
                        break
                assert (done or cut) and info["outcome"] in OUTCOMES
                assert info["outcome"]!="arrival" or info["route_intact"]
                accepted=not demo or (info["outcome"]=="arrival" and (name!="train" or longest<50))
                row=dict(route=route,rep=rep,lane=lane,seed=seed,steps=step,outcome=info["outcome"],
                         longest_slow=longest,accepted=accepted,initial_sha256=digest)
                rows.append(row)
                if accepted:
                    if collect:
                        p.roll.baseline.save_samples(folder,row,np.stack(xs),np.stack(ys))
                    break
                seed+=10000
                assert seed < base+1000000, "replacement would enter a different seed family"
    return rows


def schedule(pool,name,base,reps,selected,model_path=None,collect=False,demo=False,routes=ROUTES):
    folder=ROOT/name
    folder.mkdir(exist_ok=True,parents=True)
    index=folder/"episodes.json"
    if index.exists():
        rows=p.read(index)
    else:
        cases=[(route,rep,base) for rep in range(reps) for route in routes]
        tasks=[(name,cases[i:i+4],selected,model_path,collect,demo) for i in range(0,len(cases),4)]
        rows=[row for part in pool.map(run_episodes,tasks) for row in part]
        p.save(index,rows)
    accepted=[r for r in rows if r["accepted"]]
    assert [(r["route"],r["rep"]) for r in accepted]==[(r,i) for i in range(reps) for r in routes]
    assert len({r["seed"] for r in rows})==len(rows)
    assert not collect or all((folder/r["file"]).exists() for r in accepted)
    print(name,len(rows),dict(Counter(r["outcome"] for r in accepted)),flush=True)
    return rows


def collect_demos(pool):
    decision=phase1(pool)
    expert_report(decision)
    selected=decision["short_reserve"]
    train=schedule(pool,"train",76000000,10,selected,collect=True,demo=True)
    valid=schedule(pool,"valid",77000000,2,selected,collect=True,demo=True)
    return selected,train,valid


def dataset(rows,name):
    accepted=[r for r in rows if r["accepted"]]
    return p.roll.baseline.dagger.load_rollouts(accepted,ROOT/name)


def fit(k,x,y,vx,vy,critical,*,extended_bc=False,uniform_dir=None,initial=None):
    assert not extended_bc or k==0
    destination = uniform_dir if uniform_dir is not None else (ROOT/"bc100" if extended_bc else ROOT)
    destination.mkdir(exist_ok=True)
    folder=destination/f"fit{k}.json"
    best_path=destination/checkpoint(k).name
    if folder.is_file() and best_path.is_file():
        return
    torch.manual_seed(11)
    model=network(initial if initial is not None else (checkpoint(k-1) if k else None))
    lr=.0003 if k==0 else .0002 if k<=2 else .0001 if k<=5 else .00005
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator().manual_seed(11+k)
    cls=(1-y[:,1]).long()
    vc=(1-vy[:,1]).long()
    changed=torch.where(cls!=1)[0]
    critical=torch.tensor(critical,dtype=torch.long)
    empty=critical[:0]
    def sample(indices):
        return indices[torch.randint(len(indices),(32,),generator=generator)] if len(indices) else empty
    ledger=[]
    best=-float("inf")
    epochs=12 if k else (100 if extended_bc else 50)
    for epoch in range(1,epochs+1):
        disagree=empty
        if k and uniform_dir is None:
            model.eval()
            with torch.no_grad():
                predicted=torch.cat([model(part)[2].argmax(1) for part in x.split(4096)])
                disagree=torch.where(predicted!=cls)[0]
        model.train()
        total_acc=total_lane=0.
        batches=0
        for base in torch.randperm(len(x),generator=generator).split(256):
            if uniform_dir is None:
                acc_ids=torch.cat((base,sample(critical)))
                ids=torch.cat((acc_ids,sample(changed),sample(disagree)))
            else:
                acc_ids=ids=base
            mu,_,logits,_=model(x[ids])
            loss_acc=F.mse_loss(mu[:len(acc_ids),0],y[acc_ids,0])
            loss_lane=F.cross_entropy(logits,cls[ids])
            optimizer.zero_grad(set_to_none=True)
            (loss_acc+loss_lane).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total_acc+=loss_acc.item()
            total_lane+=loss_lane.item()
            batches+=1
        model.eval()
        with torch.no_grad():
            mu,_,z,_=model(vx)
            conf=torch.bincount(vc*3+z.argmax(1),minlength=9).reshape(3,3)
            f1=2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)
            mae=(mu[:,0]-vy[:,0]).abs().mean().item()
            score=float((f1[0]+f1[2])/2)-.1*mae
        ledger.append(dict(epoch=epoch,score=score,f1=f1.tolist(),mae=mae,disagree=len(disagree),
                           mse=total_acc/batches,cross_entropy=total_lane/batches))
        if score>best:
            best,chosen=score,epoch
            atomic_torch(dict(policy=model.state_dict(),activation="relu",hidden_sizes=[256,256],
                              obs_dim=50,seed=11,round=k,epoch=epoch),best_path)
        if extended_bc and epoch==50:
            reference=p.read(ROOT/"fit0.json")
            assert ledger==reference["epochs"] and chosen==reference["selected_epoch"]
            a=torch.load(best_path,weights_only=True)["policy"]
            b=torch.load(checkpoint(0),weights_only=True)["policy"]
            assert all(torch.equal(a[key],b[key]) for key in a)
        if extended_bc and epoch==epochs:
            atomic_torch(dict(policy=model.state_dict(),activation="relu",hidden_sizes=[256,256],
                              obs_dim=50,seed=11,round=0,epoch=epoch),destination/"last.pt")
        if epoch%5==0:
            print("fit",k,epoch,round(score,4),flush=True)
    p.save(folder,dict(epochs=ledger,selected_epoch=chosen,lr=lr,states=len(x),critical=len(critical),
                       changed=len(changed),sha256=hashlib.sha256(best_path.read_bytes()).hexdigest()))


def planned(base,reps,routes=ROUTES):
    return [base+rep*100+r for rep in range(reps) for r in routes]


def tally(rows):
    c=Counter(r["outcome"] for r in rows)
    return {**{reason:c[reason] for reason in OUTCOMES},"slow10":sum(r["longest_slow"]>=50 for r in rows)}


def pipeline(pool):
    selected,train,valid=collect_demos(pool)
    torch.set_num_threads(2)
    x,y=dataset(train,"train")
    vx,vy=dataset(valid,"valid")
    seeds=set()
    def reserve(values):
        new=set(values)
        assert len(new)==len(values) and new.isdisjoint(seeds), "seed overlap"
        seeds.update(new)
    for rows in (train,valid):
        reserve([r["seed"] for r in rows])
    for base,reps in ((95380000,8),(81260100,4),(91000000,16),(104320000,16)):
        reserve(planned(base,reps))
    screens={}
    counts={}
    critical=[]
    def rank(k):
        c=counts[k]
        return c["arrival"],-c["collision"],-c["off_route"],-k
    for k in range(9):
        if k:
            fails=Counter(r["route"] for r in screens[k-1] if r["outcome"]!="arrival")
            weak=sorted(ROUTES,key=lambda r:(-fails[r],r))[:4]
            for label,base,n,routes in (("uniform",78000000,8,ROUTES),("weak",79000000,4,weak)):
                name=f"R{k}/{label}"
                base+=k*100000
                reserve(planned(base,n,routes))
                rows=schedule(pool,name,base,n,selected,checkpoint(k-1),collect=True,routes=routes)
                assert [r["seed"] for r in rows]==planned(base,n,routes)
                offset=len(x)
                for r in rows:
                    end=offset+r["steps"]
                    if r["outcome"]!="arrival":
                        critical.extend(range(max(offset,end-15),end))
                    offset=end
                nx,ny=dataset(rows,name)
                x,y=torch.cat((x,nx)),torch.cat((y,ny))
                assert len(x)==offset
        fit(k,x,y,vx,vy,critical)
        rows=schedule(pool,f"screen{k}",95380000,8,selected,checkpoint(k))
        assert [r["seed"] for r in rows]==planned(95380000,8)
        if k:
            assert all(a["initial_sha256"]==b["initial_sha256"] for a,b in zip(screens[0],rows))
        screens[k],counts[k]=rows,tally(rows)
        p.save(ROOT/"rounds.json",counts)
        if k==5:
            c=counts[max(range(1,6),key=rank)]
            if c["arrival"]>=122 and c["collision"]<=6:
                break
    best=max(screens,key=rank)
    p.save(ROOT/"selection.json",dict(best=best,last=k,sha256=p.read(ROOT/f"fit{best}.json")["sha256"]))
    hold={}
    for m in sorted({0,best}):
        hold[m]=schedule(pool,f"hold64_{m}",81260100,4,selected,checkpoint(m))
    assert all(a["initial_sha256"]==b["initial_sha256"] for a,b in zip(hold[0],hold[best]))
    final=schedule(pool,"final256",91000000,16,selected,checkpoint(best))
    novel=schedule(pool,"novel256",104320000,16,selected,checkpoint(best))
    for rows,base,reps in ((hold[0],81260100,4),(hold[best],81260100,4),(final,91000000,16),(novel,104320000,16)):
        assert [r["seed"] for r in rows]==planned(base,reps)
    def byroute(rows):
        return {r:tally([row for row in rows if row["route"]==r]) for r in ROUTES}
    result=dict(best=best,last=k,rounds=counts,hold={m:tally(rows) for m,rows in hold.items()},
                transitions=dict(Counter(a["outcome"]+" -> "+b["outcome"] for a,b in zip(hold[0],hold[best]))),
                final=tally(final),novel=tally(novel),screen_routes=byroute(screens[best]),
                final_routes=byroute(final),novel_routes=byroute(novel),distinct_seeds=len(seeds))
    p.save(ROOT/"summary.json",result)
    return result


def final_report(result):
    def name(i):
        return "BC" if int(i)==0 else f"DAgger R{i}"
    def line(label,c):
        return "| "+label+" | "+" | ".join(str(c[key]) for key in (*OUTCOMES,"slow10"))+" |"
    def outcomes(c):
        return " / ".join(str(c[key]) for key in OUTCOMES)
    def verdict(c):
        return "충족" if c["arrival"]>=244 and c["collision"]<=12 and c["stall"]+c["timeout"]==0 else "미달"
    decision=p.read(ROOT/"expert.json")
    selected="저속 도로 12m" if decision["short_reserve"] else "기존 24m"
    text=["# 최종 전문가의 ReLU Hybrid BC와 DAgger",
          "", "| 항목 | 수업 기본 BC | 이번 실험과 이유 |", "|---|---|---|",
          "| 유지 | 256×256, ReLU, Adam 0.0003, 기본 배치 256, 50에포크 | BC에서 동일하게 적용했다. 가감속 지도 손실도 MSE다. |",
          "| 관측 | 학습 데이터 평균·표준편차로 표준화 | 환경에서 척도화한 50개 관측을 HybridActorCritic에 직접 전달했다. |",
          "| 출력 및 손실 | 연속 행동 두 개를 MSE로 회귀 | 가감속은 tanh 평균, 차로는 좌·유지·우 3클래스 교차엔트로피다. 연속값 양자화가 차로변경 명령을 유지로 바꾸는 문제를 피한다. |",
          "| 학습 배치 | 균일 배치 | BC는 차로변경 정답 32개를 추가했다. DAgger는 현재 오답 32개와 학생 실패 직전 15스텝에서 32개도 보강했다. 후자에는 가감속 손실도 적용했다. |",
          "| 검증·시드 | 에피소드 20% 검증, 검증 MSE 최소, 시드 0 | 별도 전문가 32회로 좌·우 평균 F1 − 0.1 × 가감속 MAE가 최대인 에포크를 선택했다. 초기화 시드 11, 그래디언트 노름 상한 1.0을 사용했다. |",
          "| 후속 학습 | 단일 BC | 학생이 방문한 상태에 전문가 정답을 붙였다. 각 라운드 144회씩 누적해 12에포크씩 업데이트했다. 값 함수·보상·PPO 업데이트는 사용하지 않았다. |",
          "", "## 최종 전문가와 시범", "",
          f"{selected} 전문가를 확정했다. 기존과 후보의 512회 결과와 문제 시드별 대기는 [전문가 보고서](jys3-expert-final.md)에 기록했다.",
          "모든 주행은 고교통 15,000대/시, 물리적 분기, 고가 교차 20쌍 보정, 안전게이트 OFF에서 실행했다.",
          "", "| 용도 | 주행 시도 | 채택 | 제외 및 시드 보충 | 채택 스텝 |", "|---|---:|---:|---:|---:|"]
    for folder,label in (("train","학습 시범"),("valid","오프라인 검증")):
        rows=p.read(ROOT/folder/"episodes.json")
        accepted=[r for r in rows if r["accepted"]]
        text.append(f"| {label} | {len(rows)} | {len(accepted)} | {len(rows)-len(accepted)} | {sum(r['steps'] for r in accepted):,} |")
    text+=["","학습 시범은 지정 경로 완주와 10초 미만 저속 대기를 모두 요구했다. 검증 시범은 완주를 요구했다. 제외한 주행까지 시드 검증에 포함했다. 저속 대기는 같은 도로·차로·합류 방향에서 전문가가 간격 부족으로 기다리며 실제 속도가 1m/s 미만인 상태가 50스텝 이상 연속된 주행 수다. 학생 평가의 속도와 주행은 학생 정책의 출력으로 결정했다.",
           "", "## 라운드별 고교통 128회", "",
           "| 모델 | 완주 | 충돌 | 끝 정지 | 시간초과 | 실제 이탈 | 10초 이상 대기 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k,c in result["rounds"].items():
        text.append(line(name(k),c))
    best=result["best"]
    text+=["",f"선정 모델은 **{name(best)}**이다. 128회에서 완주 수 최대, 충돌 수 최소, 실제 이탈 최소, 이른 라운드 순으로 선정했다. 취약 경로 실패 수가 같은 경우 경로 번호가 작은 순서로 보강했다.",
           "", "## 선정 후 검증", "", "| 평가 · 모델 | 완주 | 충돌 | 끝 정지 | 시간초과 | 실제 이탈 | 10초 이상 대기 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m,c in result["hold"].items():
        text.append(line("64회 · "+name(m),c))
    text+=[line("지정 256회 · "+name(best),result["final"]),
           line("추가 신규 256회 · "+name(best),result["novel"]),""]
    for key,label in (("final","지정 91M 패널"),("novel","추가 신규 패널")):
        c=result[key]
        text.append(f"- {label}은 완주 {c['arrival']/256:.2%}, 충돌 {c['collision']/256:.2%}, 끝 정지·시간초과 {c['stall']+c['timeout']}건으로 목표 **{verdict(c)}**이다.")
    text+=["",f"서로 다른 실제 사용 시드는 {result['distinct_seeds']:,}개다. 같은 평가 패널은 모델 간 비교에 재사용했다. 지정 91M 대역은 이전 프로젝트 실험에서 평가한 기록이 있어 104320000 대역 256회를 추가했다. 두 패널 모두 이번 모델의 학습과 선정에서 분리했다.",
           "", "## 선정 모델 경로별 결과","", "셀 안의 건수 순서는 완주 / 충돌 / 끝 정지 / 시간초과 / 실제 이탈이다.","",
           "| 경로 | 선정 패널(각 8회) | 지정 홀드아웃(각 16회) | 추가 신규 홀드아웃(각 16회) |","|---:|---|---|---|"]
    for route in result["screen_routes"]:
        text.append(f"| {route} | "+" | ".join(outcomes(result[k][route]) for k in ("screen_routes","final_routes","novel_routes"))+" |")
    filename="bc.pt" if best==0 else f"dagger{best}.pt"
    text+=["",f"[선정 가중치](../../results/jys3_final_relu_dagger/{filename}) · [BC 가중치](../../results/jys3_final_relu_dagger/bc.pt) · [실행 코드](../../utils/jys3_final_relu_dagger.py) · [집계 원자료](../../results/jys3_final_relu_dagger/summary.json)",
           "", "라운드별 선택 에포크·검증 점수·손실은 결과 폴더의 fit 파일에 기록했다. 에피소드별 시드와 종료 사유는 각 하위 폴더의 episodes.json에, 스텝별 시범과 전문가 라벨은 로컬의 같은 하위 폴더에 압축 파일로 보관했다."]
    (DOCS/"jys3-final-relu-dagger.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(10) as pool:
        final_report(pipeline(pool))
