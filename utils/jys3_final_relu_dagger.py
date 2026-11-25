"""Select the final expert, then train and validate its ReLU hybrid imitation policy."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
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


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(10) as pool:
        collect_demos(pool)
