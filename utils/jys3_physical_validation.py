"""Paired frozen-policy evaluation of all 16 missions with lane-driven branch traversal."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from utils import jys3_physical_forks as physical

v3 = physical.v3
ROOT = v3.ROOT.parent / "physical_forks" / "paired_300k_600k"
PANELS = {300000: (v3.SCREEN,), 600000: (v3.PAIRED, v3.FRESH, v3.NOVEL)}
SEEDS = (11, 22, 33)
OUTCOMES = ("arrival", "collision", "stall", "timeout", "off_route")


def source(arm, seed):
    root = v3.ROOT if arm == v3.ARMS[0] else v3.ROOT / "s2_extension"
    return root / "stage2" / f"{arm}_{seed}"


def result_path(arm, seed, step, panel, rep):
    return ROOT / f"{arm}_{seed}" / f"eval_{panel}_{step}_{rep}.json"


def evaluate(arm, seed, step, panel, rep):
    folder = source(arm, seed)
    checkpoint = folder / f"policy_{step}.pt"
    model = v3.core.agent(seed)
    model.policy.load_state_dict(v3.j.torch.load(checkpoint, weights_only=True)["policy"])
    model.policy.eval()
    rows = []
    with TemporaryDirectory() as temp:
        env = physical.Env(arm, Path(temp), seed)
        try:
            for route in range(1, 17):
                name = f"route_{route:02d}"
                env.ego_depart_lane = str(min(rep % 2, env.counts[env.routes[name][0]] - 1))
                traffic = panel + rep // 2 * 100 + route
                obs, _ = env.reset(seed=traffic, options={"route": name})
                start = [env.snapshot[i] for i in (0, 1, 7, -1)]
                forks = []
                for t in range(1, env.max_steps + 1):
                    before = env.snapshot
                    action = model.predict(obs, deterministic=True)
                    obs, reward, done, cut, info = env.step(action)
                    if info["fork_passed"]:
                        forks.extend(info.get("passed_edges", [env.last_external[1]]))
                    if done or cut:
                        break
                outcome = info["outcome"]
                assert outcome in OUTCOMES
                pos = env.snapshot or before
                edge, lane, path, index = pos[:4]
                expected = path[index+1] if index+1 < len(path) else None
                exits = env.links.get(edge, {})
                available = sorted(nxt for nxt, lanes in exits.items() if lane in lanes)
                detail = info.get("off_route_detail")
                if outcome == "off_route":
                    assert done and reward == -5 and detail["actual"] == edge
                rows.append(dict(route=route, start=start, traffic_seed=traffic, case=rep, steps=t,
                                 outcome=outcome, forks=forks, edge=edge, lane=lane,
                                 position_after_step=env.snapshot is not None, next_mission=expected,
                                 available=available, lane_end_m=round(pos[-1],3),
                                 terminal_speed=round(pos[7],3), off_route_detail=detail))
        finally:
            env.close()
    path = result_path(arm, seed, step, panel, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    v3.save(path, dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), rows=rows))


def jobs():
    return [(a,s,n,p,r) for n, panels in PANELS.items() for p in panels
            for r in range(4) for s in SEEDS for a in v3.ARMS]


def run(workers):
    pending = [job for job in jobs() if not result_path(*job).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        while pending or active:
            for job in list(pending):
                if len(active) >= workers:
                    break
                arm, seed, step, _, _ = job
                if (source(arm,seed) / f"policy_{step}.pt").exists():
                    active[pool.submit(evaluate,*job)] = job
                    pending.remove(job)
            if not active:
                time.sleep(10)
                continue
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
                print("verified", *active.pop(future), flush=True)


def report():
    pairs = defaultdict(list)
    finished = matched = 0
    hashes = {}
    for arm,seed,step,panel,rep in jobs():
        path = result_path(arm,seed,step,panel,rep)
        if not path.exists():
            continue
        replay = json.loads(path.read_bytes())
        checkpoint = source(arm,seed) / f"policy_{step}.pt"
        if checkpoint not in hashes:
            hashes[checkpoint] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert replay["checkpoint_sha256"] == hashes[checkpoint]
        new = replay["rows"]
        finished += len(new)
        old_path = source(arm,seed) / path.name
        if not old_path.exists():
            continue
        old = json.loads(old_path.read_bytes())
        assert len(old) == len(new) == 16
        for route,(a,b) in enumerate(zip(old,new),1):
            assert a["route"] == b["route"] == route and a["case"] == b["case"] == rep
            assert a["start"] == b["start"] and a["traffic_seed"] == b["traffic_seed"]
            pairs[step,panel,arm].append((a,b))
        matched += len(new)
    total = len(jobs())*16
    lines = ["# S1·S2 수정 환경 고정 가중치 검증","",
             "분기 외의 도로에서도 오진입이나 정지가 생기는지 확인하기 위해 16개 경로 전체를 재생한다. 정책 업데이트는 0회다. "
             "학습 시드 11·22·33의 30만 가중치는 기존 중간 평가 시드군 41110000에서 조건당 192회, "
             "60만 가중치는 기존 독립 교통 시드군 3개에서 조건당 576회 검증한다. "
             "각 비교에서 가중치와 출발 도로·차로·속도·잔여 거리, 교통 시드, 경로 번호를 맞춘다.","",
             f"수정 환경 재생 {finished}/{total}회, 기존 기록 대조 {matched}/{total}회. "
             "아래 표에는 해당 교통 시드군의 3개 학습 시드가 모두 끝난 결과만 표시한다.","",
             "|스텝|교통 시드군|모델|환경|완주|충돌|끝 정지|일반 시간초과|실제 이탈|13번 목표 진출|16번 0차로 본선|",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    full = {k:r for k,r in pairs.items() if len(r)==192}
    for (n,p,a),r in sorted(full.items()):
        for label,side in [("기존",0),("수정",1)]:
            rows=[pair[side] for pair in r]
            c=Counter(x["outcome"] for x in rows)
            hits13=sum(x["route"]==13 and "169728835" in x["forks"] for x in rows)
            hits16=sum(x["route"]==16 and x["start"][1]==0 and "86198725" in x["forks"] for x in rows)
            cells=[f"{c[k]} ({c[k]/192:.1%})" for k in OUTCOMES]
            lines.append("|"+"|".join(map(str,[n,p,a[:2],label,*cells,f"{hits13}/12",f"{hits16}/6"]))+"|")
    for n in PANELS:
        for arm in v3.ARMS:
            samples=[r for (step,p,a),rs in full.items() if step==n and a==arm for r in rs]
            if not samples:
                continue
            lines += ["",f"## {arm[:2]} {n:,}스텝 · 동일 조건 {len(samples)}회","",
                      "|기존 → 수정|완주|충돌|끝 정지|일반 시간초과|실제 이탈|","|---|---:|---:|---:|---:|---:|"]
            counts=Counter((old["outcome"],new["outcome"]) for old,new in samples)
            labels=dict(zip(OUTCOMES,("완주","충돌","끝 정지","일반 시간초과","실제 이탈")))
            for k in (*OUTCOMES,"other"):
                if any(counts[k,q] for q in OUTCOMES):
                    lines.append("|"+labels.get(k,"기타")+"|"+"|".join(str(counts[k,q]) for q in OUTCOMES)+"|")
            lines += ["","|경로|완주|충돌|끝 정지|일반 시간초과|실제 이탈|","|---:|---:|---:|---:|---:|---:|"]
            for route in range(1,17):
                counts=Counter(x["outcome"] for _,x in samples if x["route"]==route)
                lines.append(f"|{route}|"+"|".join(str(counts[q]) for q in OUTCOMES)+"|")
            sites=defaultdict(Counter)
            for _,x in samples:
                k=x["outcome"]
                if k=="arrival":
                    continue
                d=x["off_route_detail"]
                loc=(f"{d['src']} → {d['actual']} (지정 {d['expected']})" if d
                     else f"{x['edge']} · {x['lane']}차로")
                if not x["position_after_step"]:
                    loc+=" (종료 직전 위치)"
                sites[k,loc][x["route"]]+=1
            lines+=["","|종료 유형|도로와 차로 또는 실제 이동|건수|경로(건수)|","|---|---|---:|---|"]
            for (k,loc),routes in sorted(sites.items(),key=lambda x:(x[0][0],-sum(x[1].values()),x[0][1])):
                lines.append(f"|{labels[k]}|{loc}|{sum(routes.values())}|"+", ".join(f"{r}({c})" for r,c in sorted(routes.items()))+"|")
    lines+=["","충돌로 차량이 사라지면 마지막으로 관측한 도로·차로를 표시한다. 표의 분기 통과는 해당 지점 통과를 의미한다. "
            "주행별 원본은 로컬 결과 폴더 `physical_forks/paired_300k_600k`에 보관한다. "
            "[실행 코드](../../utils/jys3_physical_validation.py) · [환경 수정](jys3-physical-forks.md)"]
    doc=Path(__file__).resolve().parents[1]/"docs/experiments/jys3-physical-validation.md"
    doc.write_text("\n".join(lines)+"\n", encoding="utf8")
    print(f"replayed={finished}/{total} paired={matched}/{total}")
    return matched == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", action="store_true")
    args=parser.parse_args()
    if not args.report:
        run(args.workers)
    report()
