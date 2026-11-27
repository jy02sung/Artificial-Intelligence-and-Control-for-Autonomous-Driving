"""Matched M5-FT fine-tuning controls. Run separate arms on different computers."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from utils import jys3_ppo_clean_ladder as L

P, C, A = L.p, L.c, L.anti
ROOT = L.ROOT.parent / "jys3_m5ft_ablation"
MARKS = tuple(L.marks("M5-FT"))
SEEDS = (22, 33)
LOCAL = ("no_warm", "higher_lr", "original_entropy", "no_kl")
ALL = LOCAL
FULL = "full"


def folder(arm, seed):
    return ROOT / arm / f"s{seed}"


def original(seed):
    return L.folder("M5-FT", seed)


def settings(arm):
    assert arm in (*ALL, FULL)
    return dict(warm=0 if arm == "no_warm" else 10000,
                lr_multiplier=3 if arm == "higher_lr" else 1,
                lane_entropy=.002 if arm == "original_entropy" else .0005,
                beta=0 if arm == "no_kl" else .08,
                fork=3.5 if arm == "fork_plus" else 2.5,
                progress=8. if arm == "progress_plus" else 6.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(arm):
    cfg = settings(arm)
    assert sum(v != settings(FULL)[k] for k,v in cfg.items()) == 1
    hashes = {}
    for s in SEEDS:
        peak = A.load_peak(s)
        warm = L.fresh("M5-FT", s)
        record = L.read(original(s)/"warm_check.json")
        assert record["actor_bits_identical"] and P.digest(warm.policy) == record["sha256"]
        for key, val in peak.policy.state_dict().items():
            if not P.prev.is_value(key):
                assert L.torch.equal(val, warm.policy.state_dict()[key])
        for path in (A.checkpoint(s), original(s)/"warm.pt",original(s)/"policy_150000.pt"):
            hashes[str(path.relative_to(L.ROOT.parent))] = sha(path)
    return dict(settings=cfg,seeds=SEEDS,steps=MARKS,source_hashes=hashes,
                primary="fixed 150000, paired 576",secondary="screen-selected 576",
                panels=L.old.PANELS)


def train(job):
    arm,seed,n = job
    out = folder(arm,seed)
    if (out/f"policy_{n}.pt").exists():
        return
    cfg = settings(arm)
    def make_agent():
        return A.load_peak(seed) if not cfg["warm"] else L.fresh("M5-FT",seed)
    def make_episodes(env, name, path, *, training_seed):
        ep = L.Episodes(env,"M5-FT",path,training_seed=training_seed)
        assert ep.costed and ep.risk and env.progress_cap == 6. and L.old.driving.PASS_REWARD == 2.5
        env.progress_cap = cfg["progress"]
        L.old.driving.PASS_REWARD = cfg["fork"]
        return ep
    ref = A.load_peak(seed).policy.eval().requires_grad_(False) if cfg["beta"] else None
    initial = P.digest(make_agent().policy)
    def update(agent,ep,adv,ret):
        lr = 3e-5-2e-5*max(0,agent.num_timesteps-agent.n_steps)/(150000-agent.n_steps)
        for group in agent.optimizer.param_groups:
            group["lr"] = cfg["lr_multiplier"]*lr
        agent.lane_entropy_coef = cfg["lane_entropy"]
        m = C.verified_update(agent,ep,adv,ret,ref)
        m.update(ep.stats)
        m.update(lane_entropy_coef=cfg["lane_entropy"], beta=cfg["beta"])
        ep.stats.clear()
        return m
    P.train("M5-FT",n,directory=out,make_agent=make_agent,make_episodes=make_episodes,
            expected_hash=initial,training_seed=seed,update=update)


def panel(arm,s,n,h):
    filename = f"eval_{n}_{h}.json"
    if arm == FULL and (original(s)/filename).exists():
        return original(s)/filename
    return folder(arm,s)/filename


def part(job):
    arm,s,n,h,rep = job
    dest = folder(arm,s)
    checkpoint = (original(s) if arm == FULL else dest)/f"policy_{n}.pt"
    C.evaluate("M5-FT",s,n,h,rep,directory=dest,checkpoint=checkpoint,lateral=True)


def eval_batch(pool, keys):
    pending = [key for key in dict.fromkeys(keys) if not panel(*key).exists()]
    jobs = [(*key,rep) for key in pending for rep in range(L.old.PANELS[key[-1]][1])]
    for _ in pool.map(part,jobs):
        pass
    for arm,s,n,h in pending:
        rows = [r for i in range(L.old.PANELS[h][1]) for r in
                L.read(folder(arm,s)/"parts"/f"{n}_{h}_{i}.json")]
        base, reps = L.old.PANELS[h]
        assert [r["seed"] for r in rows] == L.final.planned(base,reps)
        L.old.paired(L.read(L.q.panel("Q1",s,A.PEAK[s],h))["rows"], rows)
        metrics = dict(Counter(r["outcome"] for r in rows))
        metrics = {k:metrics.get(k,0) for k in L.anti.OUT}
        metrics["required_arrived"] = sum(r["outcome"]=="arrival" and
                P.groups()[r["route"],r["lane"]]=="needs_change" for r in rows)
        L.save(panel(arm,s,n,h),dict(rows=rows,metrics=metrics))


def pick(arm,s):
    def rank(n):
        m=L.read(panel(arm,s,n,"screen"))["metrics"]
        return m["arrival"],-m["collision"],m["required_arrived"],-n
    return max(MARKS,key=rank)


def summary(arm,s,n):
    blocks=[L.read(panel(arm,s,n,h))["rows"] for h in L.HOLDS]
    counts=C.summary(blocks)
    for key in ("left_commands","right_commands"):
        counts[key]=sum(r[key] for b in blocks for r in b)
    return counts,blocks


def report(arms):
    outcomes = []
    table = []
    for arm in arms:
        for mode in ("fixed","selected"):
            changes = []
            for s in SEEDS:
                n = MARKS[-1] if mode=="fixed" else pick(arm,s)
                ref_n = MARKS[-1] if mode=="fixed" else pick(FULL,s)
                counts,blocks = summary(arm,s,n)
                full,refs = summary(FULL,s,ref_n)
                m3 = [L.read(L.q.panel("Q1",s,A.PEAK[s],h))["rows"] for h in L.HOLDS]
                transitions = [L.old.paired(x,y) for x,y in zip(refs,blocks)]
                retention = [L.old.paired(x,y) for x,y in zip(m3,blocks)]
                full_retention = [L.old.paired(x,y) for x,y in zip(m3,refs)]
                delta={k:counts[k]-full[k] for k in counts}
                outcomes.append(dict(arm=arm,mode=mode,seed=s,step=n,full_step=ref_n,counts=counts,
                    full=full,delta=delta,full_to_variant=transitions,m3_to_variant=retention,
                    m3_to_full=full_retention))
                changes.append(delta)
                table.append([arm,mode,s,n,ref_n,*(delta[k] for k in ("arrival","collision","off_route","stall","timeout","required","cutin"))])
            table.append([arm,mode,"평균 차이(%p)","—","—",
                *[f"{100*sum(d[k] for d in changes)/(2*(208 if k=='required' else 576)):+.2f}"
                  for k in ("arrival","collision","off_route","stall","timeout","required","cutin")]])
    out=ROOT/("_".join(arms))
    L.save(out/"comparison.json",outcomes)
    lines=["# M5-FT 단일 변경 대조","",
           "각 행은 변경 조건에서 원본 M5-FT를 뺀 값이다. fixed는 양쪽 모두 15만 스텝, selected는 각 조건의 선정 모델이다. 시드별 단위는 건, 평균 단위는 %p다."]
    lines+=L.md_table(["조건","비교","시드","변경 스텝","원본 스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주","컷인"],table)
    tmp=out/"report.tmp"
    tmp.write_text("\n".join(lines)+"\n",encoding="utf8")
    tmp.replace(out/"report.md")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("group",choices=("local",*ALL))
    ap.add_argument("--workers",type=int,choices=range(1,5),default=4)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    L.torch.set_num_threads(1)
    arms=LOCAL if args.group=="local" else (args.group,)
    configs={arm:validate(arm) for arm in arms}
    if args.check:
        print(json.dumps(configs,indent=2))
        return
    for arm,cfg in configs.items():
        path=ROOT/arm/"protocol.json"
        if path.exists():
            assert L.read(path)==json.loads(json.dumps(cfg))
        L.save(path,cfg)
    with ProcessPoolExecutor(args.workers) as pool:
        for n in MARKS:
            jobs=[(a,s,n) for a in arms for s in SEEDS]
            list(pool.map(train,jobs))
            eval_batch(pool,[(*j,"screen") for j in jobs])
            print("screen",n,[(a,s,L.read(panel(a,s,n,"screen"))["metrics"]["arrival"]) for a,s,_ in jobs],flush=True)
        eval_batch(pool,[(a,s,n,h) for a in (*arms,FULL) for s in SEEDS
                         for n in sorted({MARKS[-1],pick(a,s)}) for h in L.HOLDS])
    for a in arms:
        assert validate(a)==configs[a]
        report((a,))
    if len(arms)>1:
        report(arms)
    print("completed",arms,flush=True)


if __name__=="__main__":
    main()
