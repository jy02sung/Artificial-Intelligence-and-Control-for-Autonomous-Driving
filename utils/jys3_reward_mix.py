"""Compare reward-only interventions in the frozen jys3 driving environment."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_pure_f as core
from utils.jys3_pure_f_report import aggregate

j = core.j
BASE_ENV = core.FactorEnv
ROOT = core.ROOT / "reward_mix_v1"
ARMS = ("R1_knox_bal", "R2_cpo_stage", "R3_pot_safe")
MARKS = tuple(range(50000, 600001, 50000))
SCREEN, PAIRED, FRESH = 41110000, 43110000, 45110000
KEYS = ("progress", "arrival", "fork", "collision", "timeout", "idle_clear", "pot")


def read(path):
    return json.loads(path.read_bytes())


def collision(arm, before):
    if arm == ARMS[1]:
        return -5 if before < 200000 else -10 if before < 400000 else -15
    return -10


def idle(speed, gap, terminal):
    return -.02 if not terminal and speed is not None and speed < .5 and (gap is None or gap > 25) else 0.


def risk(gap, closing, horizon):
    if gap <= 0:
        return 1.
    return max(0., 1-gap/(horizon*closing)) if closing > 0 else 0.


def delta(previous, current):
    return 1.5*(.99*current-previous)


class MixEnv(BASE_ENV):
    progress_cap = 3.
    offset = 0

    def __init__(self, arm, folder, seed):
        super().__init__("F1", folder, seed)
        self.arm = arm
        self.clock = self.offset
        self.sums = Counter()
        assert self.progress_enabled and not self.nav_enabled and self.sim.max_steps == 1000
        assert self.visibility == 120 and self.obs_noise and self.occlusion and not self.lc_safety_enabled

    def reset(self, **kwargs):
        self.previous = 0.
        obs, info = super().reset(**kwargs)
        assert len(obs) == 52 and tuple(obs[-2:]) == (0., 0.)
        return obs, info

    def potential(self, direction):
        edge = self.snapshot[0]
        if edge.startswith(":"):
            return 0.
        assert not self.sim._occ_sensor_mask
        v = self.snapshot[7]
        front = j.traci.vehicle.getLeader(self.ego_id, self.visibility)
        maximum = 0.
        if front and front[0]:
            closing = v-j.traci.vehicle.getSpeed(front[0])
            if closing > .5:
                maximum = risk(front[1], closing, 2.5)
        for side in (-1, 1):
            for i, neighbor in enumerate(self.sim._scan_lane_neighbors(side)):
                if neighbor:
                    gap, speed = neighbor
                    if gap <= 0 or direction == side:
                        maximum = max(maximum, risk(gap, v-speed if i == 0 else speed-v, 2.))
        return -maximum

    def step(self, action):
        j.COLLISION_REWARD = collision(self.arm, self.clock)
        obs, reward, done, cut, info = super().step(action)
        self.clock += 1
        terminal = done or cut
        direction = self.sim._quantize_lane_change(action[self.lane_change_index])
        phi = self.potential(direction) if self.arm == ARMS[2] and not terminal else 0.
        parts = dict(progress=info["progress_bonus"], arrival=5.*info["arrived"],
                     fork=float(info["fork_passed"]), collision=j.COLLISION_REWARD*info["collided"],
                     timeout=info["timeout_reward"], idle_clear=idle(info["speed"], info["gap"], terminal),
                     pot=delta(self.previous, phi))
        self.previous = phi
        total = reward+parts["pot"]+parts["idle_clear"]
        assert math.isclose(total, sum(parts.values()), abs_tol=1e-7)
        self.sums.update(parts)
        return obs, total, done, cut, info


def setup():
    j.ARRIVAL_REWARD, j.PASS_REWARD = 5., 1.
    assert j.STALL_SECONDS == 30 and j.TIMEOUT_REWARD == -5
    core.ROOT, core.FactorEnv = ROOT, MixEnv


def digest(model):
    return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()


def run_train(job):
    arm, seed, end, root = job
    setup()
    d = root/f"{arm}_{seed}"
    d.mkdir(parents=True, exist_ok=False)
    model = core.agent(seed)
    env = MixEnv(arm, d/"map", seed)
    try:
        initial = digest(model)
        core.save(d/"initial.json",dict(sha256=initial))
        assert model.gamma == .99 and model.gae_lambda == .95
        sched = j.ScheduledEpisodes(env, seed, d/"episodes.jsonl")
        obs, _ = sched.reset()
        stats = [0., 0]
        for n in ([end] if end == 2048 else [k for k in MARKS if k <= end]):
            while model.num_timesteps < n:
                env.sums.clear()
                batch = min(2048, n-model.num_timesteps)
                obs, last, _ = model.collect_rollout(sched, obs, ep_stats=stats, steps=batch, timeout_is_failure=True)
                adv, ret = model.buffer.compute_gae(last, model.gamma, model.gae_lambda)
                metrics = model.update(adv, ret)
                assert all(math.isfinite(v) for v in metrics.values())
                assert model.optimizer.param_groups[0]["lr"] == .0001
                record = dict(step=model.num_timesteps, batch=batch, lr=.0001, rewards=dict(env.sums), **metrics)
                with (d/"train_updates.jsonl").open("a") as f:
                    print(json.dumps(record), file=f)
            j.atomic_torch(dict(policy=model.policy.state_dict()), d/f"policy_{n}.pt")
        return initial, record
    finally:
        env.close()


def check():
    setup()
    assert [idle(v,g,t) for v,g,t in [(0,None,False),(.49,26,False),(.49,25,False),(.5,30,False),(0,None,True)]] == [-.02,-.02,0,0,0]
    assert delta(0,0)==0 and delta(0,-risk(2,2,2.5))<0 and delta(-1,0)>0
    assert [collision(ARMS[1],n) for n in (199999,200000,399999,400000)]==[-5,-10,-10,-15]
    with TemporaryDirectory() as tmp:
        smoke = [run_train((a,11,2048,Path(tmp))) for a in ARMS]
        assert len(set(v[0] for v in smoke))==1
        assert all(abs(v[1]["approx_kl"])<.1 for v in smoke)
        env=MixEnv(ARMS[0],Path(tmp)/"arrival",11)
        try:
            bonuses=[]
            for r in range(1,17):
                env.ego_depart_lane="best"
                env.reset(seed=1000+r,options={"route":f"route_{r:02d}"})
                bonus=0.
                for _ in range(1000):
                    _,_,done,cut,info=env.step([.35,0])
                    bonus+=info["progress_bonus"]
                    if done or cut:break
                if info["arrived"]:
                    bonuses.append(bonus);break
            assert bonuses and abs(bonuses[0]-3)<1e-5
        finally:env.close()
    core.save(ROOT/"preflight_checks.json",dict(smoke=dict(zip(ARMS,smoke)),arrival_progress=bonuses,idle=True,potential=True,boundaries=True))


def rows(root, arm, seed, n, base):
    return [r for i in range(4) for r in read(root/f"{arm}_{seed}"/f"eval_{base}_{n}_{i}.json")]


def run_stage(arms, seeds, end):
    setup()
    stage = ROOT/("stage1" if end==300000 else "stage2")
    with ProcessPoolExecutor(min(6,len(arms)*len(seeds))) as tp, ProcessPoolExecutor(5) as ep:
        jobs=[tp.submit(run_train,(a,s,end,stage)) for a in arms for s in seeds]
        # Each training worker writes checkpoints atomically. Evaluate completed arms.
        ev=[]
        lookup=dict(zip(jobs,[(a,s) for a in arms for s in seeds]))
        for f in as_completed(jobs):
            f.result()
            a,s=lookup[f]
            for n in MARKS:
                if n>end:break
                ev.extend(ep.submit(eval_stage,(stage,(a,s,n,i,SCREEN))) for i in range(4))
        for f in as_completed(ev):f.result()
    summary={}
    for a in arms:
        for s in seeds:
            if end==600000 and s==11:
                for n in MARKS[:6]:
                    one=j.torch.load(ROOT/"stage1"/f"{a}_11"/f"policy_{n}.pt",weights_only=True)["policy"]
                    two=j.torch.load(stage/f"{a}_11"/f"policy_{n}.pt",weights_only=True)["policy"]
                    assert all(j.torch.equal(x,two[k]) for k,x in one.items())
            for n in MARKS:
                if n<=end:summary[f"{a}_{s}_{n}"]=aggregate(rows(stage,a,s,n,SCREEN))
    core.save(stage/"screen.json",summary)
    if end==300000:
        ranking=[]
        for a in arms:
            series=[summary[f"{a}_11_{n}"] for n in MARKS[:6]]
            if series[-1]["outcomes"].get("timeout",0)>15 or max(r["outcomes"].get("arrival",0) for r in series)<25:continue
            late=series[-3:]
            score=tuple(sum(sign*r["outcomes"].get(k,0) for r in late) for k,sign in [('arrival',1),('collision',-1),('timeout',-1)])+(sum(r["exit13"]+r["sixteen"]["main"] for r in late),)
            ranking.append((score,a))
        winners=[max(ranking)[1]] if ranking else []
        core.save(ROOT/"stage1_summary.json",dict(promoted=winners,ranking=ranking,screen=summary))
        return winners
    best={}
    for a in arms:
        def rank(n):
            c=Counter()
            for s in seeds:c.update(summary[f"{a}_{s}_{n}"]["outcomes"])
            return c["arrival"],-c["collision"],-n
        best[a]=max(MARKS,key=rank)
    with ProcessPoolExecutor(10) as ep:
        ev=[(stage,(a,s,n,i,b)) for a in arms for s in seeds for n in sorted({best[a],end}) for i in range(4) for b in [PAIRED,FRESH]]
        list(ep.map(eval_stage,ev))
    core.save(ROOT/"selection.json",best)
    return best


def eval_stage(args):
    stage,job=args
    setup()
    core.ROOT=stage
    MixEnv.offset=job[2]
    core.evaluate(job)


def report():
    """Regenerate tables from archived evaluations and paired driving records."""
    assert read(ROOT/"protocol.json")["ppo"]["lr"] == .0001
    doc=Path(__file__).resolve().parents[1]/"docs/experiments/jys3-reward-mix-v1.md"
    heading="## 정량 결과"
    lines=[heading, ""]
    stage1=read(ROOT/"stage1_summary.json")
    lines.extend(["### 1단계 선별", "", "주행 결과는 체크포인트마다 고교통 64회다. 마지막 열은 직전 5만 학습 스텝의 스텝당 평균 보상이며 순서는 전진·도착·분기·충돌·시간초과·정지 벌점·잠재함수다.", "",
                  "|조건|스텝|완주|충돌|끝정지|시간초과|13번 /4|16번 /2|학습 보상 평균|","|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for a in ARMS:
        updates=[json.loads(l) for l in (ROOT/"stage1"/f"{a}_11"/"train_updates.jsonl").read_bytes().splitlines()]
        for n in MARKS[:6]:
            r=stage1["screen"][f"{a}_11_{n}"]
            c=Counter(r["outcomes"])
            block=[u for u in updates if n-50000<u["step"]<=n]
            mean=", ".join(f"{sum(u['rewards'][k] for u in block)/50000:.5f}" for k in KEYS)
            lines.append(f"|{a}|{n:,}|{c['arrival']}|{c['collision']}|{c['stall']}|{c['timeout']}|{r['exit13']}|{r['sixteen']['main']}|{mean}|")
    lines.append("\n승격 조건은 "+(", ".join(stage1["promoted"]) or "없음")+"이다. [판정 점수](../../results/jys3_ppo_comparison/pure_f/reward_mix_v1/stage1_summary.json)")
    if (ROOT/"selection.json").exists():
        selected=read(ROOT/"selection.json")
        stage=ROOT/"stage2"
        lines.extend(["","### 2단계 중간 평가와 독립 검증","","각 시드 64회, 합산 192회. 지표는 건수이며 기준 미분류 종료는 기타에 포함한다.","",
                      "|조건|평가 시드군|스텝|학습 시드|완주|충돌|끝정지|일반 시간초과|기타|13번|16번|","|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
        transition=[]
        base_root=ROOT.parent/"terminal_ratio"/"long_600k_4"
        base_best=read(base_root/"selection.json")
        for a,best in selected.items():
            for base in (SCREEN,PAIRED,FRESH):
                for n in (MARKS if base==SCREEN else sorted({best,600000})):
                    groups=[rows(stage,a,s,n,base) for s in (11,22,33)]
                    for label,group in zip((11,22,33,"합산"),groups+[sum(groups,[])]):
                        stat=aggregate(group)
                        c=Counter(stat["outcomes"])
                        lines.append(f"|{a}|{base}|{n:,}|{label}|{c['arrival']}|{c['collision']}|{c['stall']}|{c['timeout']}|{c['other']}|{stat['exit13']}|{stat['sixteen']['main']}|")
            for baseline in ("F1","collision5"):
                for tag,newstep,oldstep in [("선정",best,base_best[baseline]),("최종",600000,600000)]:
                    counts=Counter()
                    for s in (11,22,33):
                        old=rows(base_root,baseline,s,oldstep,PAIRED)
                        new=rows(stage,a,s,newstep,PAIRED)
                        assert len(old)==len(new)==64
                        for x,y in zip(old,new):
                            assert all(x[k]==y[k] for k in ("route","start","case","traffic_seed"))
                            counts[x["outcome"],y["outcome"]]+=1
                    for origin in ("arrival","collision","stall","timeout","other"):
                        dest=[counts[origin,k] for k in ("arrival","collision","stall","timeout","other")]
                        if sum(dest):
                            transition.append(f"|{baseline} → {a}|{tag}|{origin}|"+ "|".join(map(str,dest))+f"|{dest[0]/sum(dest):.1%}|")
        lines+=["","### 동일 조건 전환표","","선정 모델끼리, 최종 모델끼리 각각 같은 출발을 비교했다. 마지막 열은 기존 완주 행에서는 완주 보존율, 실패 행에서는 완주 전환율이다.","","|비교|시점|기존 종료|완주|충돌|끝정지|시간초과|기타|완주 비율|","|---|---|---|---:|---:|---:|---:|---:|---:|",*transition]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(selected),1,squeeze=False,figsize=(8,3.6*len(selected)),layout="constrained")
        screen=read(stage/"screen.json")
        for ax,(arm,best) in zip(axes[:,0],selected.items()):
            for key in ("arrival","collision","stall","timeout"):
                curve=[sum(screen[f"{arm}_{s}_{n}"]["outcomes"].get(key,0) for s in (11,22,33))*100/192 for n in MARKS]
                ax.plot([n//1000 for n in MARKS],curve,marker=".",label=key)
            ax.axvline(best/1000,color="grey",linestyle=":")
            ax.set(title=arm,xlabel="Training steps (thousands)",ylabel="Episodes (%)",ylim=(0,100))
            ax.legend()
        fig.savefig(ROOT/"stage2_screen.png",dpi=150)
        plt.close(fig)
        lines+=["","![3개 시드의 중간 평가 곡선](../../results/jys3_ppo_comparison/pure_f/reward_mix_v1/stage2_screen.png)"]
    doc.write_text(doc.read_text(encoding="utf-8-sig").split(heading)[0].rstrip()+"\n\n"+"\n".join(lines)+"\n",encoding="utf8")

def main():
    parser=argparse.ArgumentParser()
    for flag in ("check","stage1","stage2","auto","report"):
        parser.add_argument("--"+flag,action="store_true")
    parser.add_argument("--arms",nargs="+",choices=ARMS)
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.report:
        report()
        return
    if args.auto or args.check:check()
    if args.stage1 or args.auto:
        assert (ROOT/"preflight_checks.json").exists()
        selected=run_stage(ARMS,[11],300000)
        if args.auto and selected:run_stage(selected,[11,22,33],600000)
    if args.stage2:
        selected=read(ROOT/"stage1_summary.json")["promoted"]
        assert args.arms and set(args.arms)<=set(selected)
        run_stage(args.arms,[11,22,33],600000)


if __name__=="__main__":
    main()
