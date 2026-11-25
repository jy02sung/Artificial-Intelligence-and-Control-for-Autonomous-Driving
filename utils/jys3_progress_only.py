"""Paired pure PPO: redistribute arrival reward into new route progress."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json

from utils import jys3_terminal_ratio as ratio

exp = ratio.exp
ROOT = ratio.ROOT / "progress_only_7_10"
CAPS = {"progress7": 7, "progress10": 10}
PANELS = [(32768, ratio.PAIR), (65536, ratio.PAIR), (65536, ratio.FRESH)]


class ProgressEnv(ratio.Env):
    def __init__(self, arm, folder, seed):
        super().__init__("collision5", folder, seed)
        self.progress_cap = CAPS[arm]


def setup():
    ratio.setup("collision5")
    exp.j.ARRIVAL_REWARD = 0
    exp.FactorEnv = ProgressEnv
    exp.ROOT = ROOT
    assert (exp.j.COLLISION_REWARD, exp.j.TIMEOUT_REWARD, exp.j.PASS_REWARD) == (-5, -5, .25)


def train(job):
    arm, seed = job
    setup()
    exp.train((arm, seed, ratio.MARKS))
    def initial(folder):
        return json.loads((folder / "preflight.json").read_bytes())["initial_hash"]
    assert initial(ROOT / f"{arm}_{seed}") == initial(ratio.ROOT / f"collision5_{seed}")


def evaluate(job):
    setup()
    exp.evaluate(job)


def rows(folder, step, base):
    return [r for i in range(4) for r in json.loads((folder / f"eval_{base}_{step}_{i}.json").read_bytes())]


def report():
    lines = ["# 완주 보너스를 전진 보상으로 이전한 순수 PPO 비교", "",
             "완주 보너스를 경로 전진으로 나누어 지급하면 시간초과가 줄고 완주가 늘어나는지 확인했다. F1은 전진 +2와 도착 +5, 두 실험 조건은 도착 0과 전진 +7 또는 +10이다. 분기 +0.25, 충돌·시간초과·정지 −5는 공통이며 속도 보상은 0이다.",
             "", "각 시드는 동일 출발 64회, 합산은 192회다. 시간초과율은 일반 시간초과와 정지 종료의 합이다.",
             "", "|스텝·평가|조건|시드|완주율|충돌률|시간초과율|일반 / 정지 건수|", "|---|---|---|---:|---:|---:|---:|"]
    for step, base in PANELS:
        for arm in ["F1", *CAPS]:
            counts = []
            for seed in ratio.SEEDS:
                old = rows(ratio.ROOT / f"collision5_{seed}", step, base)
                new = old if arm == "F1" else rows(ROOT / f"{arm}_{seed}", step, base)
                keys = ("route", "case", "traffic_seed", "start")
                assert len(old) == len(new) == 64 and all(all(a[k]==b[k] for k in keys) for a,b in zip(old,new))
                counts.append(Counter(r["outcome"] for r in new))
            for seed,c in zip([*ratio.SEEDS,"합산"], [*counts,sum(counts,Counter())]):
                total = sum(c.values())
                rates = [f"{v/total:.1%} ({v})" for v in (c["arrival"],c["collision"],c["timeout"]+c["stall"])]
                panel = "독립" if base == ratio.FRESH else "중간"
                lines.append(f"|{step:,} {panel}|{arm}|{seed}|"+ "|".join(rates)+f"|{c['timeout']} / {c['stall']}|")
    (ROOT / "README.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def main():
    ROOT.mkdir(exist_ok=False)
    exp.save(ROOT/"protocol.json",dict(caps=CAPS,arrival=0,collision=-5,timeout=-5,fork=.25,speed=0,seeds=ratio.SEEDS,panels=PANELS,marks=ratio.MARKS,baseline="terminal_ratio/collision5"))
    jobs = [(arm,s) for s in ratio.SEEDS for arm in CAPS]
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(train,jobs))
    with ProcessPoolExecutor(10) as pool:
        for step,base in PANELS:
            list(pool.map(evaluate,[(arm,s,step,r,base) for arm,s in jobs for r in range(4)]))
    report()


if __name__ == "__main__":
    main()
