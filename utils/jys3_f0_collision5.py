"""Run the paired F0 short trial once the four-arm long trial finishes."""
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from utils import jys3_terminal_ratio as ratio

exp = ratio.exp
ROOT = ratio.ROOT / "f0_collision5_short"
FINAL = ratio.ROOT / "long_600k_4" / "final.json"
PROTOCOL = json.loads((Path(__file__).with_name("jys3_f0_collision5.json")).read_text(encoding="utf-8-sig"))


class NoProgressEnv(ratio.Env):
    def __init__(self, arm, folder, seed):
        super().__init__("collision5", folder, seed)
        self.progress_enabled = False


def setup():
    ratio.setup("collision5")
    exp.FactorEnv = NoProgressEnv
    exp.ROOT = ROOT
    assert (exp.j.ARRIVAL_REWARD, exp.j.COLLISION_REWARD, exp.j.PASS_REWARD, exp.j.TIMEOUT_REWARD) == (5, -5, .25, -5)


def train(seed):
    setup()
    exp.train(("F0", seed, ratio.MARKS))
    def initial(folder):
        return json.loads((folder / "preflight.json").read_bytes())["initial_hash"]
    assert initial(ROOT / f"F0_{seed}") == initial(ratio.ROOT / f"collision5_{seed}")


def evaluate(job):
    setup()
    exp.evaluate(job)


def summarize(seed, step, base):
    pairs = []
    for rep in range(4):
        filename = f"eval_{base}_{step}_{rep}.json"
        old = json.loads((ratio.ROOT / f"collision5_{seed}" / filename).read_bytes())
        new = json.loads((ROOT / f"F0_{seed}" / filename).read_bytes())
        assert len(old) == len(new) == 16
        for a, b in zip(old, new):
            assert all(a[k] == b[k] for k in ("route", "case", "start", "traffic_seed"))
            assert b["bonus"] == 0
            pairs.append((a,b))
    def metrics(rows):
        return dict(outcomes=Counter(r["outcome"] for r in rows),
                    exit13=sum(r["route"] == 13 and "169728835" in r["forks"] for r in rows),
                    main16=sum(r["route"] == 16 and r["start"][1] == 0 and "86198725" in r["forks"] for r in rows))
    return dict(F1=metrics([a for a,b in pairs]), F0=metrics([b for a,b in pairs]),
                transitions=Counter(f"{a['outcome']}->{b['outcome']}" for a,b in pairs))


def main():
    print("Waiting for long-run final verification", flush=True)
    while not FINAL.exists():
        time.sleep(30)
    final = json.loads(FINAL.read_bytes())
    assert all(f"{arm}_{s}_{step}" in final["hold"] for arm in ratio.ARMS for s in ratio.SEEDS
               for step in (65536, final["best"][arm], 600000))
    ROOT.mkdir(exist_ok=False)
    exp.save(ROOT / "protocol.json", PROTOCOL)
    with ProcessPoolExecutor(3) as workers:
        list(workers.map(train, ratio.SEEDS))
    checkpoints = [(32768, ratio.PAIR), (65536, ratio.PAIR), (65536, ratio.FRESH)]
    with ProcessPoolExecutor(8) as workers:
        list(workers.map(evaluate, [("F0", s, n, r, base) for n, base in checkpoints
                                    for s in ratio.SEEDS for r in range(4)]))
    exp.save(ROOT / "summary.json", {f"{s}_{n}_{base}": summarize(s,n,base)
                                      for n,base in checkpoints for s in ratio.SEEDS})
    print("Paired comparison complete", flush=True)


if __name__ == "__main__":
    main()
