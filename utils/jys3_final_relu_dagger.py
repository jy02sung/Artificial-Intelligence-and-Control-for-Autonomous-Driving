"""Select the final expert, then train and validate its ReLU hybrid imitation policy."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from utils import jys3_rolling_margin_eval as margin

p = margin.panel
ROOT = p.ROOT.parent / "jys3_final_relu_dagger"
DOCS = ROOT.parent.parent / "docs" / "experiments"
FOCUS = (68173405, 94600413, 94601313, 94600014, 94600214)


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


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(10) as pool:
        print(phase1(pool),flush=True)
