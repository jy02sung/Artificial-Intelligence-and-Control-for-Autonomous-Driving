"""Paired 512-drive evaluation of the revised rolling-merge speed targets."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import jys3_verify_rolling_drives as panel

ROOT = panel.ROOT.parent / "jys3_rolling_margin_eval"
CONNECTOR = "170059994"
MAINLINE = "844484877"


def slow_steps(row, edge):
    return max((e["longest_slow"] for e in row["events"] if e["edge"] == edge), default=0)


def run_batch(jobs):
    with panel.roll.baseline.simulator() as env:
        return [panel.drive(env, job, revised=True) for job in jobs]


def main():
    ROOT.mkdir(exist_ok=True)
    before = {r["seed"]:r for r in panel.read(panel.ROOT / "episodes.json")}
    jobs = panel.cases()
    assert len(jobs) == len(before) == 512
    focus = {}
    for seed, row in before.items():
        if slow_steps(row, CONNECTOR) >= 50:
            focus[seed] = CONNECTOR
        elif row["outcome"] == "stall" and row["terminal"]["edge"] == MAINLINE:
            focus[seed] = MAINLINE
    assert Counter(focus.values()) == {CONNECTOR:6, MAINLINE:3}
    first = [j for j in jobs if j[2] in focus]
    rest = [j for j in jobs if j[2] not in focus]
    rows = []
    with ProcessPoolExecutor(10) as pool:
        for batch in pool.map(run_batch, [[j] for j in first]):
            rows.extend(batch)
        panel.save(ROOT / "focus.json", rows)
        tasks = [pool.submit(run_batch, rest[i:i+8]) for i in range(0, len(rest), 8)]
        for task in as_completed(tasks):
            rows.extend(task.result())
            print(len(rows), dict(Counter(r["outcome"] for r in rows)), flush=True)
    rows.sort(key=lambda r:(r["group"],r["seed"]))
    assert len({r["seed"] for r in rows}) == len(rows) == 512
    for r in rows:
        old = before[r["seed"]]
        assert all(r[k] == old[k] for k in ("group", "route", "lane"))
    panel.save(ROOT / "episodes.json", rows)
    panel.save(ROOT / "summary.json", {"all":panel.summarize(rows),
               **{g:panel.summarize([r for r in rows if r["group"] == g])
                  for g in sorted({r["group"] for r in rows})}})


if __name__ == "__main__":
    main()
