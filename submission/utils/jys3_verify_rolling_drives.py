"""Replay the queue-tail expert after the overpass sensor fix, without retrying failures."""
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("IC_LIBSUMO", "1")
import numpy as np
from utils.jys3_gap_bc import SETS
from utils import jys3_rolling_recovery as recovery

roll = recovery.roll
read, save = roll.baseline.v3.read, roll.baseline.v3.save
ROOT = recovery.ROOT.parent / "jys3_rolling_visibility_validation"


def cases():
    jobs = []
    for name, folder in SETS.items():
        rows = read(folder / "train.json")
        assert Counter(r["route"] for r in rows) == {i: 10 for i in range(1, 17)}
        for r in rows:
            file = folder / "samples" / r["file"]
            assert r["outcome"] == "arrival" and hashlib.sha256(file.read_bytes()).hexdigest() == r["sha256"]
            with np.load(file, allow_pickle=False) as sample:
                assert sample["obs"].shape == (r["steps"], 50)
                assert sample["actions"].shape == (r["steps"], 2)
            jobs.append((name, r["route"], r["seed"], r["lane"]))
    for name, base, reps in (("matched", roll.BASE, 8), ("fresh", 98643000, 4)):
        jobs.extend((name, route, base + 100 * rep + route, rep % 2)
                    for rep in range(reps) for route in range(1, 17))
    assert len(set(j[2] for j in jobs)) == len(jobs)
    return jobs


def drive(env, job, *, original=False, revised=False, short_reserve=False):
    group, route, seed, start_lane = job
    path = f"route_{route:02d}"
    env.ego_depart_lane = str(min(start_lane, env.counts[env.routes[path][0]] - 1))
    x, _ = env.reset(seed=seed, options={"route": path})
    assert env.sim.physical_forks and not env.sim.lc_safety_enabled
    events = []
    active = None
    previous_key = None
    for step in range(1, 1001):
        if original:
            action, wait, direction, reason = roll._decision(x[:50], env, "old")
        else:
            action, wait, direction, reason = recovery.control(x[:50], env, "queue_tail", revised=revised, short_reserve=short_reserve)
        edge, lane = env.snapshot[:2]
        key = (edge, lane, direction) if direction and not x[38] and not edge.startswith(":") else None
        if key != previous_key:
            active = None
            if key:
                active = dict(edge=edge, lane=lane, direction=direction, start=step, first_open=None,
                              first_command=None, move=None, wait_steps=0, slow=0, longest_slow=0,
                              accel=Counter(), reasons=Counter(), first_speed=float(env.snapshot[7]))
                events.append(active)
        previous_key = key
        if active is not None:
            speed = float(env.snapshot[7])
            i = 1 + 4 * (2 - direction)
            front, rear = float(x[i]) * env.visibility, -float(x[i+2]) * env.visibility
            closing, rear_closing = -float(x[i+1]) * env.vmax, float(x[i+3]) * env.vmax
            sensed_speed = float(x[0]) * env.vmax
            opened = (not x[37 if direction == 1 else 36]
                      and front > 3 + .45 * sensed_speed + max(0., closing)
                      and rear > 3 + .50 * max(0., sensed_speed + rear_closing) + max(0., rear_closing))
            if opened and active["first_open"] is None:
                active["first_open"] = step
            if action[1] == direction and active["first_command"] is None:
                active.update(first_command=step, front_at_command=front, rear_at_command=rear,
                              remaining_at_command=float(env.snapshot[-1]))
            if wait:
                active["wait_steps"] += 1
                active["accel"][str(int(np.sign(action[0])))] += 1
                active["reasons"][reason] += 1
            active["slow"] = active["slow"] + 1 if wait and speed < 1 else 0
            active["longest_slow"] = max(active["longest_slow"], active["slow"])
        before, snapshot = x, env.snapshot
        x, _, done, cut, info = env.step(action)
        post = env.snapshot
        if active is not None and post and post[0] == edge and post[1] == lane + direction:
            active.update(move=step, move_speed=speed)
        if done or cut:
            assert info["outcome"] != "arrival" or info["route_intact"]
            row = dict(group=group, route=route, seed=seed, lane=int(env.ego_depart_lane),
                       steps=step, outcome=info["outcome"], events=events)
            if info["outcome"] != "arrival":
                terminal = dict(edge=edge, lane=lane, speed=snapshot[7], remaining=snapshot[-1],
                                direction=direction, reason=reason, action=action.tolist(),
                                own_front_m=float(before[9])*env.visibility)
                if direction:
                    i = 1 + 4 * (2 - direction)
                    terminal.update(front_m=float(before[i])*env.visibility,
                                    rear_m=-float(before[i+2])*env.visibility,
                                    overlap=bool(before[37 if direction == 1 else 36]))
                row["terminal"] = terminal
            return row
    raise AssertionError("missing terminal outcome")


def shard(jobs):
    with roll.baseline.simulator() as env:
        return [drive(env, job) for job in jobs]


def compare_long_waits(rows):
    comparison = []
    with roll.baseline.simulator() as env:
        for r in rows:
            slow = max((e["longest_slow"] for e in r["events"]), default=0)
            if slow < 50:
                continue
            old = drive(env, (r["group"], r["route"], r["seed"], r["lane"]), original=True)
            comparison.append(old)
    save(ROOT / "original_long_waits.json", comparison)


def summarize(rows):
    events = [e for r in rows for e in r["events"]]
    waited = [e for e in events if e["wait_steps"]]
    opened = [e for e in waited if e["first_open"] is not None]
    moved = [e for e in opened if e["move"] is not None]
    speeds = [e["move_speed"] for e in moved]
    slow = [max((e["longest_slow"] for e in r["events"]), default=0) for r in rows]
    return dict(
        episodes=len(rows), outcomes=dict(Counter(r["outcome"] for r in rows)),
        failures=[(r["route"], r["seed"], r["outcome"]) for r in rows if r["outcome"] != "arrival"],
        waits=len(waited), openings_after_wait=len(opened),
        immediate_commands=sum(e["first_open"] == e["first_command"] for e in opened),
        immediate_moves=sum(e["first_open"] == e["move"] for e in opened),
        confirmed_moves=len(moved), moving_at_least_1mps=sum(v >= 1 for v in speeds),
        median_move_speed_mps=float(np.median(speeds)) if speeds else None,
        wait_with_acceleration=sum(e["accel"].get("1", 0) > 0 for e in waited),
        wait_with_braking=sum(e["accel"].get("-1", 0) > 0 for e in waited),
        slow_5s=sum(v >= 25 for v in slow), slow_10s=sum(v >= 50 for v in slow),
        max_slow_s=round(max(slow) * roll.DT, 1),
        by_route={i:dict(Counter(r["outcome"] for r in rows if r["route"] == i))
                  for i in range(1, 17)})


def main():
    jobs = cases()
    ROOT.mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(10) as pool:
        tasks = [pool.submit(shard, jobs[i:i+8]) for i in range(0, len(jobs), 8)]
        for task in as_completed(tasks):
            chunk = task.result()
            rows.extend(chunk)
            print(len(rows), "/", len(jobs), dict(Counter(r["outcome"] for r in rows)), flush=True)
    assert len(rows) == len(jobs)
    save(ROOT / "episodes.json", sorted(rows, key=lambda r: (r["group"], r["seed"])))
    save(ROOT / "summary.json", {name:summarize([r for r in rows if r["group"] == name])
                                 for name in sorted({r["group"] for r in rows})}
         | {"all":summarize(rows)})
    compare_long_waits(rows)


if __name__ == "__main__":
    main()
