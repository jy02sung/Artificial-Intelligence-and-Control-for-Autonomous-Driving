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


def drive(env, job):
    group, route, seed, start_lane = job
    path = f"route_{route:02d}"
    env.ego_depart_lane = str(min(start_lane, env.counts[env.routes[path][0]] - 1))
    x, _ = env.reset(seed=seed, options={"route": path})
    assert env.sim.physical_forks and not env.sim.lc_safety_enabled
    events = []
    active = None
    previous_key = None
    for step in range(1, 1001):
        action, wait, direction, reason = recovery.control(x[:50], env, "queue_tail")
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
        x, _, done, cut, info = env.step(action)
        post = env.snapshot
        if active is not None and post and post[0] == edge and post[1] == lane + direction:
            active.update(move=step, move_speed=speed)
        if done or cut:
            assert info["outcome"] != "arrival" or info["route_intact"]
            return dict(group=group, route=route, seed=seed, lane=int(env.ego_depart_lane),
                        steps=step, outcome=info["outcome"], events=events)
    raise AssertionError("missing terminal outcome")


def shard(jobs):
    with roll.baseline.simulator() as env:
        return [drive(env, job) for job in jobs]


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


if __name__ == "__main__":
    main()
