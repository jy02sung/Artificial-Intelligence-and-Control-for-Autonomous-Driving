"""Diagnostic only: apply a two-tick pause on observed risky lane-change requests."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from utils import jys3_gae98_cutin_replay as replay

C, L = replay.C, replay.L
ROOT = replay.ROOT / "panel"


def run_shard(job):
    panel, rep = job
    filename = f"100000_{panel}_{rep}.json"
    output = ROOT/filename
    if output.exists():
        return
    reference = L.read(replay.SOURCE/"parts"/filename)
    model = C.final.network(replay.POLICY)
    C.torch.set_num_threads(1)
    rows = []
    with C.final.p.roll.baseline.simulator() as env, C.torch.inference_mode():
        transform = C.p.lateral_transform(env)
        for row in reference:
            route = row["route"]
            env.ego_depart_lane = str(row["lane"])
            obs, _ = env.reset(seed=row["seed"],options={"route":f"route_{route:02d}"})
            assert replay.digest(obs) == row["initial_sha256"]
            locked = None
            wait = blocks = changed = 0
            for tick in range(1,1001):
                x = transform(obs[:50])
                action = model.det_action(C.torch.from_numpy(x).unsqueeze(0))[0].numpy().copy()
                cmd = env.sim._quantize_lane_change(action[1])
                edge, lane = env.snapshot[:2]
                key = (edge,lane,cmd)
                if wait == 0:
                    risk = False
                    if cmd and not edge.startswith(":") and not C.invalid(edge,lane,env.counts[edge],cmd,x):
                        view = C.diag.measurements(x,cmd,env.visibility,env.vmax)
                        risk = C.dangerous(view)
                    if not risk:
                        locked = None
                    elif key != locked:
                        locked = key
                        wait = 2
                        blocks += 1
                if wait:
                    changed += int(cmd != 0)
                    action[1] = 0
                    wait -= 1
                obs, _, done, cut, info = env.step(action)
                if done or cut:
                    break
            if not changed:
                assert (info["outcome"],tick) == (row["outcome"],row["steps"])
            rows.append(dict(seed=row["seed"],route=route,steps=tick,original=row["outcome"],
                             original_cutin=row["cutin"],outcome=info["outcome"],blocks=blocks,changed=changed))
    L.save(output,rows)


def main():
    assert replay.mid.ft.sha(replay.POLICY) == replay.mid.ARMS["lambda98"]
    jobs = [(h,i) for h in L.HOLDS for i in range(C.old.PANELS[h][1])]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_shard,jobs))
    rows = [r for h,i in jobs for r in L.read(ROOT/f"100000_{h}_{i}.json")]
    assert len(rows)==len({r["seed"] for r in rows})==576 and sum(r["original_cutin"] for r in rows)==31
    result = dict(counts=Counter(r["outcome"] for r in rows),
                  cutin_cases=Counter(r["outcome"] for r in rows if r["original_cutin"]),
                  modified_episodes=sum(r["changed"]>0 for r in rows))
    L.save(ROOT/"summary.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
