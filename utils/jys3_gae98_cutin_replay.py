"""Replay seed 33 GAE .98 tight cut-ins. Fixed 1/2-tick delay, with and without retry."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_gae_policy_midpoint as mid

L, C = mid.L, mid.C
SOURCE = mid.SOURCE / "lambda98" / "s33"
POLICY = SOURCE / "policy_100000.pt"
ROOT = mid.DEST.parent / "jys3_gae98_cutin_replay"


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def drive(env, model, transform, xml, row, event=None, delay=0, retry=False):
    env.ego_depart_lane = str(row["lane"])
    obs, _ = C.start_trace(lambda: env.reset(seed=row["seed"], options={"route":f"route_{row['route']:02d}"}), xml)
    assert digest(obs) == row["initial_sha256"] and abs(env.sim.step_length - .2) < 1e-6
    recent = deque(maxlen=3)
    prefix = hashlib.sha256()
    counts = Counter()
    retried = False
    release = None
    paid = set()
    for tick in range(1,1001):
        x = transform(obs[:50])
        action = model.det_action(C.torch.from_numpy(x).unsqueeze(0))[0].numpy().copy()
        cmd = env.sim._quantize_lane_change(action[1])
        edge, lane, path, index = env.snapshot[:4]
        if event is None or tick <= event["tick"]:
            prefix.update(obs.tobytes())
            prefix.update(action.tobytes())
        if event is None:
            counts[cmd] += 1
            fork = next((k for k in range(index,len(path)-1) if (path[k],path[k+1]) in env.sim.signs),None)
            recent.append(dict(tick=tick,edge=edge,lane=lane,cmd=cmd,mu=float(action[0]),
                               prefix=prefix.hexdigest(),speed=env.snapshot[7],remaining=env.snapshot[-1],
                               next_fork=fork, next_pair=list(path[fork:fork+2]) if fork is not None else None,
                               obs=x.copy()))
        else:
            start = event["tick"]
            if tick == start:
                assert prefix.hexdigest() == event["prefix"] and cmd == event["direction"]
            if start <= tick < start+delay:
                action[1] = 0
            if tick == start+delay:
                same_lane = [edge,lane] == [event["edge"],event["lane"]]
                release = dict(tick=tick,cmd=cmd,same_lane=same_lane,
                               view=C.diag.measurements(x,event["direction"],env.visibility,env.vmax))
                if retry and same_lane:
                    action[1] = event["direction"]
                    retried = True
        obs, _, done, cut, info = env.step(action)
        paid.update(env.sim.paid_forks)
        if done or cut:
            break
    contacts = C.diag.collision_events(env.ego_id) if info["collided"] else []
    moves = C.finish_trace(env,xml,tick)
    if event is not None:
        next_move = next(([t,a,b] for t,a,b in moves if t >= event["tick"]),None)
        fork = event["next_fork"]
        return dict(outcome=info["outcome"],steps=tick,release=release,retried=retried,first_move=next_move,
                    fork_passed=fork in paid if fork is not None else None,contacts=contacts)
    assert (tick,info["outcome"],len(moves),counts[1],counts[-1]) == (
        row["steps"],"collision",row["moves"],row["left_commands"],row["right_commands"])
    matched = []
    for t, a, b in moves:
        if tick-t >= 3 or b not in [C.diag.split_lane(e["lane"]) for e in contacts]:
            continue
        frame = next(f for f in recent if f["tick"] == t)
        direction = 1 if b[1] > a[1] else -1
        view = C.diag.measurements(frame["obs"],direction,env.visibility,env.vmax)
        if view["tight"]:
            assert [frame["edge"],frame["lane"]] == a and frame["cmd"] == direction
            matched.append(dict(**{k:v for k,v in frame.items() if k!="obs"},direction=direction,
                                view=view,dangerous=C.dangerous(view),target=b,contacts=contacts))
    assert matched, row["seed"]
    return matched[-1]


def batch(rows):
    C.torch.set_num_threads(1)
    model = C.final.network(POLICY)
    with C.final.p.roll.baseline.simulator() as env, C.torch.inference_mode(), TemporaryDirectory() as tmp:
        transform = C.p.lateral_transform(env)
        for row in rows:
            xml = Path(tmp)/"moves.xml"
            event = drive(env,model,transform,xml,row)
            probes = {f"{mode}_{delay}":drive(env,model,transform,xml,row,event,delay,mode=="retry")
                      for mode in ("release","retry") for delay in (1,2)}
            L.save(ROOT/"cases"/f"{row['seed']}.json",dict(original=row,event=event,probes=probes))


def main():
    assert mid.ft.sha(POLICY) == mid.ARMS["lambda98"]
    rows = [r for panel in L.HOLDS for rep in range(C.old.PANELS[panel][1])
            for r in L.read(SOURCE/"parts"/f"100000_{panel}_{rep}.json")]
    cases = [r for r in rows if r["cutin"]]
    assert len({r["seed"] for r in rows}) == len(rows) == 576
    assert len(cases)==31 and all(r["outcome"]=="collision" for r in cases)
    pending = [r for r in cases if not (ROOT/"cases"/f"{r['seed']}.json").exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(batch,[pending[i:i+4] for i in range(0,len(pending),4)]))
    results = [L.read(ROOT/"cases"/f"{r['seed']}.json") for r in cases]
    assert mid.ft.sha(POLICY) == mid.ARMS["lambda98"]
    L.save(ROOT/"summary.json",dict(n=len(results),sha=mid.ARMS["lambda98"],
           danger=sum(r["event"]["dangerous"] for r in results),
           outcomes={k:Counter(r["probes"][k]["outcome"] for r in results) for k in results[0]["probes"]},
           all_moves=sum(r["moves"] for r in rows),risky_moves=sum(r["risky"] for r in rows),
           completed_risky_moves=sum(r["risky"] for r in rows if r["outcome"]=="arrival")))
    print(L.read(ROOT/"summary.json"),flush=True)


if __name__=="__main__":
    main()
