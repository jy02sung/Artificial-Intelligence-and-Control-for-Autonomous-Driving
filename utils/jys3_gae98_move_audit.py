"""Validate floor-gap and fork-direction features on every unmodified actual lane change."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_gae98_cutin_replay as replay

C,L = replay.C,replay.L
ROOT = replay.ROOT / "move_audit"


def shard(job):
    h,i = job
    file = f"100000_{h}_{i}.json"
    if (ROOT/file).exists():
        return
    C.torch.set_num_threads(1)
    model = C.final.network(replay.POLICY)
    records = []
    with C.final.p.roll.baseline.simulator() as env,C.torch.inference_mode(),TemporaryDirectory() as tmp:
        transform = C.p.lateral_transform(env)
        xml = Path(tmp)/"changes.xml"
        for row in L.read(replay.SOURCE/"parts"/file):
            env.ego_depart_lane = str(row["lane"])
            obs,_ = C.start_trace(lambda:env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"}),xml)
            assert replay.digest(obs)==row["initial_sha256"]
            candidates = {}
            for tick in range(1,1001):
                x = transform(obs[:50])
                action = model.det_action(C.torch.from_numpy(x).unsqueeze(0))[0].numpy()
                cmd = env.sim._quantize_lane_change(action[1])
                if cmd:
                    edge,lane,path,idx = env.snapshot[:4]
                    dist_delta = None
                    if idx+1 < len(path) and edge == path[idx] and (edge,path[idx+1]) in env.sim.signs:
                        lanes = env.sim.links[edge][path[idx+1]]
                        dist_delta = min(abs(lane+cmd-y) for y in lanes)-min(abs(lane-y) for y in lanes)
                    candidates[tick] = (cmd,C.diag.measurements(x,cmd,env.visibility,env.vmax),dist_delta)
                obs,_,done,cut,info = env.step(action)
                if done or cut:
                    break
            contacts = [C.diag.split_lane(e["lane"]) for e in C.diag.collision_events(env.ego_id)] if info["collided"] else []
            moves = C.finish_trace(env,xml,tick)
            assert (len(moves),info["outcome"],tick)==(row["moves"],row["outcome"],row["steps"])
            sample = []
            for t,a,b in moves:
                cmd,m,delta = candidates[t]
                assert cmd == b[1]-a[1]
                sample.append(dict(seed=row["seed"],route=row["route"],outcome=row["outcome"],tick=t,
                      edge=a[0],lane=a[1],target=b[1],direction=cmd,fork_distance_delta=delta,
                      view=m,danger=C.dangerous(m),cutin=bool(tick-t<3 and b in contacts and m["tight"])))
            assert sum(v["danger"] for v in sample)==row["risky"] and any(v["cutin"] for v in sample)==bool(row["cutin"])
            records.extend(sample)
    L.save(ROOT/file,records)


def main():
    assert replay.mid.ft.sha(replay.POLICY)==replay.mid.ARMS["lambda98"]
    jobs = [(h,i) for h in L.HOLDS for i in range(C.old.PANELS[h][1])]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(shard,jobs))
    rows = [r for h,i in jobs for r in L.read(ROOT/f"100000_{h}_{i}.json")]
    assert len(rows)==1070 and len({r["seed"] for r in rows if r["cutin"]})==31
    print("Verified",len(rows),"actual lane changes",flush=True)


if __name__=="__main__":
    main()
