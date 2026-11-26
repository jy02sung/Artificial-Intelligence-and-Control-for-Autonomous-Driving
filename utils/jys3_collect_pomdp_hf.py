"""Collect balanced deterministic dagger5 episodes, one atomic Parquet shard per episode."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["IC_LIBSUMO"] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from utils import jys3_final_relu_dagger as run
from env.toegyewon_crossings import CROSSINGS

DEST = run.ROOT.parent / "jys3_dagger5_hf_pilot"
MODEL = run.checkpoint(5)
I, F, B, S = pa.int32(), pa.float32(), pa.bool_(), pa.string()
def struct(**fields):
    return pa.struct(list(fields.items()))
vehicle = struct(id=S,x=F,y=F,speed=F,lane=I,lane_id=S,road_id=S,lane_pos=F)
state = struct(ego_alive=B,ego_x=F,ego_y=F,ego_speed=F,ego_lane=I,ego_lane_id=S,
               ego_road_id=S,ego_lane_pos=F,vehicles=pa.list_(vehicle))
detected = struct(id=S,x=F,y=F,rel_pos=F,speed=F,lane_offset=I)
action_fields = struct(accel_raw=F,lane_change_raw=F,lane_change=I,lane_change_applied=B)
event_fields = struct(collided=B,arrived=B,lane_before=I,lane_after=I)
schema = pa.schema(dict(episode=I,t=I,route=S,seed=I,outcome=S,observation=pa.list_(F),
    action_raw=pa.list_(F),action=action_fields,reward=F,terminated=B,truncated=B,
    state=state,detected_vehicles=pa.list_(detected),event=event_fields,
    next_observation=pa.list_(F),next_state=state))
def checked(obj):
    if isinstance(obj, float):
        if not np.isfinite(obj):
            raise ValueError("nonfinite scalar")
        return round(obj, 3)
    if isinstance(obj, dict):
        return {k:checked(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [checked(v) for v in obj]
    return obj
def vector(a,n):
    assert len(a)==n and np.isfinite(a).all()
    return a.tolist()
def shard(ep):
    torch.set_num_threads(1)
    model=run.network(MODEL)
    route=f"route_{ep%16+1:02}"
    seed=1706000000+ep
    path=DEST / f"{route}-{seed}.parquet"
    if not path.exists():
        rows=[]
        with run.p.roll.baseline.simulator() as env, torch.inference_mode():
            assert env.sim.physical_forks and env.sim.occlusion and not env.sim.lc_safety_enabled
            assert len(CROSSINGS)==20 and np.isclose(env.sim.step_length,.2)
            env.ego_depart_lane=str(min(ep//16%2,env.counts[env.routes[route][0]]-1))
            obs,_=env.reset(seed=seed,options={"route":route})
            for t in range(1000):
                assert obs.shape==(52,) and not obs[50:].any()
                current=env.get_privileged_state()
                assert current["ego_alive"] and current["vehicles"]
                visible=env.get_detected_vehicles()
                a=model.det_action(torch.from_numpy(obs[:50]).unsqueeze(0))[0].numpy()
                nxt,r,d,c,info=env.step(a)
                assert np.isfinite(r)
                end=d or c
                rows.append(dict(episode=ep,t=t,route=route,seed=seed,observation=vector(obs[:50],50),
                    action_raw=vector(a,2),action={k:info[k] for k in action_fields.names},
                    reward=r,terminated=d,truncated=c,state=checked(current),detected_vehicles=checked(visible),
                    event={k:info[k] for k in event_fields.names},
                    next_observation=vector(nxt[:50],50) if end else None,
                    next_state=checked(env.get_privileged_state()) if end else None))
                obs=nxt
                if end: break
            assert end and info["outcome"] in run.OUTCOMES
            for row in rows: row["outcome"]=info["outcome"]
        tmp=path.with_suffix(".partial")
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),tmp,compression="zstd",compression_level=9)
        validate(tmp,ep,route,seed)
        assert tmp.stat().st_size<200000000
        tmp.replace(path)
    return validate(path,ep,route,seed)
def validate(path,ep,route,seed):
    table=pq.read_table(path)
    assert table.schema.equals(schema)
    rows=table.to_pylist()
    for t,r in enumerate(rows):
        last=t==len(rows)-1
        assert (r["episode"],r["t"],r["route"],r["seed"])==(ep,t,route,seed)
        assert all(r[k] is not None for k in schema.names if not k.startswith("next_"))
        assert len(r["observation"])==50 and len(r["action_raw"])==2
        assert all(v is not None for v in r["action"].values())
        assert all(v is not None for v in r["state"].values())
        assert (r["terminated"] or r["truncated"])==last
        assert (r["next_state"] is not None)==(r["next_observation"] is not None)==last
        assert r["outcome"]==rows[-1]["outcome"]
    return dict(episode=ep,route=route,seed=seed,steps=len(rows),outcome=rows[-1]["outcome"],bytes=path.stat().st_size)
if __name__=="__main__":
    DEST.mkdir(exist_ok=True)
    start=time.monotonic()
    with ProcessPoolExecutor(2) as pool:
        rows=list(pool.map(shard,range(16)))
    report=dict(rows=rows,seconds=time.monotonic()-start,checkpoint_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest())
    (DEST/"pilot.json").write_text(json.dumps(report,indent=2),encoding="utf8")
