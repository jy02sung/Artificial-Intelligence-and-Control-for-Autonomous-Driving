"""Collect balanced, on-expert BC demonstrations. No optimizer or learned policy."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import json
import tempfile
import numpy as np
from utils.jys3 import environment, m, traci, MAX_LANES, OBS_DIM, SCHEMA

OUT = m.ROOT / "results" / SCHEMA


def expert(obs, env):
    speed = float(obs[0]) * env.vmax
    limit = float(obs[31]) * env.vmax
    lane, count = np.rint(obs[32:34] * MAX_LANES).astype(int)
    sign, distance = obs[34:36]
    links = obs[40:].reshape(MAX_LANES, 2)[:count]
    eligible = links[:, int(sign > 0)] if distance >= 0 else links.max(axis=1)
    options = np.flatnonzero(eligible)
    desired = lane if not options.size else min(options, key=lambda i: abs(i-lane))
    direction = int(np.sign(desired - lane))
    target = max(1., limit)
    lead_gap = max(0., float(obs[9]) * env.visibility)
    closing = -float(obs[10]) * env.vmax
    command = 0
    if direction and not obs[38]:
        index = 1 + 4 * (2-direction)
        front_gap = float(obs[index]) * env.visibility
        front_closing = -float(obs[index+1]) * env.vmax
        rear_gap = -float(obs[index+2]) * env.visibility
        rear_closing = float(obs[index+3]) * env.vmax
        room = (front_gap > 3.+0.45*speed+max(0.,front_closing)
                and rear_gap > 3.+0.50*max(0.,speed+rear_closing)+max(0.,rear_closing)
                and obs[37 if direction==1 else 36] == 0.)
        if room:
            command = direction
            if front_gap < lead_gap:
                lead_gap, closing = front_gap, front_closing
        else:
            target = min(target, max(2., np.sqrt(2.8*float(obs[39])*400.)))
    spacing = 2.5 + max(0., 1.05*speed + speed*closing/(2*np.sqrt(2.2*3.2)))
    interaction = (spacing/max(.2,lead_gap))**2 if lead_gap < env.visibility else 0.
    accel = 2.2 * (1.-(speed/target)**4-interaction)
    return np.array([np.clip(accel/(env.max_accel if accel>=0 else env.max_decel),-1,1),command], np.float32)


def probe():
    with tempfile.TemporaryDirectory() as d:
        env=environment("none",11,Path(d))
        try:
            for edge,lane,routes in [('86198728',0,(13,16)),('1063527748',2,(13,14))]:
                samples=[]
                for n in routes:
                    env.ego_depart_lane="0"
                    env.reset(seed=98711,options={"route":f'route_{n:02d}'})
                    lid=f'{edge}_{lane}'
                    traci.vehicle.moveTo(env.ego_id,lid,traci.lane.getLength(lid)-60)
                    obs=env._get_obs()
                    samples.append(obs)
                assert np.flatnonzero(samples[0]!=samples[1]).tolist()==[34]
                print(edge, "navigation", samples[0][34],samples[1][34], "map",samples[0][40:].tolist(),flush=True)
        finally:
            env.close()


def collect(rep):
    summaries=[]
    with tempfile.TemporaryDirectory() as d:
        env=environment("high_real_ratio",11,Path(d))
        try:
            for n in range(1,17):
                rt=f"route_{n:02d}"
                seed=24112400+100*rep+n
                lane=min(rep%2,env.counts[env.routes[rt][0]]-1)
                env.ego_depart_lane=str(lane)
                obs,_=env.reset(seed=seed, options={"route":rt})
                states, actions, applied=[],[],[]
                while True:
                    action=expert(obs,env)
                    before=env.snapshot[:2]
                    states.append(obs)
                    actions.append(action)
                    obs,_,done,cut,info=env.step(action)
                    post=env.snapshot
                    applied.append(int(post[1]-before[1]) if post and post[0]==before[0] else 0)
                    if done or cut:
                        break
                outcome=("collision" if info["collided"] else "arrival" if info["arrived"]
                         else "stall" if info["stalled"] else "timeout" if cut else "other")
                name=f"{rt}_{seed}.npz"
                np.savez_compressed(OUT/name,obs=states,actions=actions,actual_lane_delta=applied)
                summaries.append(dict(file=name,route=rt,seed=seed,lane=lane,outcome=outcome,steps=len(states)))
        finally:
            env.close()
    return summaries


def main():
    probe()
    OUT.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(6) as pool:
        rows=[row for block in pool.map(collect,range(6)) for row in block]
    commands=Counter()
    for row in rows:
        with np.load(OUT/row["file"]) as data:
            obs,act=data["obs"],data["actions"]
            assert obs.shape==(row["steps"],OBS_DIM) and act.shape==(len(obs),2)
            assert all(np.isfinite(a).all() and abs(a).max()<=1 for a in (obs,act))
            assert set(act[:,1])<={-1.,0.,1.}
            commands.update(map(int,act[:,1]))
    manifest=dict(schema=SCHEMA,traffic="high_real_ratio",episodes=rows,commands=dict(commands),
                  samples=sum(commands.values()),outcomes=dict(Counter(r["outcome"] for r in rows)))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(manifest["samples"],manifest["outcomes"],commands)

if __name__=="__main__":
    main()
