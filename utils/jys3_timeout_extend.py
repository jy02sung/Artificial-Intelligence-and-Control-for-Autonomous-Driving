"""Continue near-goal or still-moving evaluation timeouts by another 100 simulated seconds."""
import csv
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path
from utils.jys3_timeout_distances import DEST,ROOT,FactorEnv,j,agent,TimeoutDistances

def replay(r):
    seed=int(r["seed"])
    model=agent(seed)
    model.policy.load_state_dict(j.torch.load(ROOT/f"{r['arm']}_{seed}"/f"policy_{r['step']}.pt",weights_only=True)["policy"])
    model.policy.eval()
    with TemporaryDirectory() as tmp:
        env=FactorEnv(r["arm"],Path(tmp),seed)
        try:
            name=f"route_{int(r['route']):02d}"
            env.sim.max_steps=1500
            env.ego_depart_lane=str(min(int(r["case"])%2,env.counts[env.routes[name][0]]-1))
            obs,_=env.reset(seed=int(r["traffic_seed"]),options={"route":name})
            meter=TimeoutDistances()
            for t in range(1,1501):
                obs,_,done,cut,info=env.step(model.predict(obs,deterministic=True))
                if t<=1000:
                    metrics=meter.observe(env,t,info)
                    if t==1000:
                        assert all(abs(metrics[key]-float(r[key]))<.0011 for key in ("remaining_m","last30_m"))
                if done or cut:break
            assert t>1000
            out="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout"
            return dict(**r,extended_steps=t,extended_outcome=out)
        finally:env.close()

if __name__=="__main__":
    with (DEST/"episodes.csv").open(encoding="utf-8-sig") as f:
        selected=[r for r in csv.DictReader(f) if float(r["remaining_m"])<=100 or float(r["last30_m"])>10]
    with ProcessPoolExecutor(4) as pool:rows=list(pool.map(replay,selected))
    with (DEST/"extended_subset.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print([(r["route"],r["extended_outcome"],r["extended_steps"]) for r in rows])
