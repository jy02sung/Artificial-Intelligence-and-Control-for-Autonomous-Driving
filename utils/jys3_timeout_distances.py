"""Replay all previously classified F-series timeouts, without retraining."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
from utils.jys3_pure_f import ROOT,FactorEnv,agent,j,TimeoutDistances
DEST=ROOT/"timeout_distances"

def replay(path):
    targets=[r for r in json.loads(path.read_bytes()) if r["outcome"]=="timeout"]
    if not targets:return []
    arm,seed=path.parent.name.split("_")
    _,base,n,rep=path.stem.split("_")
    model=agent(int(seed))
    model.policy.load_state_dict(j.torch.load(path.parent/f"policy_{n}.pt",weights_only=True)["policy"])
    model.policy.eval()
    output=[]
    with TemporaryDirectory() as tmp:
        env=FactorEnv(arm,Path(tmp),int(seed))
        try:
            for r in targets:
                env.ego_depart_lane=str(r["start"][1])
                obs,_=env.reset(seed=r["traffic_seed"],options={"route":f"route_{r['route']:02d}"})
                assert [env.snapshot[i] for i in (0,1,7,-1)]==r["start"]
                meter=TimeoutDistances()
                for t in range(1,1001):
                    obs,_,done,cut,info=env.step(model.predict(obs,deterministic=True))
                    distances=meter.observe(env,t,info)
                    if done or cut:break
                assert t==r["steps"]==1000 and cut and not done and not info["stalled"] and distances["intact"]
                output.append(dict(arm=arm,seed=int(seed),step=int(n),case=int(rep),route=r["route"],traffic_seed=r["traffic_seed"],**distances))
        finally:env.close()
    return output

if __name__=="__main__":
    DEST.mkdir(exist_ok=True)
    paths=sorted(ROOT.glob("F*_*" + "/eval_*.json"))
    rows=[]
    with ProcessPoolExecutor(10) as pool:
        for f in as_completed([pool.submit(replay,p) for p in paths]):
            rows.extend(f.result())
    expected=sum(r["outcome"]=="timeout" for p in paths for r in json.loads(p.read_bytes()))
    assert len(rows)==expected
    rows.sort(key=lambda r:(r["seed"],r["step"],r["arm"],r["route"],r["case"]))
    with (DEST/"episodes.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(len(rows),flush=True)
