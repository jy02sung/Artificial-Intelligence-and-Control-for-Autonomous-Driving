"""Local paired road-choice test for frozen pilot policies on the shared approach."""
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path
from utils.jys3_pure_f import ROOT,FactorEnv,agent,j,save,ARMS

def drive(arm):
    model=agent(11)
    model.policy.load_state_dict(j.torch.load(ROOT/f"{arm}_11"/"policy_32768.pt",weights_only=True)["policy"])
    model.policy.eval()
    rows=[]
    with TemporaryDirectory() as tmp:
        e=FactorEnv(arm,Path(tmp),11)
        try:
            for seed in (38110011,38110111):
                for lane in (0,1,2):
                    for route in (13,14):
                        e.reset(seed=seed,options={"route":f"route_{route:02d}"})
                        edge="170059994"
                        j.traci.vehicle.moveTo(e.ego_id,f"{edge}_{lane}",1.)
                        j.traci.vehicle.setPreviousSpeed(e.ego_id,12.)
                        obs=e.observe(e.sim._get_obs())
                        e.sim.last_external=e.snapshot[3],edge
                        e.start=e.high=e.coordinate()
                        start=obs.tolist()
                        prep=None
                        passed=False
                        moves=0
                        for t in range(1,1001):
                            s=e.snapshot
                            if s[0]==edge:prep=s[1]
                            act=model.predict(obs,deterministic=True)
                            obs,_,done,cut,info=e.step(act)
                            if e.snapshot and e.snapshot[0]==s[0] and e.snapshot[1]!=s[1]:moves+=1
                            if info["fork_passed"] and e.last_external[1] in ("169728835","797789364"):passed=True
                            if done or cut:break
                        rows.append(dict(arm=arm,seed=seed,lane=lane,route=route,initial=start,prep=prep,passed=passed,moves=moves,
                            outcome="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout"))
        finally:e.close()
    save(ROOT/f"local_{arm}.json",rows)
if __name__=="__main__":
    with ProcessPoolExecutor(4) as pool:list(pool.map(drive,ARMS))
