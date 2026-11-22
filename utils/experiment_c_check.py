"""Fixed-action SUMO and PPO preflight for the four C conditions."""
import json,hashlib
import xml.etree.ElementTree as ET
import numpy as np
from utils.experiment_b import agent_for,digest,save,traci,STAGES,OUT as B
from utils.experiment_c_env import OUT,SEEDS,ARMS,HOLD,SELECT,make_c,constraint
from utils.experiment_c_timeout_check import run as check_timeout

def place(env,route,edge,lane,rem):
    env.ego_depart_lane="0"
    env.reset(seed=765412,options={"route":route})
    lid=f"{edge}_{lane}"
    traci.vehicle.moveTo(env.ego_id,lid,traci.lane.getLength(lid)-rem)
    return env._get_obs()
def drive(env,cmd):
    obs,r,done,cut,info=env.step(np.array([.5,cmd],np.float32))
    return [info["c_prep"],info["c_pass"],bool(done),bool(cut)]

def run():
    data=dict(timeout=check_timeout())
    for arm in ("C2","C3"):
        e=make_c("none",arm,11,OUT/"preflight_configs")
        try:
            road="86198728"
            trials=[]
            for lane,commands,expected in ((0,[1,-1],[0,0]),(1,[-1,1,-1],[.3,0,0]),(0,[],[])):
                place(e,"route_13",road,lane,90)
                seq=[drive(e,c) for c in commands]
                assert [x[0] for x in seq]==expected
                for _ in range(50):
                    seq.append(drive(e,0))
                    if e.snapshot[0]=="37927508":break
                assert sum(x[1] for x in seq)==.25 and e.snapshot[0]=="37927508", (seq,e.snapshot,e.events)
                trials.append(seq)
            place(e,"route_13",road,1,90)
            e.max_steps=1
            timeout=drive(e,0)
            assert timeout==[0,0,False,True]
            e.max_steps=3000
            place(e,"route_13",road,0,180)
            v=traci.vehicle
            v.add("obstacle","route_13",typeID="ego_type",departLane="0")
            traci.simulationStep()
            lid=road+"_0"
            pos=v.getLanePosition(e.ego_id)
            v.moveTo("obstacle",lid,pos+18)
            v.setSpeedMode("obstacle",0)
            v.setLaneChangeMode("obstacle",0)
            v.setSpeed("obstacle",0.)
            e._get_obs()
            crash=[]
            for _ in range(30):
                crash.append(drive(e,0))
                if crash[-1][2]:break
            assert crash[-1][2] and e.ego_id in traci.simulation.getCollidingVehiclesIDList() and not sum(x[1] for x in crash)
            place(e,"route_02","544292616",1,20)
            a=constraint(e,e.snapshot)[0]
            trace=[]
            for _ in range(40):
                b=e.snapshot
                tick=drive(e,0)
                trace.append([b[0],b[1],tick])
                if e.snapshot and e.snapshot[0]=="844484877":
                    assert constraint(e,e.snapshot)[0]==a
                    drive(e,0)
                    break
            assert e.snapshot[0]=="844484877" and e.c_best[a]==0 and not any(x[2][0] for x in trace)
            data[arm]=dict(drives=trials,timeout=timeout,collision=crash,boundary=trace)
        finally:e.close()
    for arm in ARMS:
        e=make_c("none",arm,11,OUT/"preflight_configs")
        try:
            inputs=[place(e,r,"86198728",1,80).tolist() for r in ("route_13","route_16")]
            assert [i for i,(a,b) in enumerate(zip(*inputs)) if a!=b]==[34]
            probes=[]
            for dist in (150,25):
                obs=place(e,"route_02","844484877",0,dist)
                physical=float(obs[23])
                assert abs(physical-min(1,dist/e.physical_link_horizon))<1e-5
                probes.append([dist,physical])
            data[arm+"_obs"]=dict(probes=probes,visibility=e.visibility,length=e.navigation.lengths["844484877"],outside400="road too short",paired_inputs=inputs)
        finally:e.close()
    base=json.loads((B/"design.json").read_text())
    for stage in STAGES:
        from utils.experiment_b import m
        cfg=m.prepare_scenario(stage,OUT/"configs")
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in cfg.parent.iterdir() if f.is_file()}
        assert hashes==base["stages"][stage]["hashes"]
        flows=ET.parse(cfg.parent/"highway.rou.xml").findall("flow")
        demand=sum(int(f.get("number"))*3600/(float(f.get("end"))-float(f.get("begin"))) for f in flows)
        assert demand==base["stages"][stage]["demand"]
        data[stage]=dict(config=str(cfg),hashes=hashes,demand=demand)
    for seed in SEEDS:
        hashes=[]
        for arm in ARMS:
            folder=OUT/f"{arm}_{seed}"
            folder.mkdir(exist_ok=True)
            agent=agent_for(seed)
            assert not agent.optimizer.state and agent.num_timesteps==0
            hashes.append(digest(agent))
            agent.save(str(folder/"policy_0.pt"))
        assert len(set(hashes))==1
        data[str(seed)]=hashes
    save(OUT/"design.json",dict(pairs=base["pairs"],ppo=base["ppo"],arms=ARMS,train_seeds=SEEDS,
        select_base=SELECT,holdout_base=HOLD,selection=base["selection"],checks=data))
if __name__=="__main__":
    if list(OUT.glob("*/progress.json")):raise RuntimeError("Training already started")
    run()

