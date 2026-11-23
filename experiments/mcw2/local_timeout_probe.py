import json
from collections import deque
from .local_iterate import D,x,a,init
def main():
    rows=json.loads((D/"Z60_dev.json").read_bytes())["rows"]
    cases=[r for r in rows if r["outcome"]=="timeout"][:4]
    init(D/"Z60.pt")
    out=[]
    for r in cases:
        env=x.env;obs,_=env.reset(seed=r["seed"],options={"route":r["route"]})
        trace=[];last=deque(maxlen=8); moving=0
        while True:
            act=x.agent.predict(obs,deterministic=True)
            lane=a.V.getLaneID(env.ego_id)
            if lane.startswith("170059994_"):
                remain=x.m.traci.lane.getLength(lane)-a.V.getLanePosition(env.ego_id)
                f=[env.step_count,lane,round(remain,1),round(a.V.getSpeed(env.ego_id),1),int(act[1]),round(float(act[0]),2),
                   round(float(120*obs[13]),1),round(float(-120*obs[15]),1),round(float(obs[24]-obs[23]),2)]
                if len(trace)<48:trace.append(f)
                last.append(f)
                moving+=int(act[1])!=0
            obs,_,done,cut,info=env.step(act)
            if done or cut:break
        assert cut and env.step_count==r["steps"]
        out.append(dict(route=r["route"],seed=r["seed"],entry=trace,end=list(last),moves=moving,last_lane=lane))
    x.write_json(D/"Z60_timeouts_probe.json",out);x.env.close()
if __name__=="__main__":main()
