from . import local_bc_ppo as e
from collections import Counter
import sys,json
from pathlib import Path
def main(name,stage='screen'):
    e.setup(e.D/(name+".pt"))
    rows=json.loads((e.D/(name+"_"+stage+".json")).read_bytes())["rows"]
    cases=[]
    routes=("route_10","route_05","route_02","route_04","route_03","route_13","route_15") if stage=="dev" else ("route_01","route_02","route_05","route_13","route_14","route_15","route_07")
    for rt in routes:
        bad=next((r for r in rows if r["route"]==rt and r["outcome"]=="timeout"),None)
        if bad:cases.append(bad)
    results=[]
    for row in cases:
        obs,_=e.env.reset(seed=row["seed"],options={"route":row["route"]});hist=Counter();last=[]
        while True:
            act=e.agent.predict(obs,True);edge=e.v.getRoadID("ego");lane=e.v.getLaneID("ego")
            hist[(lane,int(act[1]))]+=1
            if e.env.step_count>=2995:
                pos=float(e.v.getLanePosition("ego"))
                last.append([lane,pos,float(e.v.getSpeed("ego")),act.tolist(),float(obs[9])*120,float(e.m.traci.lane.getLength(lane))-pos])
            obs,_,done,cut,info=e.env.step(act)
            if done or cut:break
        assert cut;results.append(dict(route=row["route"],seed=row["seed"],hist=hist.most_common(3),last=last))
        print(row["route"],hist.most_common(2),last[-1],flush=True)
    e.x.write_json(e.D/(name+("_dev" if stage=="dev" else "")+"_diagnosis.json"),results);e.env.close()
if __name__=="__main__":main(*sys.argv[1:])
