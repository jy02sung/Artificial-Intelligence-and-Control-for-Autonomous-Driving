"""Find a gap that would still be insufficient at constant speed."""
import json
from concurrent.futures import ProcessPoolExecutor
from . import find_gap_example as g
def init():
    g.curve.init_eval(g.MODEL)
    g.replay.V=g.d.m.traci.vehicle
    g.replay.ego=g.d.env.ego_id
def test(row):
    event,_=g.run(row,stronger=True)
    if not event:return
    _,control=g.run(row,event,True)
    f=next((f for f in control if f["step"]==event["change"]),None)
    if f is None:return
    p=next((p for p in f["near"] if p["id"]==event["foe"] and p["edge"]==f["av"]["edge"]),None)
    if not p:return
    gap=g.measure(f["av"],p)[event["side"]]
    if gap>2 or event['gap_end']-gap<3:return
    _,frames=g.run(row,event)
    after=[f for f in frames if f["step"]>event["change"] and f["av"]["edge"]==event["edge"]]
    if len(after)<8 or any(f["av"]["lane"]!=event["lane"] or f["cmd"]!=0 for f in after[:8]):return
    return dict(model=g.MODEL,row=row,event=event,frames=frames,control=control)
if __name__=="__main__":
    rows=json.loads((g.curve.OUT/(g.MODEL+"_eval.json")).read_bytes())["rows"]
    rows=[r for r in rows if r["outcome"]=="arrival" and r["route"] in ("route_02","route_05","route_08","route_13","route_14","route_10","route_11","route_16")]
    with ProcessPoolExecutor(4,initializer=init) as pool:
        for i in range(0,min(len(rows),256),8):
            for z in pool.map(test,rows[i:i+8]):
                if z:
                    (g.OUT/"stable-example.json").write_text(json.dumps(z))
                    print(z["row"],z["event"],flush=True)
                    break
            else:continue
            break
