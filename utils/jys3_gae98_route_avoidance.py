"""On the original 29 departures, test one different lane decision at the first clear opportunity."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_gae98_cutin_replay as replay

C,L = replay.C,replay.L
ROOT = replay.ROOT.parent/"jys3_gae98_route_avoidance"


def drive(env,model,transform,xml,row,trigger=None):
    env.ego_depart_lane=str(row["lane"])
    obs,_=C.start_trace(lambda:env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"}),xml)
    assert replay.digest(obs)==row["initial_sha256"]
    frames=[]
    counts=Counter()
    entered=False
    for t in range(1,1001):
        edge,lane,path,idx=env.snapshot[:4]
        x=transform(obs[:50])
        tensor=C.torch.from_numpy(x).unsqueeze(0)
        a=model.det_action(tensor)[0].numpy().copy()
        cmd=env.sim._quantize_lane_change(a[1])
        counts[cmd]+=1
        if trigger and t==trigger["step"]:
            assert trigger["hash"]==replay.digest(x) and cmd!=trigger["dir"]
            a[1]=trigger["dir"]
        if trigger is None:
            dst=path[idx+1] if idx+1<len(path) else None
            lanes=env.sim.links.get(edge,{}).get(dst,set())
            delta=min((i-lane for i in lanes),key=lambda n:(abs(n),n),default=0)
            direc=(delta>0)-(delta<0)
            event=dict(edge=edge,lane=lane,dst=dst,step=t,cmd=cmd,dir=direc,clear=False,
                       speed=env.snapshot[7],remaining=env.snapshot[-1])
            if direc and x[38]<=.5:
                m=C.diag.measurements(x,direc,env.visibility,env.vmax)
                if not C.dangerous(m):
                    prob=model(tensor)[2].softmax(-1)[0].tolist()
                    event.update(clear=True,view=m,hash=replay.digest(x),p_keep=prob[1],p_toward=prob[direc+1])
            frames.append(event)
        obs,_,done,cut,info=env.step(a)
        if trigger and t>=trigger["step"] and env.snapshot and env.snapshot[0]==trigger["dst"]:
            entered=True
        if done or cut:
            break
    moves=C.finish_trace(env,xml,t)
    if trigger:
        return dict(outcome=info["outcome"],entered=entered,actual=any(
            k==trigger["step"] and b[1]-a[1]==trigger["dir"] for k,a,b in moves),steps=t)
    assert (t,info["outcome"],counts[-1],counts[1],len(moves))==(
        row["steps"],row["outcome"],row["right_commands"],row["left_commands"],row["moves"])
    if info["outcome"]=="timeout":
        tail=frames[-150:]
        return dict(tail_edges=sorted({f["edge"] for f in tail}),metres=sum(f["speed"]*.2 for f in tail),
                    last_speed=tail[-1]["speed"],last_edge=tail[-1]["edge"])
    dep=info["off_route_detail"]
    approach=[f for f in frames if f["edge"]==dep["src"] and f["dst"]==dep["expected"]]
    assert approach
    clear=[f for f in approach if f["clear"]]
    missed=[f for f in clear if f["cmd"]!=f["dir"]]
    return dict(detail=dep,clear_steps=len(clear),toward_clear=sum(f["cmd"]==f["dir"] for f in clear),
                keep_clear=sum(f["cmd"]==0 for f in clear),opposite_clear=sum(f["cmd"]==-f["dir"] for f in clear),
                first=missed[0] if missed else None)


def batch(rows):
    C.torch.set_num_threads(1)
    model=C.final.network(replay.POLICY)
    with C.final.p.roll.baseline.simulator() as env,C.torch.inference_mode(),TemporaryDirectory() as tmp:
        transform=C.p.lateral_transform(env)
        for row in rows:
            xml=Path(tmp)/"moves.xml"
            result=drive(env,model,transform,xml,row)
            trigger=result.get("first")
            if trigger:
                result["probe"]=drive(env,model,transform,xml,row,trigger)
            L.save(ROOT/"cases"/f"{row['seed']}.json",dict(original=row,**result))


def main():
    assert replay.mid.ft.sha(replay.POLICY)==replay.mid.ARMS["lambda98"]
    rows=[r for h in L.HOLDS for i in range(C.old.PANELS[h][1])
          for r in L.read(replay.SOURCE/"parts"/f"100000_{h}_{i}.json") if r["outcome"] in ("timeout","off_route")]
    assert Counter(r["outcome"] for r in rows)=={"timeout":3,"off_route":29}
    pending=[r for r in rows if not (ROOT/"cases"/f"{r['seed']}.json").exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(batch,[pending[i:i+4] for i in range(0,len(pending),4)]))
    results=[L.read(ROOT/"cases"/f"{r['seed']}.json") for r in rows]
    summary=dict(clear=sum(bool(r.get("clear_steps")) for r in results),
                 probes=Counter(r["probe"]["outcome"] for r in results if "probe" in r))
    L.save(ROOT/"summary.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    main()
