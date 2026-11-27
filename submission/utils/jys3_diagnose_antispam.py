"""Read-only SUMO collision replay and fixed-state policy drift measurements."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils import jys3_pure_ppo_antispam as exp
from env.sumo_env import traci

np, torch, read = exp.np, exp.torch, exp.read
ROOT = exp.ROOT.parent / "jys3_diagnose_antispam"
SPECS = [("S2", 33, 75000), ("Q1", 22, 200000), ("S1", 22, 300000)]
STEPS = (0, 25000, 50000, 75000)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8")
    temp.replace(path)


def split_lane(lane):
    edge, index = lane.rsplit("_", 1)
    return [edge, int(index)]


def measurements(obs, direction, visibility, vmax):
    i = 1 + 4 * (2-direction)
    a,b,c,d = map(float, obs[i:i+4])
    missing_f = a >= 1-1e-6 and b >= 1-1e-6
    missing_r = c <= -1+1e-6 and d <= -1+1e-6
    fg,fc = (None,None) if missing_f else (a*visibility,-b*vmax)
    rg,rc = (None,None) if missing_r else (-c*visibility,d*vmax)
    ttc = [gap/v for gap,v in ((fg,fc),(rg,rc)) if gap is not None and v>0]
    overlap = direction!=0 and bool(obs[37 if direction==1 else 36]>.5)
    speed = float(obs[0])*vmax
    tight = (overlap or (fg is not None and fg<=3+.45*speed+max(0,fc))
             or (rg is not None and rg<=3+.50*max(0,speed+rc)+max(0,rc)))
    return dict(front_gap=fg, rear_gap=rg, front_closing=fc, rear_closing=rc,
                ttc=min(ttc,default=None), overlap=overlap, tight=bool(tight))


def merge_lanes(env):
    net=ET.parse(Path(env.sim.cfg_path).parent/"network.net.xml")
    sources=defaultdict(set)
    for c in net.findall("connection"):
        edge=c.get("from")
        if not edge.startswith(":"):
            sources[(c.get("to"),int(c.get("toLane")))].add(edge)
    return {lane for lane,edges in sources.items() if len(edges)>1}


def neighbors():
    v=traci.vehicle
    return {i:[v.getRoadID(i),v.getLaneIndex(i),v.getLanePosition(i),v.getSpeed(i),v.getLength(i)]
            for i in v.getIDList()}


def collision_events(ego):
    return [dict(collider=c.collider,victim=c.victim,kind=c.type,lane=c.lane,pos=c.pos,
                 collider_speed=c.colliderSpeed,victim_speed=c.victimSpeed)
            for c in traci.simulation.getCollisions() if ego in (c.collider,c.victim)]


def one(env, model, transform, merges, row, probe, states):
    ego=env.ego_id
    env.ego_depart_lane=str(row["lane"])
    obs,_=env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"})
    assert hashlib.sha256(obs.tobytes()).hexdigest()==row["initial_sha256"]
    tail=deque(maxlen=5)
    recent=deque(maxlen=20)
    left=right=moves=0
    last_external=None
    for t in range(1,1001):
        state=transform(obs[:50])
        action=model.det_action(torch.from_numpy(state).unsqueeze(0))[0].numpy()
        cmd=env.sim._quantize_lane_change(action[1])
        left+=int(cmd==1)
        right+=int(cmd==-1)
        before=list(env.snapshot[:2])
        if not before[0].startswith(":"):
            last_external=before
        detailed=not probe and t>=row["steps"]-4
        if detailed:
            cars=neighbors()
            leader=traci.vehicle.getLeader(ego,env.visibility)
            frame=dict(step=t,before=before,obs=state.tolist(),cmd=cmd,mu=float(action[0]),
                       speed=cars[ego][3],lead_id=leader[0] if leader else None,cars=cars,
                       view={str(d):measurements(state,d,env.visibility,env.vmax) for d in (-1,0,1)})
        if probe and (t-1)%4==0:
            states.append(state.copy())
        obs,_,done,cut,info=env.step(action)
        after=list(env.snapshot[:2]) if env.snapshot else None
        if after and before[0]==after[0] and before[1]!=after[1]:
            moves+=1
        if not probe:
            recent.append([t,before,after])
        if detailed:
            frame.update(after=after,requested_applied=bool(info["lane_change_applied"]))
            tail.append(frame)
        if done or cut:
            break
    assert (t,info["outcome"],left,right)==(row["steps"],row["outcome"],row["left_commands"],row["right_commands"])
    if "within_edge_moves" in row:
        assert moves==row["within_edge_moves"]
    identity={k:row[k] for k in ("seed","route","lane","steps","outcome")}
    if probe:
        return dict(**identity,moves=moves)
    events=collision_events(ego) if info["outcome"]=="collision" else []
    final_lane=split_lane(events[0]["lane"]) if events else after
    ids={v for e in events for v in (e["victim"],e["collider"])} | {ego}
    for f in tail:
        f["cars"]={k:v for k,v in f["cars"].items() if k in ids}
    in_merge=any(a and b and a[0]!=b[0] and tuple(b) in merges for _,a,b in list(recent)[-3:])
    return dict(**identity,ego=ego,events=events,tail=list(tail),recent=list(recent),
                final_lane=final_lane,last_external=last_external,
                merge_entry=in_merge,off_route=info.get("off_route_detail"))


def policy_path(track,seed,step):
    return exp.checkpoint(seed) if track=="Q1" else exp.folder(track,seed)/f"policy_{step}.pt"


def event_replay(env, model, transform, merges, row, output):
    start = traci.start
    with patch.object(traci, "start", lambda cmd: start(cmd + ["--lanechange-output", str(output)])):
        r = one(env, model, transform, merges, row, False, [])
    end_time = traci.simulation.getTime()
    dt = env.sim.step_length
    env.close()  # Flush SUMO's authoritative from/to record, including removed vehicles.
    participants = {r["ego"]} | {v for e in r["events"] for v in (e["victim"], e["collider"])}
    moves = []
    for change in ET.parse(output).getroot().findall("change"):
        if change.get("id") not in participants:
            continue
        step = r["steps"] + 1 - round((end_time - float(change.get("time"))) / dt)
        if r["steps"] - 20 < step <= r["steps"]:
            moves.append([step, change.get("id"), split_lane(change.get("from")), split_lane(change.get("to"))])
    r["lane_changes"] = moves
    seen = {(t,tuple(a),tuple(b)) for t, vehicle, a, b in moves if vehicle == r["ego"]}
    for t, a, b in r["recent"]:
        if b and a[0] == b[0] and a != b:
            assert (t,tuple(a),tuple(b)) in seen, (row["seed"], t, a, b, seen)
    return r


def shard(job):
    track,seed,step,probe,idx,rows=job
    name=f"{track}_{seed}_{'probe' if probe else 'verified'}_{idx}"
    out=ROOT/"shards"/(name+".json")
    bank=out.with_suffix(".npz")
    if out.exists() and (not probe or bank.exists()):
        return
    torch.set_num_threads(1)
    model=exp.final.network(policy_path(track,seed,step))
    states=[]
    with exp.final.p.roll.baseline.simulator() as env, torch.inference_mode():
        transform=exp.p.lateral_transform(env)
        merges=merge_lanes(env)
        if probe:
            items=[one(env,model,transform,merges,r,True,states) for r in rows]
        else:
            with TemporaryDirectory() as tmp:
                items=[event_replay(env,model,transform,merges,r,Path(tmp)/"moves.xml") for r in rows]
    if probe:
        with bank.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f,states=np.stack(states))
        bank.with_suffix(".tmp").replace(bank)
    save(out,items)


def collect():
    (ROOT/"shards").mkdir(parents=True,exist_ok=True)
    jobs=[]
    for track,seed,step in SPECS:
        rows=[r for panel in exp.q.HOLDS for r in read(exp.folder(track,seed)/f"eval_{step}_{panel}.json")["rows"]]
        group=[r for r in rows if r["outcome"]=="collision" or (track=="S2" and r["outcome"]=="off_route")]
        assert len(rows)==576 and len(group)=={"S2":87,"Q1":50,"S1":64}[track]
        jobs.extend((track,seed,step,False,i,group[i:i+8]) for i in range(0,len(group),8))
    for seed,step in exp.PEAK.items():
        rows=read(exp.q.panel("Q1",seed,step,"screen"))["rows"]
        jobs.extend(("Q1",seed,step,True,i,rows[i:i+8]) for i in range(0,len(rows),8))
    with ProcessPoolExecutor(4) as pool:
        for _ in pool.map(shard,jobs):
            pass


def drift():
    x=torch.from_numpy(np.concatenate([np.load(f)["states"] for f in sorted((ROOT/"shards").glob("*_probe_*.npz"))]))
    results=[]
    for seed in exp.SEEDS:
        folder=exp.folder("S2",seed)
        ref=exp.final.network(exp.checkpoint(seed))
        mu0,s0,z0,_=ref(x)
        l0=z0.log_softmax(-1)
        change=z0.argmax(1)!=1
        warm=read(folder/"warm_check.json")
        updates=[json.loads(l) for l in (folder/"updates.jsonl").read_text().splitlines()]
        for step in STEPS:
            if step:
                net=exp.final.network(policy_path("S2",seed,step))
                rows=read(folder/f"eval_{step}_screen.json")["rows"]
                cost=sum(u["metrics"]["command_cost"] for u in updates if u["step"]<=step)
            else:
                net=exp.final.network()
                weights=torch.load(folder/"warm.pt",map_location="cpu",weights_only=True)
                net.load_state_dict({k:v for k,v in weights.items() if not k.startswith("value_trunk.")})
                rows=[r for f in (ROOT/"shards").glob(f"Q1_{seed}_probe_*.json") for r in read(f)]
                cost=0.
            mu,std,z,_=net(x)
            lane=(l0.exp()*(l0-z.log_softmax(-1))).sum(1)
            accel=(torch.log(std/s0)+(s0.square()+(mu-mu0).square())/(2*std.square())-.5).flatten()
            if not step:
                assert torch.equal(mu,mu0) and torch.equal(z,z0) and torch.equal(std,s0)
            needs=[r for r in rows if exp.p.groups()[r["route"],r["lane"]]=="needs_change"]
            assert len(rows)==128 and len(needs)==52
            warm_cost=sum(u["command_cost"] for u in warm["updates"])
            results.append(dict(seed=seed,step=step,n=len(x),kl_lane=lane.mean().item(),kl_accel=accel.mean().item(),
                                kl=(lane+accel).mean().item(),kl_change=lane[change].mean().item(),
                                change_n=int(change.sum()),retained=float((z.argmax(1)[change]==z0.argmax(1)[change]).float().mean()),
                                warm_cost=warm_cost,ppo_cost=cost,total_cost=cost+warm_cost,
                                moves=sum(r.get("moves",r.get("within_edge_moves")) for r in rows),
                                required_arrivals=sum(r["outcome"]=="arrival" for r in needs),
                                off_route=sum(r["outcome"]=="off_route" for r in rows),arrivals=sum(r["outcome"]=="arrival" for r in rows)))
    return results


def command_history():
    """Measure right-command streaks at the first r16 fork and in the longest-count holdout."""
    records=[]
    for seed,step in exp.PEAK.items():
        rows=[r for panel in exp.q.HOLDS for r in read(exp.folder("Q1",seed)/f"eval_{step}_{panel}.json")["rows"]]
        top=max(rows,key=lambda r:r["right_commands"])
        cases=[r for r in rows if r["route"]==16 and r["lane"]==0]
        if top not in cases:
            cases.append(top)
        model=exp.final.network(exp.checkpoint(seed))
        with exp.final.p.roll.baseline.simulator() as env:
            transform=exp.p.lateral_transform(env)
            for row in cases:
                env.ego_depart_lane=str(row["lane"])
                obs,_=env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"})
                assert hashlib.sha256(obs.tobytes()).hexdigest()==row["initial_sha256"]
                first=env.snapshot[0]
                whole=Counter()
                approach=Counter()
                streak=longest=0
                for t in range(1,1001):
                    edge,lane=env.snapshot[:2]
                    state=transform(obs[:50])
                    action=model.det_action(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                    cmd=env.sim._quantize_lane_change(action[1])
                    obs,_,done,cut,info=env.step(action)
                    nav=env.snapshot
                    values=dict(steps=1,right=int(cmd==-1),left=int(cmd==1),
                                right_in_zero=int(cmd==-1 and lane==0 and not edge.startswith(":")),
                                changed_right=int(cmd==-1 and nav is not None and nav[0]==edge and nav[1]<lane))
                    whole.update(values)
                    if edge==first:
                        approach.update(values)
                    streak=streak+1 if cmd==-1 else 0
                    longest=max(streak,longest)
                    if done or cut:
                        break
                assert (info["outcome"],t,whole["right"],whole["left"])==(row["outcome"],row["steps"],row["right_commands"],row["left_commands"])
                records.append(dict(model_seed=seed,seed=row["seed"],route=row["route"],lane=row["lane"],
                                    first=first,outcome=row["outcome"],whole=dict(whole),approach=dict(approach),
                                    longest=longest,max_right_case=row==top))
    return records


def boundary_counterfactual():
    results=[]
    for row in read(ROOT/"command_history.json"):
        if not row["max_right_case"]:
            continue
        seed=row["model_seed"]
        model=exp.final.network(exp.checkpoint(seed))
        trials=[]
        with exp.final.p.roll.baseline.simulator() as env:
            transform=exp.p.lateral_transform(env)
            for replace in (False,True):
                env.ego_depart_lane=str(row["lane"])
                obs,_=env.reset(seed=row["seed"],options={"route":f"route_{row['route']:02d}"})
                hasher=hashlib.sha256()
                rewards=[]
                count=0
                for _ in range(1000):
                    hasher.update(obs.tobytes())
                    action=model.det_action(torch.from_numpy(transform(obs[:50])).unsqueeze(0))[0].numpy().copy()
                    edge,lane=env.snapshot[:2]
                    if not edge.startswith(":") and lane==0 and env.sim._quantize_lane_change(action[1])==-1:
                        count+=1
                        if replace:
                            action[1]=0.
                    obs,reward,done,cut,info=env.step(action)
                    rewards.append(float(reward))
                    if done or cut:
                        break
                hasher.update(obs.tobytes())
                trials.append(dict(steps=len(rewards),hash=hasher.hexdigest(),outcome=info["outcome"],commands=count,
                                   reward_hash=hashlib.sha256(np.array(rewards,dtype=np.float64).tobytes()).hexdigest()))
        results.append(dict(model_seed=seed,route=row["route"],seed=row["seed"],original=trials[0],
                            replaced=trials[1],identical=trials[0]==trials[1]))
    return results


if __name__=="__main__":
    collect()
    with torch.inference_mode():
        torch.set_num_threads(1)
        save(ROOT/"drift.json",drift())
        save(ROOT/"command_history.json",command_history())
        save(ROOT/"counterfactual.json",boundary_counterfactual())
    from utils.jys3_antispam_collision_report import generate
    generate()
