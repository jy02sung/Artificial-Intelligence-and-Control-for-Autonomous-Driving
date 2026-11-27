from pathlib import Path
from collections import Counter
import hashlib
from utils.storage import save_samples
from utils.simulator import simulator
from utils.storage import load_rollouts
from utils.expert import control
from utils.storage import read
from utils.storage import save
import numpy as np
import torch
import torch.nn.functional as F
from utils.networks import HybridActorCritic
from utils.storage import atomic_torch
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_final_relu_dagger"
ROUTES = tuple(range(1,17))
OUTCOMES = ("arrival","collision","stall","timeout","off_route")


def network(checkpoint=None):
    model=HybridActorCritic(50,hidden_sizes=(256,256),activation="relu")
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["policy"])
    return model.eval()


def checkpoint(k):
    return ROOT / ("bc.pt" if k==0 else f"dagger{k}.pt")


def run_episodes(task, *, record_terminal=False, record_commands=False, policy_input_factory=None):
    name, cases, selected, model_path, collect, demo = task
    torch.set_num_threads(1)
    model = network(model_path) if model_path else None
    folder=ROOT/name
    folder.mkdir(exist_ok=True,parents=True)
    rows=[]
    with simulator() as env, torch.inference_mode():
        assert env.sim.physical_forks and not env.sim.lc_safety_enabled and env.sim.occlusion
        assert np.isclose(env.sim.step_length,.2)
        transform = policy_input_factory(env) if policy_input_factory else None
        for route,rep,base in cases:
            path=f"route_{route:02d}"
            lane=min(rep%2,env.counts[env.routes[path][0]]-1)
            env.ego_depart_lane=str(lane)
            seed=base+rep*100+route
            while True:
                obs,_=env.reset(seed=seed,options={"route":path})
                digest=hashlib.sha256(obs.tobytes()).hexdigest()
                left=right=0
                slow=longest=0
                previous=None
                xs,ys=[],[]
                for step in range(1,1001):
                    assert obs.shape==(52,) and not obs[50:].any()
                    state=obs[:50]
                    label,wait,direction,_=control(state,env,"queue_tail",revised=True,short_reserve=selected)
                    key=(*env.snapshot[:2],direction) if wait and env.snapshot[7]<1. else None
                    slow=(slow+1 if key==previous else 1) if key else 0
                    longest=max(longest,slow)
                    previous=key
                    if collect:
                        xs.append(state.copy())
                        ys.append(label)
                    policy_state = transform(state) if transform else state
                    action=label if model is None else model.det_action(torch.from_numpy(policy_state).unsqueeze(0))[0].numpy()
                    if record_commands:
                        left += int(action[1] > .5)
                        right += int(action[1] < -.5)
                    if record_terminal:
                        before = env.snapshot
                    obs,_,done,cut,info=env.step(action)
                    if done or cut:
                        break
                assert (done or cut) and info["outcome"] in OUTCOMES
                assert info["outcome"]!="arrival" or info["route_intact"]
                accepted=not demo or (info["outcome"]=="arrival" and (name!="train" or longest<50))
                row=dict(route=route,rep=rep,lane=lane,seed=seed,steps=step,outcome=info["outcome"],
                         longest_slow=longest,accepted=accepted,initial_sha256=digest)
                if record_commands:
                    row.update(left_commands=left, right_commands=right)
                if record_terminal:
                    end = env.snapshot or before
                    row.update(start_edge=env.routes[path][0], end_edge=end[0], end_lane=end[1],
                               last_speed=float(end[7]), remaining_lane_m=float(end[8]),
                               before_edge=before[0], off_route_detail=info.get("off_route_detail"))
                rows.append(row)
                if accepted:
                    if collect:
                        save_samples(folder,row,np.stack(xs),np.stack(ys))
                    break
                seed+=10000
                assert seed < base+1000000, "replacement would enter a different seed family"
    return rows


def schedule(pool,name,base,reps,selected,model_path=None,collect=False,demo=False,routes=ROUTES):
    folder=ROOT/name
    folder.mkdir(exist_ok=True,parents=True)
    index=folder/"episodes.json"
    if index.exists():
        rows=read(index)
    else:
        cases=[(route,rep,base) for rep in range(reps) for route in routes]
        tasks=[(name,cases[i:i+4],selected,model_path,collect,demo) for i in range(0,len(cases),4)]
        rows=[row for part in pool.map(run_episodes,tasks) for row in part]
        save(index,rows)
    accepted=[r for r in rows if r["accepted"]]
    assert [(r["route"],r["rep"]) for r in accepted]==[(r,i) for i in range(reps) for r in routes]
    assert len({r["seed"] for r in rows})==len(rows)
    assert not collect or all((folder/r["file"]).exists() for r in accepted)
    print(name,len(rows),dict(Counter(r["outcome"] for r in accepted)),flush=True)
    return rows


def collect_demos(pool):
    selected=True  # The final expert uses a 12 m queue reserve on low-speed links.
    save(ROOT/"expert.json", {"short_reserve":selected})
    train=schedule(pool,"train",76000000,10,selected,collect=True,demo=True)
    valid=schedule(pool,"valid",77000000,2,selected,collect=True,demo=True)
    return selected,train,valid


def dataset(rows,name):
    accepted=[r for r in rows if r["accepted"]]
    return load_rollouts(accepted,ROOT/name)


def fit(k,x,y,vx,vy,critical,*,extended_bc=False,uniform_dir=None,initial=None,output_dir=None,learning_rate=None):
    assert not extended_bc or k==0
    destination = uniform_dir if uniform_dir is not None else (ROOT/"bc100" if extended_bc else ROOT)
    if output_dir is not None:
        destination=output_dir
    destination.mkdir(exist_ok=True)
    folder=destination/f"fit{k}.json"
    best_path=destination/checkpoint(k).name
    if folder.is_file() and best_path.is_file():
        return
    torch.manual_seed(11)
    model=network(initial if initial is not None else (checkpoint(k-1) if k else None))
    lr=.0003 if k==0 else .0002 if k<=2 else .0001 if k<=5 else .00005
    if learning_rate is not None:
        lr=learning_rate
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator().manual_seed(11+k)
    cls=(1-y[:,1]).long()
    vc=(1-vy[:,1]).long()
    changed=torch.where(cls!=1)[0]
    critical=torch.tensor(critical,dtype=torch.long)
    empty=critical[:0]
    def sample(indices):
        return indices[torch.randint(len(indices),(32,),generator=generator)] if len(indices) else empty
    ledger=[]
    best=-float("inf")
    epochs=12 if k else (100 if extended_bc else 50)
    for epoch in range(1,epochs+1):
        disagree=empty
        if k and uniform_dir is None:
            model.eval()
            with torch.no_grad():
                predicted=torch.cat([model(part)[2].argmax(1) for part in x.split(4096)])
                disagree=torch.where(predicted!=cls)[0]
        model.train()
        total_acc=total_lane=0.
        batches=0
        for base in torch.randperm(len(x),generator=generator).split(256):
            if uniform_dir is None:
                acc_ids=torch.cat((base,sample(critical)))
                ids=torch.cat((acc_ids,sample(changed),sample(disagree)))
            else:
                acc_ids=ids=base
            mu,_,logits,_=model(x[ids])
            loss_acc=F.mse_loss(mu[:len(acc_ids),0],y[acc_ids,0])
            loss_lane=F.cross_entropy(logits,cls[ids])
            optimizer.zero_grad(set_to_none=True)
            (loss_acc+loss_lane).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total_acc+=loss_acc.item()
            total_lane+=loss_lane.item()
            batches+=1
        model.eval()
        with torch.no_grad():
            mu,_,z,_=model(vx)
            conf=torch.bincount(vc*3+z.argmax(1),minlength=9).reshape(3,3)
            f1=2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)
            mae=(mu[:,0]-vy[:,0]).abs().mean().item()
            score=float((f1[0]+f1[2])/2)-.1*mae
        ledger.append(dict(epoch=epoch,score=score,f1=f1.tolist(),mae=mae,disagree=len(disagree),
                           mse=total_acc/batches,cross_entropy=total_lane/batches))
        if score>best:
            best,chosen=score,epoch
            atomic_torch(dict(policy=model.state_dict(),activation="relu",hidden_sizes=[256,256],
                              obs_dim=50,seed=11,round=k,epoch=epoch),best_path)
        if extended_bc and epoch==50:
            reference=read(ROOT/"fit0.json")
            assert ledger==reference["epochs"] and chosen==reference["selected_epoch"]
            a=torch.load(best_path,weights_only=True)["policy"]
            b=torch.load(checkpoint(0),weights_only=True)["policy"]
            assert all(torch.equal(a[key],b[key]) for key in a)
        if extended_bc and epoch==epochs:
            atomic_torch(dict(policy=model.state_dict(),activation="relu",hidden_sizes=[256,256],
                              obs_dim=50,seed=11,round=0,epoch=epoch),destination/"last.pt")
        if epoch%5==0:
            print("fit",k,epoch,round(score,4),flush=True)
    save(folder,dict(epochs=ledger,selected_epoch=chosen,lr=lr,states=len(x),critical=len(critical),
                       changed=len(changed),sha256=hashlib.sha256(best_path.read_bytes()).hexdigest()))


def planned(base,reps,routes=ROUTES):
    return [base+rep*100+r for rep in range(reps) for r in routes]


def tally(rows):
    c=Counter(r["outcome"] for r in rows)
    return {**{reason:c[reason] for reason in OUTCOMES},"slow10":sum(r["longest_slow"]>=50 for r in rows)}


def pipeline(pool):
    selected,train,valid=collect_demos(pool)
    torch.set_num_threads(2)
    x,y=dataset(train,"train")
    vx,vy=dataset(valid,"valid")
    seeds=set()
    def reserve(values):
        new=set(values)
        assert len(new)==len(values) and new.isdisjoint(seeds), "seed overlap"
        seeds.update(new)
    for rows in (train,valid):
        reserve([r["seed"] for r in rows])
    for base,reps in ((95380000,8),(81260100,4),(91000000,16),(104320000,16)):
        reserve(planned(base,reps))
    screens={}
    counts={}
    critical=[]
    def rank(k):
        c=counts[k]
        return c["arrival"],-c["collision"],-c["off_route"],-k
    for k in range(9):
        if k:
            fails=Counter(r["route"] for r in screens[k-1] if r["outcome"]!="arrival")
            weak=sorted(ROUTES,key=lambda r:(-fails[r],r))[:4]
            for label,base,n,routes in (("uniform",78000000,8,ROUTES),("weak",79000000,4,weak)):
                name=f"R{k}/{label}"
                base+=k*100000
                reserve(planned(base,n,routes))
                rows=schedule(pool,name,base,n,selected,checkpoint(k-1),collect=True,routes=routes)
                assert [r["seed"] for r in rows]==planned(base,n,routes)
                offset=len(x)
                for r in rows:
                    end=offset+r["steps"]
                    if r["outcome"]!="arrival":
                        critical.extend(range(max(offset,end-15),end))
                    offset=end
                nx,ny=dataset(rows,name)
                x,y=torch.cat((x,nx)),torch.cat((y,ny))
                assert len(x)==offset
        fit(k,x,y,vx,vy,critical)
        rows=schedule(pool,f"screen{k}",95380000,8,selected,checkpoint(k))
        assert [r["seed"] for r in rows]==planned(95380000,8)
        if k:
            assert all(a["initial_sha256"]==b["initial_sha256"] for a,b in zip(screens[0],rows))
        screens[k],counts[k]=rows,tally(rows)
        save(ROOT/"rounds.json",counts)
        if k==5:
            c=counts[max(range(1,6),key=rank)]
            if c["arrival"]>=122 and c["collision"]<=6:
                break
    best=max(screens,key=rank)
    save(ROOT/"selection.json",dict(best=best,last=k,sha256=read(ROOT/f"fit{best}.json")["sha256"]))
    hold={}
    for m in sorted({0,best}):
        hold[m]=schedule(pool,f"hold64_{m}",81260100,4,selected,checkpoint(m))
    assert all(a["initial_sha256"]==b["initial_sha256"] for a,b in zip(hold[0],hold[best]))
    final=schedule(pool,"final256",91000000,16,selected,checkpoint(best))
    novel=schedule(pool,"novel256",104320000,16,selected,checkpoint(best))
    for rows,base,reps in ((hold[0],81260100,4),(hold[best],81260100,4),(final,91000000,16),(novel,104320000,16)):
        assert [r["seed"] for r in rows]==planned(base,reps)
    def byroute(rows):
        return {r:tally([row for row in rows if row["route"]==r]) for r in ROUTES}
    result=dict(best=best,last=k,rounds=counts,hold={m:tally(rows) for m,rows in hold.items()},
                transitions=dict(Counter(a["outcome"]+" -> "+b["outcome"] for a,b in zip(hold[0],hold[best]))),
                final=tally(final),novel=tally(novel),screen_routes=byroute(screens[best]),
                final_routes=byroute(final),novel_routes=byroute(novel),distinct_seeds=len(seeds))
    save(ROOT/"summary.json",result)
    return result
