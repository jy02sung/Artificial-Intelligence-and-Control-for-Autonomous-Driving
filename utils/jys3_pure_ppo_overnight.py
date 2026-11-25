"""Six matched PPO conditions. Run python -m utils.jys3_pure_ppo_overnight to start or resume."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
os.environ["IC_LIBSUMO"]="1"

import numpy as np
import torch
from torch import nn
from utils import jys3_ppo_redesign_v2 as prev
from utils import jys3_reward_mix_v2 as roads

old,final,read,save=prev.old,prev.final,prev.read,prev.save
ROOT=old.ROOT.parent/"jys3_pure_ppo_overnight"
REPORT=final.DOCS/"jys3-pure-ppo-overnight.md"
ARMS=tuple(f"P{i}" for i in range(6))
EXPERIENCE={"P1","P5"}
PREP={"P2","P3","P4"}
LATERAL={"P4","P5"}
MARK=50000
GEOMETRY=prev.diagnosis.geometry


def agent():
    a=old.PPO(50,2,hidden_sizes=(256,256),activation="relu",seed=11,device="cpu",
              gamma=.995,gae_lambda=.95,n_steps=2048,n_epochs=10,minibatch_size=64,
              lr=.0001,lane_keep_bias=1.,lane_entropy_coef=.002,entropy_coef=0.)
    prev.separate(a)
    return a


def digest(policy):
    h=hashlib.sha256()
    for k,v in sorted(policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def config(arm,step):
    if arm in {"P3","P4","P5"} and step<160000:
        if step<80000:
            return (5000,True) if arm=="P3" else (3000,False)
        return 8000,False
    return 15000,False


def lateral_transform(env):
    # The expert wait metric and raw initial-state fingerprint use the original evaluator.
    sim=env.sim
    original=sim.signs,sim.road_flags
    roads.configure_roads(sim,Path(sim.cfg_path).parent/"network.net.xml")
    signs,flags=sim.signs,sim.road_flags
    sim.signs,sim.road_flags=original
    assert set(signs)==set(sim.signs)
    def transform(x):
        edge,_,path,_,fork,*_=env.snapshot
        x=x.copy()
        x[34]=signs[path[fork],path[fork+1]] if fork is not None else 0.
        x[40:50]=flags.get(edge,sim.no_links)
        return x
    return transform


class Map:
    def __init__(self,env):
        self.env=env
        self.demand=prev.Demand(env)
        net=ET.parse(Path(env.sim.cfg_path).parent/"network.net.xml")
        self.links=defaultdict(list)
        for link in net.findall("connection"):
            self.links[link.get("from"),link.get("to")].append((int(link.get("fromLane")),int(link.get("toLane"))))
        self.short={}
        self.active=None
        for r in range(1,17):
            path=env.routes[f"route_{r:02d}"]
            forks=[i for i in range(len(path)-1) if (path[i],path[i+1]) in env.signs]
            if forks:
                suffix=path[forks[-1]:]
                ready=sorted(env.links[suffix[0]][suffix[1]])
                other=sorted(set(range(env.counts[suffix[0]]))-set(ready))
                self.short[r]=(suffix,ready,other)
        self.flexible=[r for r,(_,_,other) in self.short.items() if other]

    def set_stage(self,stage):
        if stage==self.active:
            return
        vph,short=stage
        root=deepcopy(self.demand.high)
        flows=root.findall("flow")
        sizes=np.array([int(f.get("number")) for f in flows])*vph/15000.
        numbers=np.floor(sizes).astype(int)
        numbers[np.argsort(-(sizes-numbers),kind="stable")[:vph-int(numbers.sum())]]+=1
        assert numbers.sum()==vph
        for flow,n in zip(flows,numbers):
            flow.set("number",str(n))
        if short:
            for r,(edges,_,_) in self.short.items():
                key=f"short_{r:02d}"
                root.insert(list(root).index(flows[0]),ET.Element("route",id=key,edges=" ".join(edges)))
                self.env.sim.routes[key]=edges
        ET.ElementTree(root).write(self.demand.path,encoding="utf-8",xml_declaration=True)
        assert sum(int(f.get("number")) for f in ET.parse(self.demand.path).findall("flow"))==vph
        self.active=stage

    def departure(self,r,block):
        if self.active[1] and r in self.short:
            _,ready,other=self.short[r]
            # Two forced-ready routes, thirteen flexible. Over 26 repeats per
            # route exactly 195 of 390 fork starts are ready and 195 need a change.
            assert len(self.short)==15 and len(self.flexible)==13
            use_ready=not other or (block+2*self.flexible.index(r))%26<11
            lanes=ready if use_ready else other
            return f"short_{r:02d}",lanes[block%len(lanes)],"ready" if use_ready else "change"
        key=f"route_{r:02d}"
        return key,min(block%2,self.env.counts[self.env.routes[key][0]]-1),"full"

    @lru_cache(maxsize=512)
    def ready(self,path,index,fork):
        allowed=set(self.env.links[path[fork]][path[fork+1]])
        for i in range(fork-1,index-1,-1):
            allowed={a for a,b in self.links[path[i],path[i+1]] if b in allowed}
        return allowed


class Experience:
    def __init__(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(910)
            self.net=nn.Sequential(nn.Linear(50,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.opt=torch.optim.Adam(self.net.parameters(),lr=.001)
        self.rng=np.random.default_rng(910)
        self.x=np.zeros((20000,50),np.float32)
        self.y=np.zeros(20000,np.int8)
        self.cursor=self.n=self.updates=0
        self.labels=Counter()
        self.fresh=[]

    @torch.no_grad()
    def prob(self,x):
        return self.net(torch.as_tensor(x)).flatten().sigmoid().numpy()

    def label(self,states,y):
        if not states:
            return
        x=np.stack(states)
        # Test on each newly resolved fork with the policy predictor frozen for the rollout.
        self.fresh.append((np.full(len(x),y,np.int8),self.prob(x)))
        ids=(self.cursor+np.arange(len(x)))%len(self.x)
        self.x[ids]=x
        self.y[ids]=y
        self.cursor=(self.cursor+len(x))%len(self.x)
        self.n=min(len(self.x),self.n+len(x))
        self.labels[str(y)]+=1

    def update(self):
        validation=None
        if self.fresh:
            y,p=map(np.concatenate,zip(*self.fresh))
            pos=y==1
            n=int(pos.sum()); m=len(y)-n
            auc=None
            if n and m:
                negatives=np.sort(p[~pos])
                below=np.searchsorted(negatives,p[pos],side="left")
                at_most=np.searchsorted(negatives,p[pos],side="right")
                auc=float(np.mean((below+at_most)/(2*m)))
            validation=dict(n=len(y),positive=n,correct=int(((p>=.5)==pos).sum()),auc=auc)
        self.fresh=[]
        ids=[np.flatnonzero(self.y[:self.n]==v) for v in (0,1)]
        loss=None
        if min(map(len,ids))>=200:
            for _ in range(10):
                batch=np.concatenate([self.rng.choice(s,64,replace=False) for s in ids])
                z=self.net(torch.from_numpy(self.x[batch])).flatten()
                error=nn.functional.binary_cross_entropy_with_logits(z,torch.from_numpy(self.y[batch]).float())
                self.opt.zero_grad()
                error.backward()
                self.opt.step()
            self.updates+=1
            loss=error.item()
        return dict(online_validation=validation,buffer_counts=list(map(len,ids)),
                    labeled_forks=dict(self.labels),optimizer_steps=self.updates*10,bce=loss)

    def state(self):
        return dict(net=self.net.state_dict(),opt=self.opt.state_dict(),rng=self.rng.bit_generator.state,
                    x=self.x,y=self.y,cursor=self.cursor,n=self.n,updates=self.updates,labels=self.labels)

    def restore(self,s):
        self.net.load_state_dict(s["net"])
        self.opt.load_state_dict(s["opt"])
        self.rng.bit_generator.state=s["rng"]
        for k in ("x","y","cursor","n","updates","labels"):
            setattr(self,k,s[k])


class Episodes:
    def __init__(self,env,arm,folder):
        self.env,self.arm,self.folder=env,arm,folder
        self.map=Map(env)
        self.transform=lateral_transform(env) if arm in LATERAL else lambda x:x
        self.exp=Experience() if arm in EXPERIENCE else None
        self.steps=self.count=0
        old.driving.ARRIVAL_REWARD,old.driving.PASS_REWARD=10.,2.5
        old.driving.COLLISION_REWARD,old.driving.TIMEOUT_REWARD=-12.,-8.
        env.progress_cap,env.off_route_penalty=6.,-8.
        assert env.physical_forks and env.occlusion and not env.lc_safety_enabled and not env.prefill_enabled

    def observe(self,x):
        assert len(x)==52 and not x[50:].any()
        return self.transform(x[:50]) if self.env.snapshot is not None else x[:50]

    def reset(self):
        block,i=divmod(self.count,16)
        self.route=int(np.random.default_rng(11+block).permutation(16)[i])+1
        self.seed=205000000+self.count
        self.count+=1
        self.map.set_stage(config(self.arm,self.steps))
        key,lane,self.start_type=self.map.departure(self.route,block)
        self.env.ego_depart_lane=str(lane)
        self.start_lane=lane
        x,info=self.env.reset(seed=self.seed,options={"route":key})
        assert self.env.snapshot[0]==self.env.routes[key][0] and self.env.snapshot[1]==lane
        self.obs=self.observe(x)
        self.actions=[]
        self.idle=0
        self.segment=None
        self.seq=[]
        self.initial_ready={}
        self.prepaid=set()
        self.exp_sum=self.prep_sum=self.total=0.
        return self.obs,info

    def drive(self,action):
        self.actions.append(np.asarray(action).copy())
        x,r,d,c,info=self.env.step(action)
        return self.observe(x),r,d,c,info

    def visible(self,obs,snap):
        return snap[4] if snap and obs[35]>=0 else None

    def is_ready(self,snap):
        _,lane,path,idx,fork,*_=snap
        return lane in self.map.ready(tuple(path),idx,fork)

    def step(self,action):
        before=self.env.snapshot
        key=self.visible(self.obs,before)
        if key is not None:
            if self.exp:
                if key!=self.segment:
                    self.segment,self.seq=key,[]
                self.seq.append(self.obs)
            if self.arm in PREP and key not in self.initial_ready:
                self.initial_ready[key]=self.is_ready(before)
        paid_before=set(self.env.sim.paid_forks)
        nxt,reward,done,cut,info=self.drive(action)
        passed=self.env.sim.paid_forks-paid_before
        next_key=self.visible(nxt,self.env.snapshot)
        self.steps+=1
        clear=info.get("speed") is not None and info["speed"]<2 and (info["gap"] is None or info["gap"]>25)
        self.idle=self.idle+1 if clear else 0
        if clear:
            reward-=.02
        if self.idle>=50 and not(done or cut):
            reward-=9.
            info["outcome"]="clear_stop"
            done=True
        shaping=prep=0.
        if self.exp:
            if self.segment in passed:
                self.exp.label(self.seq,1)
                self.segment,self.seq=None,[]
            elif (done or cut) and info["outcome"] in ("collision","off_route","stall"):
                self.exp.label(self.seq,0)
                self.segment,self.seq=None,[]
            if self.exp.updates and key is not None and key==next_key and not (passed or done or cut):
                p0,p1=self.exp.prob(np.stack([self.obs,nxt]))
                shaping=float(np.clip(.995*p1-p0,-.2,.2))
        if self.arm in PREP:
            if key is not None and key==next_key and not(done or cut) and not self.initial_ready[key] and key not in self.prepaid and self.is_ready(self.env.snapshot):
                self.prepaid.add(key)
                prep+=1.
            if info["fork_passed"] and self.prepaid.intersection(passed):
                prep-=1.
        reward+=shaping+prep
        self.exp_sum+=shaping
        self.prep_sum+=prep
        self.total+=reward
        self.obs=nxt
        if done or cut:
            old.append(self.folder/"episodes.jsonl",dict(step=self.steps,seed=self.seed,route=self.route,lane=self.start_lane,
                       start_type=self.start_type,vph=self.map.active[0],outcome=info["outcome"],r_exp=self.exp_sum,
                       prep_adjustment=self.prep_sum,episode_return=self.total))
        return nxt,reward,done,cut,info

    def state(self):
        return {k:getattr(self,k) for k in ("count","steps","obs","actions","idle","segment","seq","initial_ready",
                                             "prepaid","exp_sum","prep_sum","total")}

    def restore(self,s):
        self.count=s["count"]-1
        self.steps=s["steps"]-len(s["actions"])
        self.reset()
        for action in s["actions"]:
            x,_,d,c,_=self.drive(action)
            assert not(d or c)
            self.obs=x
        assert np.array_equal(self.obs,s["obs"]), "Active drive replay differs from checkpoint"
        for k,v in s.items():
            setattr(self,k,v)


def folder(arm):
    return ROOT/arm


def policy_path(arm,n):
    return folder(arm)/f"policy_{n}.pt"


def train(arm,end):
    torch.set_num_threads(1)
    f=folder(arm)
    f.mkdir(exist_ok=True,parents=True)
    a=agent()
    initial=digest(a.policy)
    assert initial==read(ROOT/"protocol.json")["initial_sha256"]
    save(f/"initial.json",dict(sha256=initial,seed=11))
    resume=f/"resume.pt"
    state=torch.load(resume,weights_only=False) if resume.exists() else None
    logs=("episodes.jsonl","updates.jsonl")
    with final.p.roll.baseline.simulator() as env:
        ep=Episodes(env,arm,f)
        stats=[0.,0]
        if state:
            assert state["initial"]==initial and state["arm"]==arm
            a.policy.load_state_dict(state["model"])
            a.optimizer.load_state_dict(state["optimizer"])
            a.num_timesteps=state["step"]
            ep.restore(state["episodes"])
            if ep.exp:
                ep.exp.restore(state["experience"])
            stats=state["stats"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["np_rng"])
        else:
            ep.reset()
        for name in logs:
            with (f/name).open("ab") as out:
                offset=state["offsets"][name] if state else 0
                assert out.tell()>=offset
                out.truncate(offset)
        obs=ep.obs
        while a.num_timesteps<end:
            limit=min(end,((a.num_timesteps//MARK)+1)*MARK)
            if arm in {"P3","P4","P5"}:
                limit=min([limit]+[n for n in (80000,160000) if a.num_timesteps<n])
            size=min(2048,limit-a.num_timesteps)
            obs,last,_=a.collect_rollout(ep,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=a.buffer.compute_gae(last,a.gamma,a.gae_lambda)
            metrics=a.update(adv,ret)
            diag=ep.exp.update() if ep.exp else None
            n=a.num_timesteps
            old.append(f/"updates.jsonl",dict(step=n,size=size,metrics=metrics,predictor=diag))
            if config(arm,n)!=ep.map.active:
                # The last trajectory bootstrapped from its own observation in compute_gae.
                obs,_=ep.reset()
                stats=[0.,0]
            if n%MARK==0 or n==end:
                state=dict(arm=arm,step=n,initial=initial,model=a.policy.state_dict(),optimizer=a.optimizer.state_dict(),
                           episodes=ep.state(),experience=ep.exp.state() if ep.exp else None,stats=stats,
                           torch_rng=torch.get_rng_state(),np_rng=np.random.get_state(),
                           offsets={name:(f/name).stat().st_size for name in logs})
                final.atomic_torch(state,resume)
        n=a.num_timesteps
        assert n==end
        final.atomic_torch(dict(policy={k:v for k,v in a.policy.state_dict().items() if not k.startswith("value_trunk.")},
                               value_trunk=a.policy.value_trunk.state_dict(),activation="relu",step=n,
                               obs_dim=50,hidden_sizes=[256,256]),policy_path(arm,n))
    return arm,end


def panel_path(arm,n,panel):
    return folder(arm)/f"eval_{n}_{panel}.json"


def eval_part(arm,n,panel,rep):
    base,reps=old.PANELS[panel]
    assert 0<=rep<reps
    out=folder(arm)/"parts"/f"{n}_{panel}_{rep}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        rows=final.run_episodes((str(folder(arm)),[(r,rep,base) for r in final.ROUTES],True,
                                 policy_path(arm,n),False,False),record_commands=True,
                                policy_input_factory=lateral_transform if arm in LATERAL else None)
        save(out,rows)
    return read(out)


@lru_cache(maxsize=1)
def groups():
    return GEOMETRY()


def measure(rows):
    need=[r for r in rows if groups()[r["route"],r["lane"]]=="needs_change"]
    def commands(a):
        return dict(left=sum(r["left_commands"] for r in a),right=sum(r["right_commands"] for r in a),
                    steps=sum(r["steps"] for r in a))
    return dict(n=len(rows),**final.tally(rows),required=len(need),required_arrived=sum(r["outcome"]=="arrival" for r in need),
                commands=commands(rows),required_commands=commands(need))


def combine(arm,n,panel):
    dest=panel_path(arm,n,panel)
    rows=[r for i in range(old.PANELS[panel][1]) for r in read(folder(arm)/"parts"/f"{n}_{panel}_{i}.json")]
    assert [r["seed"] for r in rows]==final.planned(*old.PANELS[panel])
    baseline=read(old.data_path("R5",0,panel))
    old.paired(baseline,rows)
    save(dest,dict(metrics=measure(rows),rows=rows))


def marks(end):
    return range(MARK,end+1,MARK)


def ranking(arm,n):
    x=read(panel_path(arm,n,"screen"))["metrics"]
    return x["arrival"],-x["collision"],x["required_arrived"],-n


def best(arm,end):
    return max(marks(end),key=lambda n:ranking(arm,n))


def block(pool,train_jobs,eval_jobs):
    # A single four-process pool executes both PPO and SUMO. An evaluation shard is sixteen routes.
    futures={}
    pending=Counter()
    def queue(arm,n,panel):
        if panel_path(arm,n,panel).exists():
            return
        key=arm,n,panel
        for i in range(old.PANELS[panel][1]):
            futures[pool.submit(eval_part,arm,n,panel,i)]=("eval",key)
            pending[key]+=1
    for arm,n in train_jobs:
        if policy_path(arm,n).exists():
            queue(arm,n,"screen")
        else:
            futures[pool.submit(train,arm,n)]=("train",(arm,n))
    for arm,n,panel in eval_jobs:
        queue(arm,n,panel)
    while futures:
        fut=next(as_completed(futures))
        kind,key=futures.pop(fut)
        fut.result()
        if kind=="train":
            queue(*key,"screen")
        else:
            pending[key]-=1
            if not pending[key]:
                combine(*key)
                report()
                print(*key,read(panel_path(*key))["metrics"],flush=True)


def predictor_stats(arm,end):
    f=folder(arm)
    logs=[json.loads(line) for line in (f/"updates.jsonl").read_text().splitlines()]
    eps=[json.loads(line) for line in (f/"episodes.jsonl").read_text().splitlines()]
    selected=[e for e in eps if e["step"]<=end]
    vals=[r["predictor"]["online_validation"] for r in logs if r["step"]<=end and r["predictor"]]
    vals=[v for v in vals if v]
    n=sum(v["n"] for v in vals)
    pairs=sum(v["positive"]*(v["n"]-v["positive"]) for v in vals)
    return dict(n=len(selected),exp_sum=sum(e["r_exp"] for e in selected),states=n,
                correct=sum(v["correct"] for v in vals),pairs=pairs,
                auc=sum(v["auc"]*v["positive"]*(v["n"]-v["positive"]) for v in vals if v["auc"] is not None)/pairs if pairs else None)


def report():
    protocol=read(ROOT/"protocol.json")
    win=read(ROOT/"winner.json")["arm"] if (ROOT/"winner.json").exists() else None
    stages=[(a,300000) for a in ARMS]+([(win,600000)] if win else [])
    records=[]; timeline=[]; causes=[]
    for arm,end in stages:
        for n in marks(end):
            if end==600000 and n<=300000:continue
            path=panel_path(arm,n,"screen")
            if path.exists():
                timeline.append((arm,n,read(path)["metrics"]))
        if not all(panel_path(arm,n,"screen").exists() for n in marks(end)): continue
        n=best(arm,end)
        rec=dict(arm=arm,end=end,selected=n,screen=read(panel_path(arm,n,"screen"))["metrics"])
        if all(panel_path(arm,n,p).exists() for p in old.PANELS if p!="screen"):
            blocks={p:read(panel_path(arm,n,p)) for p in old.PANELS if p!="screen"}
            rows=[r for b in blocks.values() for r in b["rows"]]
            rec["hold"]=measure(rows)
            rec["panels"]={p:b["metrics"] for p,b in blocks.items()}
            need=[r for p in ("designated","additional") for r in blocks[p]["rows"]]
            rec["required_512"]=measure(need)
            assert len(set(r["seed"] for r in rows))==576 and rec["required_512"]["required"]==208
            fails=Counter(r["route"] for r in rows if r["outcome"]!="arrival")
            causes.append((arm,end,fails.most_common(3)))
        rec["predictor"]=predictor_stats(arm,n)
        records.append(rec)
    def table(head,rows):
        return ["","| "+" | ".join(head)+" |","|"+"|".join("---" for _ in head)+"|",
                *["| "+" | ".join(map(str,row))+" |" for row in rows],""]
    def rate(m,k):
        return f"{m[k]}/{m['n']} ({m[k]/m['n']:.1%})"
    def scores(m):
        return [rate(m,k) for k in ("arrival","collision","off_route","stall","timeout")]
    doc=["# 순수 PPO 여섯 조건과 60만 스텝 연장","",
         "자가 주행 경험만으로 얻은 분기 보상이 올바른 경로 선택을 늘리는지 동일한 출발에서 비교했다. 최종 제출 후보는 보강형 DAgger R5다.",
         "",f"완료한 고교통 중간 평가 {len(timeline)}/42개. 576회 홀드아웃을 집계한 모델 {sum('hold' in r for r in records)}/7개.",
         "", "## 선정 모델 비교"]
    source_rows=[r for panel in old.PANELS if panel!="screen" for r in read(old.data_path("R5",0,panel))]
    original=final.tally(source_rows)
    assert len(source_rows)==576 and original["arrival"]==561 and original["collision"]==5
    doc+=["",f"같은 홀드아웃의 보존 모델 DAgger R5는 완주 {original['arrival']}/576 "
          f"({original['arrival']/576:.2%}), 충돌 {original['collision']}/576 ({original['collision']/576:.2%})였다. "
          "아래 순수 PPO는 모두 동일한 무작위 초기 가중치에서 출발했다."]
    completed={r["arm"]:r for r in records if r["end"]==300000 and "hold" in r}
    if len(completed)==6:
        top=max(completed.values(),key=lambda r:(r["hold"]["arrival"],-r["hold"]["collision"]))
        doc+=["",f"30만 이내 선정 모델 중 홀드아웃 완주가 가장 많았던 조건은 **{top['arm']}**로 "
              f"{top['hold']['arrival']}/576 완주, {top['hold']['collision']}/576 충돌이었다."]
        for key,other in (("P0","P1"),("P4","P5")):
            a,b=completed[key]["hold"],completed[other]["hold"]
            doc.append(f"- {key}→{other}의 자가 경험 보상 대조에서 완주는 {a['arrival']}→{b['arrival']}, "
                       f"충돌은 {a['collision']}→{b['collision']}, 이탈은 {a['off_route']}→{b['off_route']}건이었다.")
        all_goals=any(r["hold"]["arrival"]/576>=.95 and r["hold"]["collision"]/576<.05 for r in completed.values())
        doc+=["",f"6개 조건 중 완주율 95% 이상·충돌률 5% 미만을 동시에 충족한 조건은 {'있었다' if all_goals else '없었다'}."]
    cells=[]
    for r in records:
        for key in ("screen","hold"):
            if key in r:
                cells.append([r["arm"],r["end"],r["selected"],"선정 128회" if key=="screen" else "홀드아웃 576회",*scores(r[key])])
    doc+=table(["조건","학습 한도","선정 스텝","패널","완주","충돌","이탈","끝 정지","일반 시간초과"],cells)
    doc+=["## 팀원 제안 대조","", "같은 576개 출발에서 후보의 건수에서 대조군의 건수를 뺐다. 온라인 예측 정확도는 새로 판정된 접근 상태의 학습 전 예측 기준이며, AUC는 같은 업데이트 안의 양성·음성 쌍 수로 가중했다."]
    rows=[]; comparisons={}; transitions={}
    for left,right in (("P0","P1"),("P0","P2"),("P1","P2"),("P2","P3"),("P2","P4"),("P4","P5")):
        pair=[next((r for r in records if r["arm"]==a and r["end"]==300000 and "hold" in r),None) for a in (left,right)]
        if any(r is None for r in pair):continue
        delta={k:pair[1]["hold"][k]-pair[0]["hold"][k] for k in ("arrival","off_route","collision")}
        comparisons[left+"→"+right]=delta
        trips=[[trip for panel in old.PANELS if panel!="screen"
               for trip in read(panel_path(r["arm"],r["selected"],panel))["rows"]] for r in pair]
        switches=Counter((a["outcome"],b["outcome"]) for a,b in zip(*trips))
        transitions[left+"→"+right]=dict(**old.paired(*trips),matrix={f"{a}->{b}":v for (a,b),v in switches.items()})
        rows.append([left+" → "+right,*[f"{x:+d}" for x in delta.values()]])
    doc+=table(["대조 → 후보","완주 증감","이탈 증감","충돌 증감"],rows)
    doc+=table(["같은 출발의 완주 전환","양쪽 완주","기존 완주 상실","새 완주"],
               [[k,v["kept"],v["lost"],v["gained"]] for k,v in transitions.items()])
    rows=[]
    for r in records:
        s=r["predictor"]
        rows.append([r["arm"],r["selected"],s["n"],f"{s['exp_sum']:+.2f}",f"{s['exp_sum']/max(1,s['n']):+.4f}",
                     f"{s['correct']/s['states']:.1%}" if s["states"] else "—",
                     f"{s['auc']:.3f}" if s["auc"] is not None else "—"])
    doc+=table(["조건","선정 스텝","종료한 학습 주행","경험 보상 합","주행당 평균","온라인 정확도","AUC"],rows)
    rows=[]
    for r in records:
        if "required_512" not in r: continue
        x=r["required_512"]; c=x["required_commands"]
        commanded=c["left"]+c["right"]
        mix=f"{c['left']/commanded:.1%} / {c['right']/commanded:.1%}" if commanded else "명령 0회"
        rows.append([r["arm"],r["end"],f"{x['required_arrived']}/208 ({x['required_arrived']/208:.1%})",
                     c["left"],c["right"],f"{c['left']/c['steps']:.1%}",f"{c['right']/c['steps']:.1%}",mix])
    doc+=["## 변경이 필요한 출발","", "지정·추가 256회에서 첫 분기 전 변경이 필요한 208회를 집계했다. 좌·우 비율 분모는 해당 주행의 전체 행동 스텝이다."]
    doc+=table(["조건","학습 한도","완주","좌 명령","우 명령","좌 / 전체 스텝","우 / 전체 스텝","변경 명령 중 좌 / 우"],rows)
    doc+=["## 5만 스텝 간격 중간 평가"]
    doc+=table(["조건","스텝","완주","충돌","이탈","끝 정지","일반 시간초과","10초 대기","변경 필요 완주","좌 명령","우 명령"],
               [[a,n,*scores(m),m["slow10"],f"{m['required_arrived']}/{m['required']}",m["commands"]["left"],m["commands"]["right"]] for a,n,m in timeline])
    doc+=["## 실험 조건 및 해석","",
          "- P0은 공통 보상, P1은 P0에 자가 경험 포텐셜 차분을 추가했다. P2는 부적합 출발에서 준비 1점을 선지급하며 통과 시 차감한다. 처음부터 적합한 차로에 있으면 정상 통과 시 전체 2.5점이 지급된다.",
          "- P3는 마지막 분기 접근 도로에서 시작해 전 경로로 확장한다. 9번은 분기가 없어 원래 출발을 쓰고, 6·12번은 모든 차로가 연결된다. 분기가 있는 경로만 26회씩 반복하면 연결/변경 필요가 195:195다. 5천→8천→1만5천 대/시로 수요도 함께 바뀐다.",
          "- P4·P5는 도로 중심선의 횡방향으로 분기 좌우와 물리 차로 연결표를 함께 인코딩한다. 3천→8천→1만5천 대/시로 바뀌고, P4는 준비 선지급, P5는 자가 경험 보상이다. 두 커리큘럼의 경계는 8만 및 16만 스텝이다.",
          "- 공통 수치는 전진 +6, 분기 +2.5, 완주 +10, 충돌 −12, 이탈·끝 정지·시간초과 −8, 앞이 빈 저속 −0.02/스텝, 50스텝 연속 시 −9 종료다. 평가 시 학습용 10초 종료는 껐다. 동일 시드 11에서 256×256 ReLU, 분리 가치망, γ 0.995, λ 0.95, Adam 1e−4, 10에포크, 미니배치 64를 사용했다.",
          "- 모델 선정은 지정 128회에서 완주 수, 충돌 수, 변경 필요 출발 완주 수 순이다. 동률은 이른 스텝과 조건 번호 순으로 정했다. 60만 연장은 우승 조건 30만의 정책·가치망·옵티마이저·경험 버퍼에서 이어갔다. 60만 최선은 5만~60만 후보 전체에서 고른다.",
          "- 경험 보상 예측기는 자기 주행의 해당 분기 정상 통과·이탈·충돌·끝 정지로 라벨링한다. 일반 시간제한 및 학습용 저속 조기종료는 라벨에서 제외한다. 예측기는 성공·실패 상태 각각 200개가 쌓여 첫 학습을 끝낸 다음부터 보상에 사용한다. 클리핑과 분기 경계 초기화가 있는 차분이라 모든 왕복에 대한 보상 불변성은 보장되지 않는다.",
          "- 학습 교통 시드는 공통으로 205000000 + 에피소드 순번을 사용한다. 같은 순번은 같은 경로 편성이며 정책별 에피소드 길이에 따라 예산 안에서 실제로 사용한 시드 수는 달라진다. 평가 시드는 매 주행의 시작 관측 해시까지 대조했다.",
          "- 학습 롤아웃은 기본 2,048스텝이며 5만 저장 지점과 교통 전환에서는 남은 스텝만 처리한다. 교통 전환 시 진행 중 주행의 마지막 관측으로 가치를 부트스트랩한다. 모든 평가는 고교통, 16개 경로, 같은 물리적 분기와 시야 보정으로 진행했다.",
          "",f"동일 초기 가중치 SHA-256 `{protocol['initial_sha256']}`. 원본 R5 SHA-256 `{old.HASH['R5']}`."]
    if causes:doc+=table(["조건","한도","실패가 많은 경로 (건수)"],[[a,n,", ".join(f"{k}번 ({v})" for k,v in ls)] for a,n,ls in causes])
    if comparisons:
        doc+=["## 변경 효과"]
        for pair,d in comparisons.items():
            doc.append(f"- {pair}에서 완주 {d['arrival']:+d}건, 이탈 {d['off_route']:+d}건, 충돌 {d['collision']:+d}건이었다.")
        doc.append("- P0→P1은 경험 보상 단독 효과, P0→P2는 준비 선지급 단독 효과다. P4→P5는 동일한 커리큘럼·방향 관측 위에서 두 보상을 교체한 효과다. P2→P3에는 출발 지점과 교통량, P2→P4에는 방향 인코딩과 교통량이 함께 바뀐다. 수치는 학습 시드 11에 대한 결과다.")
        for key in ("P0→P1","P4→P5"):
            if key in comparisons:
                d=comparisons[key]
                achieved=d["arrival"]>0 and d["off_route"]<0 and d["collision"]<=0
                doc.append(f"- {key}의 자가 경험 보상이 완주 증가·이탈 감소·충돌 유지 또는 감소를 동시에 "
                           f"달성했는지에 대한 판정은 **{'충족' if achieved else '미달'}**이다.")
    extra=None
    if win:
        doc+=["",f"자동 연장 조건은 **{win}**이다."]
        extension=list(range(350000,600001,MARK))
        if all(panel_path(win,n,"screen").exists() for n in extension):
            n=max(extension,key=lambda v:ranking(win,v))
            if all(panel_path(win,n,p).exists() for p in old.PANELS):
                held=[r for p in old.PANELS if p!="screen" for r in read(panel_path(win,n,p))["rows"]]
                screened=read(panel_path(win,n,"screen"))["metrics"]
                extra=dict(arm=win,selected=n,screen=screened,hold=measure(held))
                doc+=["","## 연장 구간에서 새로 학습한 모델",
                      "35만~60만의 여섯 체크포인트에 같은 선정 순위를 적용했다. 연장 전 체크포인트가 전체 60만 구간 1위를 유지한 경우에도 새로 학습한 정책을 같은 576회에서 비교한다."]
                doc+=table(["조건","선정 스텝","패널","완주","충돌","이탈","끝 정지","일반 시간초과"],
                           [[win,n,label,*scores(extra[k])] for label,k in [("128회","screen"),("576회","hold")]])

    doc+=["","[실행 코드](../../utils/jys3_pure_ppo_overnight.py) · [전체 집계](../../results/jys3_pure_ppo_overnight/summary.json)"]
    tmp=REPORT.with_suffix(".md.tmp")
    formatted=[]
    for line in doc:
        if formatted and formatted[-1] and (formatted[-1].startswith("#") or (line.startswith("- ") and not formatted[-1].startswith("- "))):
            formatted.append("")
        if line or (formatted and formatted[-1]):
            formatted.append(line)
    tmp.write_text("\n".join(formatted)+"\n",encoding="utf8")
    tmp.replace(REPORT)
    save(ROOT/"summary.json",dict(winner=win,models=records,comparisons=comparisons,paired_transitions=transitions,extension_only=extra,screens=len(timeline)))


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(exist_ok=True,parents=True)
    initial_model=agent().policy
    initial=digest(initial_model)
    initial_path=ROOT/"initial.pt"
    if not initial_path.exists():
        final.atomic_torch(dict(model=initial_model.state_dict(),seed=11,sha256=initial),initial_path)
    initial_model.load_state_dict(torch.load(initial_path,weights_only=True)["model"])
    assert digest(initial_model)==initial
    protocol=ROOT/"protocol.json"
    if not protocol.exists():
        design=old.preflight()
        design.update(initial_sha256=initial,training_seed_base=205000000,training_seed_ranges=[[205000000,205250000]],
                      arms=list(ARMS),workers=4,commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      steps=[300000,600000],screen_interval=MARK,lr=.0001,beta=0.,lane_entropy_coef=.002,
                      seed=11,critic="independent trunk",prep=1.,exp_alpha=1.,exp_clip=.2,traffic_switches=[80000,160000])
        design["rewards"].update(collision=-12,clear_stop=-9)
        save(protocol,design)
    assert read(protocol)["initial_sha256"]==initial
    old.check_sources()
    report()
    with ProcessPoolExecutor(4) as pool:
        for n in marks(300000):
            block(pool,[(a,n) for a in ARMS],[])
        selections={a:best(a,300000) for a in ARMS}
        winner=max(ARMS,key=lambda a:ranking(a,selections[a]))
        save(ROOT/"winner.json",dict(arm=winner,selected_at_300k=selections,continue_from=300000))
        for n,locked_arm in zip(range(350000,600001,MARK),ARMS):
            locked=[(locked_arm,selections[locked_arm],p) for p in old.PANELS if p!="screen"]
            block(pool,[(winner,n)],locked)
        selections[("extension",winner)]=best(winner,600000)
        candidates=list(dict.fromkeys((a if isinstance(a,str) else a[1],n) for a,n in selections.items()))
        block(pool,[],[(a,n,p) for a,n in candidates for p in old.PANELS if p!="screen"])
    extension_check()
    assert len(read(ROOT/"summary.json")["models"])==7 and all("hold" in r for r in read(ROOT/"summary.json")["models"])
    save(ROOT/"finished.json",dict(winner=winner,protected_sha256=old.check_sources(),workers=4))


def extension_check():
    win=read(ROOT/"winner.json")["arm"]
    n=max(range(350000,600001,MARK),key=lambda v:ranking(win,v))
    missing=[(win,n,p) for p in old.PANELS if not panel_path(win,n,p).exists()]
    if missing:
        with ProcessPoolExecutor(4) as pool:
            block(pool,[],missing)
    report()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",action="store_true",help="Rebuild the tables from saved evaluations.")
    parser.add_argument("--extension-check",action="store_true",help="Verify the best post-300k policy on matched holdouts.")
    args=parser.parse_args()
    if args.extension_check:
        extension_check()
    elif args.report:
        report()
    else:
        main()

