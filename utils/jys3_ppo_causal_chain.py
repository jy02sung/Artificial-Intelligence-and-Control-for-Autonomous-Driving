"""Matched pure PPO lineage and S3. Run python -m utils.jys3_ppo_causal_chain to resume."""
import hashlib
import math
import sys
import time
import subprocess
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree as ET

from utils import jys3_diagnose_antispam as diag

anti = diag.exp
q, p, old, final = anti.q, anti.p, anti.old, anti.final
torch, np, traci, read, save = diag.torch, diag.np, diag.traci, diag.read, diag.save
ROOT = anti.ROOT.parent / "jys3_ppo_causal_chain"
REPORT = anti.REPORT.parent / "jys3-ppo-causal-chain.md"
STAGES = [[("Link1",11), ("P0",22), ("P0",33)]] + [
    [(arm,22),(arm,33)] for arm in ("S3-A","S3-B","S3-C")]
HOLDS = anti.q.HOLDS


def dest(arm,seed):
    return ROOT/f"{arm}_s{seed}"


def marks(arm):
    return range(25000,150001,25000) if arm.startswith("S3") else range(50000,250001 if arm=="Link1" else 300001,50000)


def protective_hashes():
    folders=[old.ROOT/"C",p.folder("P0"),*(q.folder("Q1",s) for s in anti.SEEDS)]
    files=[f for directory in folders for f in directory.glob("*.pt")]+[old.ORIG["R5"]]
    assert len(files)>20 and all(anti.checkpoint(s) in files for s in anti.SEEDS)
    return {f.relative_to(ROOT.parent).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def invalid(edge,lane,count,cmd,obs):
    return bool(cmd and (obs[38]>.5 or (cmd==-1 and lane==0) or (cmd==1 and lane==count-1)))


def dangerous(m):
    return m["overlap"] or any(gap is not None and (gap<distance or
           (closing>.5 and gap/closing<1.5)) for gap,closing,distance in [
               (m["front_gap"],m["front_closing"],6.),(m["rear_gap"],m["rear_closing"],10.)])


def start_trace(thunk,path):
    start=traci.start
    with patch.object(traci,"start",lambda args:start(args+["--lanechange-output",str(path)])):
        return thunk()


def finish_trace(env,path,last):
    end=traci.simulation.getTime()
    env.close()
    moves=[]
    for change in ET.parse(path).getroot().iter("change"):
        if change.get("id")==env.ego_id:
            step=last+1-round((end-float(change.get("time")))/env.sim.step_length)
            a,b=map(diag.split_lane,(change.get("from"),change.get("to")))
            if a[0]==b[0] and a[1]!=b[1]:
                moves.append((step,a,b))
    return moves


class Episodes(p.Episodes):
    def __init__(self,env,arm,path,*,training_seed,warm=False):
        super().__init__(env,arm,path,training_seed=training_seed)
        self.costed=arm.startswith("S3")
        self.risk=arm in ("S3-B","S3-C")
        self.xml=Path(env.sim.cfg_path).parent/"ego-changes.xml"
        self.stats=Counter()
        if self.costed:
            self.count=200000 if warm else 100000
            self.transform=p.lateral_transform(env)
        if self.risk:
            old.driving.COLLISION_REWARD=-16.

    def reset(self):
        reset=super().reset
        return start_trace(reset,self.xml) if self.risk else reset()

    def drive(self,action):
        if not self.costed:
            return super().drive(action)
        env=self.env
        cmd=env.sim._quantize_lane_change(action[1])
        edge,lane=env.snapshot[:2]
        count=env.counts.get(edge) or traci.edge.getLaneNumber(edge)
        obs=self.obs
        bad=invalid(edge,lane,count,cmd,obs)
        nxt,reward,done,cut,info=super().drive(action)
        charge=False
        if self.risk:
            direction=0
            if info["collided"]:
                last=len(self.actions)
                for t,a,b in finish_trace(env,self.xml,last):
                    if t==last:
                        direction=int(np.sign(b[1]-a[1]))
            elif env.snapshot and env.snapshot[0]==edge:
                direction=int(np.sign(env.snapshot[1]-lane))
            if direction:
                m=diag.measurements(obs,direction,env.visibility,env.vmax)
                charge=dangerous(m)
        self.stats.update(invalid_commands=int(bad),risk_moves=int(charge))
        return nxt,reward-.02*bad-1.5*charge,done,cut,info


def verified_update(agent,ep,adv,ret,ref=None):
    param=next(agent.policy.trunk.parameters())
    def count():
        return int(agent.optimizer.state.get(param,{}).get("step",0))
    before=count()
    metrics=agent.update(adv,ret,reference_policy=ref,reference_coef=.08 if ref is not None else 0.,
                         stratified_reference=ref is not None)
    actual=count()-before
    assert actual==10*math.ceil(len(ret)/64)
    metrics.update(minibatches=actual,lr=agent.optimizer.param_groups[0]["lr"])
    return metrics


def train_chunk(arm,seed,step):
    torch.set_num_threads(1)
    path=dest(arm,seed)
    path.mkdir(exist_ok=True,parents=True)
    if (path/f"policy_{step}.pt").exists():
        return
    ref=None
    if arm.startswith("S3"):
        if not (path/"warm.pt").exists():
            anti.warmup(seed,arm=arm,dest=path,episode_type=Episodes)
        digest=read(path/"warm_check.json")["sha256"]
        def build():
            a=p.agent(seed)
            a.policy.load_state_dict(torch.load(path/"warm.pt",weights_only=True))
            return a
        if arm=="S3-C":
            ref=anti.load_peak(seed).policy.eval().requires_grad_(False)
    else:
        build=partial(p.agent,seed)
        digest=read(q.folder("Q1",seed)/"initial.json")["sha256"]
    def update(a,ep,adv,ret):
        if arm.startswith("S3"):
            for group in a.optimizer.param_groups:
                group["lr"]=3e-5-2e-5*max(0,a.num_timesteps-a.n_steps)/(150000-a.n_steps)
            a.lane_entropy_coef=.0005
        m=verified_update(a,ep,adv,ret,ref)
        m.update(ep.stats)
        ep.stats.clear()
        return m
    p.train(arm,step,directory=path,make_agent=build,make_episodes=Episodes,expected_hash=digest,
            training_seed=seed,update=update)


class OriginalC(old.TrainingEpisodes):
    """Retain the exact C rollout boundaries, with active-episode replay for recovery."""
    def reset(self):
        self.actions=[]
        obs,info=super().reset()
        self.obs=obs
        return obs,info

    def step(self,action):
        self.actions.append(action.copy())
        self.obs,r,d,c,i=super().step(action)
        return self.obs,r,d,c,i


def train_link():
    torch.set_num_threads(1)
    path=dest("Link1",11)
    path.mkdir(exist_ok=True,parents=True)
    a=p.agent(11)
    raw=old.PPO(50,2,hidden_sizes=(256,256),activation="relu",seed=11,device="cpu",lr=1e-4,
                gamma=.995,gae_lambda=.95,n_steps=2048,n_epochs=10,minibatch_size=64,
                entropy_coef=0.,lane_entropy_coef=.002,lane_keep_bias=1.)
    for k,v in raw.policy.state_dict().items():
        assert torch.equal(v,a.policy.state_dict()[k])
    save(path/"initial.json",dict(original=p.digest(raw.policy),split=p.digest(a.policy),actor_value_head_identical=True))
    def snapshot(n):
        state=a.policy.state_dict()
        final.atomic_torch(dict(policy={k:v for k,v in state.items() if not k.startswith("value_trunk.")},
            value_trunk=a.policy.value_trunk.state_dict(),activation="relu",obs_dim=50,hidden_sizes=[256,256],step=n),path/f"policy_{n}.pt")
    def callback(n):
        if n in (50000,100000,150000):
            snapshot(n)
    with final.p.roll.baseline.simulator() as env:
        ep=OriginalC(env,"C","ppo",callback,root=ROOT)
        ep.folder=path
        stats=[0.,0]
        resume=path/"resume.pt"
        logs=[path/"updates.jsonl",path/"ppo_episodes.jsonl"]
        state=torch.load(resume,weights_only=False) if resume.exists() else None
        for f in logs:
            with f.open("ab") as out:
                out.truncate(state["offsets"][f.name] if state else 0)
        if state:
            a.policy.load_state_dict(state["policy"])
            a.optimizer.load_state_dict(state["opt"])
            a.num_timesteps=state["step"]
            ep.count=state["count"]-1
            ep.steps=a.num_timesteps-len(state["actions"])
            obs,_=ep.reset()
            ep.on_step=lambda n: None
            for action in state["actions"]:
                obs,_,d,c,_=ep.step(action)
                assert not(d or c)
            ep.on_step=callback
            assert np.array_equal(obs,state["obs"])
            ep.outcomes,ep.first_outcomes,ep.first_rewards=state["counters"]
            stats=state["stats"]
            torch.set_rng_state(state["torch"])
            np.random.set_state(state["np"])
        else:
            obs,_=ep.reset()
        while a.num_timesteps<250000:
            start=a.num_timesteps
            cutoff=200000 if start<200000 else 250000
            size=min(2048,cutoff-start)
            obs,last,_=a.collect_rollout(ep,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=a.buffer.compute_gae(last,a.gamma,a.gae_lambda)
            metrics=verified_update(a,ep,adv,ret)
            n=a.num_timesteps
            old.append(logs[0],dict(step=n,metrics=metrics))
            if n==cutoff:
                snapshot(n)
            if start//50000!=n//50000:
                final.atomic_torch(dict(policy=a.policy.state_dict(),opt=a.optimizer.state_dict(),step=n,
                    count=ep.count,actions=ep.actions,obs=obs,stats=stats,torch=torch.get_rng_state(),
                    np=np.random.get_state(),counters=(ep.outcomes,ep.first_outcomes,ep.first_rewards),
                    offsets={f.name:f.stat().st_size for f in logs}),resume)


def train(arm,seed):
    if arm=="Link1":
        train_link()
    else:
        for n in marks(arm):
            train_chunk(arm,seed,n)
    return arm,seed

def panel(arm,seed,n,name):
    return dest(arm,seed)/f"eval_{n}_{name}.json"


def evaluate(arm,seed,n,name,rep):
    path=dest(arm,seed)
    out=path/"parts"/f"{n}_{name}_{rep}.json"
    if out.exists():
        return
    out.parent.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    source=anti.checkpoint(seed) if arm=="Q1" else path/f"policy_{n}.pt"
    model=final.network(source)
    base=old.PANELS[name][0]
    rows=[]
    with final.p.roll.baseline.simulator() as env, torch.inference_mode(), TemporaryDirectory() as tmp:
        xml=Path(tmp)/"changes.xml"
        transform=p.lateral_transform(env) if arm.startswith("S3") or arm=="Q1" else lambda x:x
        for route in range(1,17):
            key=f"route_{route:02d}"
            lane=min(rep%2,env.counts[env.routes[key][0]]-1)
            env.ego_depart_lane=str(lane)
            seed_eval=base+rep*100+route
            obs,_=start_trace(lambda:env.reset(seed=seed_eval,options={"route":key}),xml)
            initial=hashlib.sha256(obs.tobytes()).hexdigest()
            left=right=bad=seen=0
            history=deque(maxlen=3)
            for t in range(1,1001):
                x=transform(obs[:50])
                action=model.det_action(torch.from_numpy(x).unsqueeze(0))[0].numpy()
                cmd=env.sim._quantize_lane_change(action[1])
                edge,i=env.snapshot[:2]
                count=env.counts.get(edge) or traci.edge.getLaneNumber(edge)
                left+=int(cmd==1)
                right+=int(cmd==-1)
                bad+=int(invalid(edge,i,count,cmd,x))
                history.append((t,x))
                obs,_,done,cut,info=env.step(action)
                if env.snapshot and env.snapshot[0]==edge and env.snapshot[1]!=i:
                    seen+=1
                if done or cut:
                    break
            collisions=diag.collision_events(env.ego_id) if info["collided"] else []
            moves=finish_trace(env,xml,t)
            event_cut=False
            if collisions:
                assert len(collisions)==1
                contact=diag.split_lane(collisions[0]["lane"])
                for tick,a,b in moves:
                    if t-tick<3 and b==contact:
                        m=diag.measurements(dict(history)[tick],int(np.sign(b[1]-a[1])),env.visibility,env.vmax)
                        event_cut|=m["tight"]
            assert bool(collisions)==(info["outcome"]=="collision")
            rows.append(dict(seed=seed_eval,rep=rep,route=route,lane=lane,initial_sha256=initial,steps=t,
                             outcome=info["outcome"],left_commands=left,right_commands=right,invalid=bad,
                             valid=left+right-bad,moves=len(moves),observed_moves=seen,cutin=int(event_cut)))
    save(out,rows)


def finish_panel(arm,seed,n,name):
    rows=[r for rep in range(old.PANELS[name][1]) for r in read(dest(arm,seed)/"parts"/f"{n}_{name}_{rep}.json")]
    ref=read(q.panel("Q1",seed,anti.PEAK.get(seed,100000),name))["rows"]
    old.paired(ref,rows)
    if arm=="Q1":
        assert all(all(a[k]==b[k] for k in ("outcome","steps","left_commands","right_commands")) for a,b in zip(ref,rows))
    assert [r["seed"] for r in rows]==final.planned(*old.PANELS[name])
    counts=Counter(r["outcome"] for r in rows)
    metrics={k:counts[k] for k in anti.OUT}
    metrics["required_arrived"]=sum(r["outcome"]=="arrival" and p.groups()[r["route"],r["lane"]]=="needs_change" for r in rows)
    save(panel(arm,seed,n,name),dict(rows=rows,metrics=metrics))


def evaluate_all(pool,keys):
    pending={}
    for key in sorted(set(keys)):
        if panel(*key).exists():
            continue
        for rep in range(old.PANELS[key[-1]][1]):
            pending[pool.submit(evaluate,*key,rep)]=key
    counts=Counter(pending.values())
    for fut in as_completed(pending):
        fut.result()
        key=pending[fut]
        counts[key]-=1
        if not counts[key]:
            finish_panel(*key)
            print("evaluated",*key,read(panel(*key))["metrics"]["arrival"],flush=True)


def selected(arm,seed):
    def rank(n):
        m=read(panel(arm,seed,n,"screen"))["metrics"]
        if arm=="P0":
            return m["arrival"],-m["collision"],m["required_arrived"],-n
        return m["arrival"],-m["collision"],-m["stall"]-m["timeout"],-n
    return max(marks(arm),key=rank)


def summary(blocks):
    rows=[r for block in blocks for r in block]
    assert len(rows)==len({r["seed"] for r in rows})==576
    needs=[r for block in blocks[1:] for r in block if p.groups()[r["route"],r["lane"]]=="needs_change"]
    assert len(needs)==208
    counts=Counter(r["outcome"] for r in rows)
    values={k:counts[k] for k in anti.OUT}
    values["required"]=sum(r["outcome"]=="arrival" for r in needs)
    for field in ("valid","invalid","moves","cutin"):
        if field in rows[0]:
            values[field]=sum(r[field] for r in rows)
    return values


def report():
    table={}
    for arm,seed,n,fn in [
        ("C",11,100000,lambda h:read(old.data_path("C",100000,h))),
        ("P0",11,250000,lambda h:read(p.panel_path("P0",250000,h))["rows"]),
        *(("Q1",s,n,lambda h,s=s,n=n:read(q.panel("Q1",s,n,h))["rows"]) for s,n in ((11,100000),(22,200000),(33,250000)))]:
        table[f"{arm}/{seed}"]=dict(step=n,**summary([fn(h) for h in HOLDS]))
    for arm,seed in [x for stage in STAGES for x in stage]+[("Q1",22),("Q1",33)]:
        if arm=="Q1":
            n=anti.PEAK[seed]
        elif all(panel(arm,seed,n,"screen").exists() for n in marks(arm)):
            n=selected(arm,seed)
        else:
            continue
        if all(panel(arm,seed,n,h).exists() for h in HOLDS):
            table[f"{arm}/{seed}"]=dict(step=n,**summary([read(panel(arm,seed,n,h))["rows"] for h in HOLDS]))
    save(ROOT/"summary.json",table)
    def fmt(n,den=576):
        return f"{n} ({100*n/den:.1f}%)"
    def draw(header,rows):
        return ["","| "+" | ".join(header)+" |","|"+"---|"*len(header),*["| "+" | ".join(map(str,row))+" |" for row in rows],""]
    doc=["# 순수 PPO 분리형 Critic과 컷인 보상 대조","",
         "각 모델은 같은 고교통 576회에서 평가했다. 선정 패널은 별도 128회다. 변경 필요 출발은 두 256회 패널 안의 208회다. 실제 이동은 SUMO 차로변경 출력으로 충돌 스텝까지 기록했다. 유효 명령은 지정한 경계·실선 무효 조건에 걸리지 않은 좌우 명령 수다.",
         "", "## Part 1 단계별 결과"]
    count=sum(f"{arm}/{s}" in table for stage in STAGES for arm,s in stage)
    doc.insert(2,f"신규 9런 중 576회 최종 검증 {count}/9 완료. 고교통 15,000대/시, 16경로, 수정 분기, 고가 20쌍 보정, 50차원 관측, 안전게이트 OFF 환경을 사용했다. 학습용 10초 조기 종료는 평가에서 껐다.")
    def row(key,m):
        return [key,m["step"],*[fmt(m[c]) for c in anti.OUT]]
    doc+=draw(["모델/시드","스텝","완주","충돌","이탈","끝 정지","시간초과"],[row(k,table[k]) for k in
        ("C/11","Link1/11","P0/11","P0/22","P0/33","Q1/11","Q1/22","Q1/33") if k in table])
    doc+=["C와 Link-1은 원래 C의 2,048스텝 롤아웃 경계와 20만 종료 업데이트를 맞췄다. P0는 5만마다 롤아웃을 잘라 업데이트한다. Link-1→P0에는 두 벌점과 이 업데이트 경계 차이가 함께 있다. Q1의 관측 교정은 34번 방향과 40~49번 도로 특성을 함께 변환한다."]
    matched=[]
    for n,baseline in [(100000,"C"),(250000,"P0")]:
        if all(panel("Link1",11,n,h).exists() for h in HOLDS):
            link=summary([read(panel("Link1",11,n,h))["rows"] for h in HOLDS])
            other=table[baseline+"/11"]
            first,last=(other,link) if baseline=="C" else (link,other)
            label="C → Link1" if baseline=="C" else "Link1 → P0"
            change="분리 Critic 단독, 같은 10만" if baseline=="C" else "두 벌점 및 롤아웃 경계, 같은 25만"
            matched.append([label,change,*[f"{100*(last[k]-first[k])/576:+.2f}" for k in ("arrival","collision","off_route")],"—"])
            doc.append(f"- 같은 {n:,}스텝의 {baseline}과 Link-1 완주 수는 {other['arrival']}건과 {link['arrival']}건, 충돌 수는 {other['collision']}건과 {link['collision']}건이었다.")
    doc+=["","## Part 2와 시드별 변화"]
    cols=["모델/시드","스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주 /208","유효 명령","무효 명령","실제 변경","컷인 충돌"]
    doc+=draw(cols,[row(key,table[key])+[fmt(table[key]["required"],208),*[table[key].get(c,"—") for c in ("valid","invalid","moves","cutin")]]
                    for s in (22,33) for arm in ("Q1","S3-A","S3-B","S3-C") if (key:=f"{arm}/{s}") in table])
    means=[]
    for arm,seeds in [("P0",(11,22,33)),("Q1",(11,22,33)),*((a,(22,33)) for a in ("Q1","S3-A","S3-B","S3-C"))]:
        if all(f"{arm}/{s}" in table for s in seeds):
            ms=[table[f"{arm}/{s}"] for s in seeds]
            means.append([arm,",".join(map(str,seeds)),*[f"{np.mean([m[k] for m in ms])*100/576:.2f}%" for k in anti.OUT],f"{np.mean([m['required'] for m in ms])*100/208:.2f}%",
                         *[f"{np.mean([m[k] for m in ms]):.1f}" if all(k in m for m in ms) else "—" for k in ("valid","invalid","moves","cutin")]])
    doc+=draw(["조건","시드","평균 완주율","충돌률","이탈률","끝 정지율","시간초과율","변경 필요 완주율","평균 유효","평균 무효","평균 변경","평균 컷인"],means)
    effects=matched.copy()
    for arm,base,mechanism,seeds in [
        ("Q1","P0","도로 단위 방향·도로 특성 교정",(11,22,33)),
        ("S3-A","Q1","무효 명령 비용과 추가 미세조정",(22,33)),
        ("S3-B","S3-A","컷인 −1.5 및 충돌 −12→−16",(22,33)),
        ("S3-C","S3-B","원본 Q1 정책 대비 90:10 Self-KL",(22,33))]:
        if all(f"{a}/{s}" in table for a in (arm,base) for s in seeds):
            def delta(k):
                return sum(table[f"{arm}/{s}"][k]-table[f"{base}/{s}"][k] for s in seeds)
            effects.append([base+" → "+arm,mechanism,f"{100*delta('arrival')/(576*len(seeds)):+.2f}",f"{100*delta('collision')/(576*len(seeds)):+.2f}",
                            f"{100*delta('off_route')/(576*len(seeds)):+.2f}",f"{delta('cutin'):+d}" if all('cutin' in table[f"{a}/{s}"] for a in (arm,base) for s in seeds) else "—"])
    doc+=["## 발표용 대조표"]+draw(["대조","변경 사항","완주 Δ%p","충돌 Δ%p","이탈 Δ%p","컷인 증감 합계"],effects)
    if all(f"S3-C/{s}" in table for s in anti.SEEDS):
        for s in anti.SEEDS:
            m,b=table[f"S3-C/{s}"],table[f"Q1/{s}"]
            doc.append(f"- 시드 {s}의 Q1 대비 완주 증가·충돌 감소 동시 달성 여부는 "+("달성" if m["arrival"]>b["arrival"] and m["collision"]<b["collision"] else "미달")+"이다.")
    doc+=["", "## 컷인과 무효 명령 대조", "",
          "컷인 충돌은 이전 진단과 같은 기준으로 집계했다. 충돌 포함 직전 3스텝 안에 같은 도로의 접촉 차로로 실제 이동했고 변경 전 관측의 동적 앞·뒤 간격 또는 겹침 기준을 충족한 사건이다. B·C 학습의 6m/10m/1.5초 기준은 보상 계산에 사용한다. 두 정의의 분모를 구분했다.",
          "이전 S2/22는 576회에서 이탈 228건, S2/33은 충돌 55건 중 컷인 49건이었다. 다음 표는 각 모델을 평가한 동일 교통 시드에서 명령 억제와 위험 변화를 함께 보여준다."]
    paired=[]
    for seed,traffic in ((22,104320713),(33,104320011)):
        for arm in ("Q1","S3-A","S3-B","S3-C"):
            key=f"{arm}/{seed}"
            if key not in table or "cutin" not in table[key]:
                continue
            m=table[key]
            row=next(r for r in read(panel(arm,seed,m["step"],"additional"))["rows"] if r["seed"]==traffic)
            paired.append([key,traffic,row["route"],row["outcome"],row["steps"],row["right_commands"],row["invalid"],row["moves"],m["off_route"],f"{m['cutin']} / {m['collision']}"])
    doc+=draw(["모델","교통 시드","경로","종료","스텝","우 명령","좌우 무효 합","실제 변경","전체 이탈 /576","전체 컷인 /충돌"],paired)
    doc.append("Q1 선택은 시드 11·22·33 모두 30만 이내에 일어났다. P0와 Q1의 시드별 비교는 같은 초기화와 교통 시드를 사용한다. S3-A/B/C는 매번 같은 Q1에서 독립 출발한다. 한 시드에서는 새 보상 구성이 방문 상태와 그 이후 수집 데이터를 바꿀 수 있다.")
    doc+=["","## 중간 선정 패널 128회"]
    checkpoints=[]
    for stage in STAGES:
        for arm,s in stage:
            for step in marks(arm):
                f=panel(arm,s,step,"screen")
                if f.exists():
                    m=read(f)["metrics"]
                    checkpoints.append([arm,s,step,*[fmt(m[k],128) for k in anti.OUT],m["required_arrived"]])
    doc+=draw(["런","시드","스텝","완주","충돌","이탈","끝 정지","시간초과","변경 필요 완주 /52"],checkpoints)
    doc+=["","[실험 코드](../../utils/jys3_ppo_causal_chain.py) · [결과 JSON](../../results/jys3_ppo_causal_chain/summary.json)"]
    tmp=REPORT.with_suffix(".tmp")
    tmp.write_text("\n".join(doc)+"\n",encoding="utf8")
    tmp.replace(REPORT)


def main():
    ROOT.mkdir(exist_ok=True)
    hashes=protective_hashes()
    manifest=ROOT/"protected_start.json"
    if manifest.exists():
        assert hashes==read(manifest)
    else:
        save(manifest,hashes)
    assert len(old.CROSSINGS)==20 and hashes[old.ORIG["R5"].relative_to(ROOT.parent).as_posix()]==old.HASH["R5"]
    save(ROOT/"protocol.json",dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         stages=STAGES,panels=old.PANELS,workers=4,epochs=10,target_kl=None,training_vph=15000,
         seeds=anti.SEEDS,source_step=anti.PEAK,split_critic=True,observation=50,actor="256x256 ReLU",
         invalid=-.02,risk=-1.5,collision=[-12,-16],reference_kl=.08))
    with ProcessPoolExecutor(4) as pool:
        for stage in STAGES:
            jobs={pool.submit(train,arm,s):(arm,s) for arm,s in stage}
            for fut in as_completed(jobs):
                arm,s=fut.result()
                evaluate_all(pool,[(arm,s,n,"screen") for n in marks(arm)])
                choices={selected(arm,s)}
                if arm=="Link1":
                    choices.update((100000,250000))
                evaluate_all(pool,[(arm,s,n,h) for n in choices for h in HOLDS])
                report()
                print("completed",arm,s,flush=True)
            if stage==STAGES[0]:
                evaluate_all(pool,[("Q1",s,anti.PEAK[s],h) for s in anti.SEEDS for h in HOLDS])
                report()
    assert protective_hashes()==hashes
    save(ROOT/"protected_end.json",hashes)
    save(ROOT/"finished.json",dict(completed=9,hashes_match=True))
    report()


def final_audit():
    assert read(ROOT/"finished.json")["completed"]==9
    assert protective_hashes()==read(ROOT/"protected_start.json")==read(ROOT/"protected_end.json")
    checked=[]
    for stage in STAGES:
        for arm,seed in stage:
            path=dest(arm,seed)
            if arm.startswith("S3"):
                assert read(path/"warm_check.json")["actor_bits_identical"]
            updates=[p.json.loads(line) for line in (path/"updates.jsonl").read_text().splitlines()]
            previous=0
            for u in updates:
                gap=u["step"]-previous
                assert 0<gap<=2048 and u["metrics"]["minibatches"]==10*math.ceil(gap/64)
                previous=u["step"]
            assert previous==max(marks(arm))
            for n in marks(arm):
                assert (path/f"policy_{n}.pt").is_file()
                assert len(read(panel(arm,seed,n,"screen"))["rows"])==128
            n=selected(arm,seed)
            blocks=[read(panel(arm,seed,n,h))["rows"] for h in HOLDS]
            m=summary(blocks)
            assert sum(m[k] for k in anti.OUT)==576 and 0<=m["cutin"]<=m["collision"]
            for block in blocks:
                assert all(r["valid"]+r["invalid"]==r["left_commands"]+r["right_commands"] for r in block)
            checked.append(dict(arm=arm,seed=seed,selected=n,updates=len(updates),steps=previous,holdout=m))
    for seed in anti.SEEDS:
        for h in HOLDS:
            rows=read(panel("Q1",seed,anti.PEAK[seed],h))["rows"]
            archived=read(anti.folder("Q1",seed)/f"eval_{anti.PEAK[seed]}_{h}.json")["rows"]
            old.paired(archived,rows)
            assert all(a["within_edge_moves"]==b["observed_moves"] and a["outcome"]==b["outcome"]
                       and a["steps"]==b["steps"] for a,b in zip(archived,rows))
    assert sum(r["cutin"] for h in HOLDS for r in read(panel("Q1",22,200000,h))["rows"])==32
    report()
    save(ROOT/"final_audit.json",dict(runs=checked,original_hashes_verified=True,q1_replayed=1152,
                                      cutin_consistent=True))


def early_screens():
    # The main runner occupies only three slots until the first full training job completes.
    # Release this fourth slot well before that at the first 200k checkpoint.
    stage=STAGES[0]
    def reserved():
        return any((dest(arm,s)/"policy_200000.pt").exists() for arm,s in stage)
    while not reserved():
        advanced=False
        for n in (50000,100000):
            for arm,s in stage:
                if panel(arm,s,n,"screen").exists() or not (dest(arm,s)/f"policy_{n}.pt").exists():
                    continue
                for rep in range(old.PANELS["screen"][1]):
                    if reserved():
                        return
                    evaluate(arm,s,n,"screen",rep)
                finish_panel(arm,s,n,"screen")
                report()
                advanced=True
        if not advanced:
            time.sleep(10)


if __name__=="__main__":
    if sys.argv[1:]==["--early-screens"]:
        early_screens()
    elif sys.argv[1:]==["--audit-final"]:
        final_audit()
    else:
        main()
