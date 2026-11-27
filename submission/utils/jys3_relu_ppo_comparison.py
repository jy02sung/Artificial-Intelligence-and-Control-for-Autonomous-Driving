"""Matched ReLU PPO runs in the final 16-route simulator. Run as a module."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("IC_LIBSUMO", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

from algorithms.ppo import PPO
from env.toegyewon_crossings import CROSSINGS
from utils import jys3_final_relu_dagger as final
from utils import jys3 as driving

ROOT = final.ROOT.parent / "jys3_relu_ppo_comparison"
ORIG = {"BC": final.checkpoint(0), "R5": final.checkpoint(5)}
HASH = {"BC":"61f77baac57252c3adccea5af9e2855f135c0237a14f8458ea6fc9d7ab495e7e",
        "R5":"fb3a775fa4d4ccb56ab1c414b9e05af1a99ebf0623c992e096b063f2967728cd"}
SOURCE = {"A":"BC", "B":"R5"}
RATE = {"A":2e-5,"B":1e-5,"C":1e-4}
BETA = {"A":.5,"B":1.,"C":0.}
PANELS = {"screen":(95380000,8),"hold64":(81260100,4),
          "designated":(91000000,16),"additional":(104320000,16)}
ORIGINAL_ROWS = {"BC":{"screen":"screen0","hold64":"hold64_0"},
                 "R5":{"screen":"screen5","hold64":"hold64_5","designated":"final256","additional":"novel256"}}
WARM = 20000
MARK = 8192
read = final.p.read
save = final.p.save


def check_sources():
    hashes = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in ORIG.items()}
    assert hashes == HASH, hashes
    return hashes


def path_for(arm, step):
    return ORIG[arm] if arm in ORIG else ROOT/arm/f"policy_{step}.pt"


def data_path(arm, step, panel):
    return ROOT/"evaluation"/f"{arm}_{step}_{panel}.json"


def episode_bases(arm, phase):
    return 200000000 + "ABC".index(arm)*2000000 + (1000000 if phase=="ppo" else 0)


def preflight():
    existing=set()
    for f in (list(final.ROOT.glob("*/episodes.json"))+
              list(final.ROOT.glob("R*/*/episodes.json"))):
        existing.update(r["seed"] for r in read(f))
    for base,reps in PANELS.values():
        existing.update(final.planned(base,reps))
    ranges=[]
    for arm in RATE:
        for phase in ("warm","ppo") if arm in SOURCE else ("ppo",):
            first=episode_bases(arm,phase)
            assert not any(first <= seed < first+250000 for seed in existing)
            ranges.append((first,first+250000))
    assert all(b<=c for (_,b),(c,_) in zip(sorted(ranges),sorted(ranges)[1:]))
    assert len(CROSSINGS)==20
    assert final.tally(read(final.ROOT/"screen0/episodes.json"))["arrival"]==107
    assert final.tally(read(final.ROOT/"screen5/episodes.json"))["arrival"]==125
    return dict(hashes=check_sources(),protected_seeds=len(existing),training_seed_ranges=ranges,
                panels=PANELS,lr=RATE,beta=BETA,gamma=.995,gae_lambda=.95,n_steps=2048,
                n_epochs=10,minibatch_size=64,physical_forks=True,crossings=20,obs_dim=50,
                demand_vph=15000,safety_gate=False,activation="relu",hidden=[256,256],
                rewards=dict(progress_cap=6,fork=2.5,arrival=10,collision=-10,off_route=-8,
                             stall=-8,timeout=-8,idle_step=-.02,clear_stop=-7))


def append(path, record):
    with path.open("a",encoding="utf8") as f:
        print(json.dumps(record,allow_nan=False),file=f)


class TrainingEpisodes:
    def __init__(self, env, arm, phase, on_step, *, root=ROOT, clear_penalty=-7.):
        self.env,self.arm,self.phase,self.on_step=env,arm,phase,on_step
        self.base=episode_bases(arm,phase)
        self.count=self.steps=0
        self.folder=root/arm
        self.clear_penalty=clear_penalty
        self.outcomes=Counter()
        self.first_rewards=Counter()
        self.first_outcomes=Counter()
        env.progress_cap=6.
        env.off_route_penalty=-8.
        driving.ARRIVAL_REWARD,driving.PASS_REWARD=10.,2.5
        driving.COLLISION_REWARD,driving.TIMEOUT_REWARD=-10.,-8.
        assert env.physical_forks and env.occlusion and not env.lc_safety_enabled
        assert env.max_steps==1000 and env.step_length==.2 and len(env.routes)==16

    def reset(self):
        block,index=divmod(self.count,16)
        # Matched route order and departure lanes, distinct SUMO seeds across conditions.
        order=np.random.default_rng(11+block).permutation(16)+1
        self.route=int(order[index])
        path=f"route_{self.route:02d}"
        self.env.ego_depart_lane=str(min(block%2,self.env.counts[self.env.routes[path][0]]-1))
        self.seed=self.base+self.count
        assert self.count<250000
        self.count+=1
        obs,info=self.env.reset(seed=self.seed,options={"route":path})
        self.idle=0
        self.episode_return=0.
        assert obs.shape==(52,) and not obs[50:].any()
        return obs[:50],info

    def step(self,action):
        env=self.env
        obs,reward,done,cut,info=env.step(action)
        self.steps+=1
        clear=(info.get("speed") is not None and info["speed"]<2. and
               (info["gap"] is None or info["gap"]>25))
        self.idle=self.idle+1 if clear else 0
        # Physical terminal events take precedence. Charge per-step idle when observed.
        if clear:
            reward-=.02
            env.sums["idle_clear"]-=.02
        if self.idle>=50 and not (done or cut):
            reward+=self.clear_penalty
            env.sums["clear_stop"]+=self.clear_penalty
            done=True
            info["outcome"]="clear_stop"
        self.episode_return+=reward
        if self.arm=="C" and self.steps<=50000:
            parts={"arrival":10.*info["arrived"],"fork":2.5*info["fork_passed"],
                   "progress":info["progress_bonus"],"collision":-10.*info["collided"],
                   "off_route":-8.*info["off_route"],"timeout_stall":info["timeout_reward"],
                   "idle_clear":-.02*clear,"clear_stop":self.clear_penalty*(info["outcome"]=="clear_stop")}
            assert np.isclose(sum(parts.values()),reward)
            self.first_rewards.update(parts)
            if done or cut:
                self.first_outcomes[info["outcome"]]+=1
            if self.steps==50000:
                save(self.folder/"early_50k.json",dict(steps=self.steps,outcomes=dict(self.first_outcomes),
                     rewards=dict(self.first_rewards),return_sum=sum(self.first_rewards.values())))
        if done or cut:
            self.outcomes[info["outcome"]]+=1
            append(self.folder/f"{self.phase}_episodes.jsonl",
                   dict(step=self.steps,seed=self.seed,route=self.route,outcome=info["outcome"],
                        episode_return=self.episode_return))
        self.on_step(self.steps)
        return obs[:50],reward,done,cut,info


def store_checkpoint(agent,arm,step):
    final.atomic_torch(dict(policy=agent.policy.state_dict(),activation="relu",
                            obs_dim=50,hidden_sizes=[256,256],step=step),path_for(arm,step))


def warm_critic(agent, reference, arm):
    for name,param in agent.policy.named_parameters():
        param.requires_grad_(name.startswith("v_head."))
    optimizer=torch.optim.Adam(agent.policy.v_head.parameters(),lr=2e-5)
    with final.p.roll.baseline.simulator() as env:
        episodes=TrainingEpisodes(env,arm,"warm",lambda _:None)
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<WARM:
            size=min(2048,WARM-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            _,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            states,*_=agent.buffer.to_tensors(agent.device)
            with torch.no_grad():
                features=agent.policy.trunk(states)
            target=torch.as_tensor(ret)
            def mse(ids):
                return (agent.policy.v_head(features[ids]).flatten()-target[ids]).square().mean()
            before=mse(slice(None)).item()
            for _ in range(10):
                for ids in torch.randperm(size).split(64):
                    loss=mse(ids)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy.v_head.parameters(),.5)
                    optimizer.step()
            append(ROOT/arm/"warm_updates.jsonl",dict(step=agent.num_timesteps,mse_before=before,
                                                       mse_after=mse(slice(None)).item()))
    for k,v in agent.policy.state_dict().items():
        if not k.startswith("v_head."):
            assert torch.equal(v.view(torch.uint8),reference.state_dict()[k].view(torch.uint8)),k
    assert not torch.equal(agent.policy.v_head.weight,reference.v_head.weight)
    agent.policy.requires_grad_(True)
    save(ROOT/arm/"warm_check.json",dict(steps=WARM,actor_bits_identical=True,critic_changed=True))
    agent.num_timesteps=0


def train(arm, *, fixed_budget=False):
    torch.set_num_threads(1)
    check_sources()
    folder=ROOT/arm
    folder.mkdir(parents=True,exist_ok=False)
    agent=PPO(50,2,activation="relu",hidden_sizes=(256,256),seed=11,device="cpu",lr=RATE[arm],
              gamma=.995,gae_lambda=.95,n_steps=2048,n_epochs=10,minibatch_size=64,
              entropy_coef=0.,lane_entropy_coef=.002 if arm=="C" else 0.,lane_keep_bias=1.)
    reference=None
    if arm in SOURCE:
        agent.policy.load_state_dict(final.network(ORIG[SOURCE[arm]]).state_dict())
        reference=deepcopy(agent.policy).eval().requires_grad_(False)
        warm_critic(agent,reference,arm)
    target=200000 if arm=="C" else 32768
    def snapshot_inside_rollout(step):
        if arm=="C" and step in (50000,100000,150000):
            # Policy in force at this exact interaction, between full 2,048-step updates.
            store_checkpoint(agent,arm,step)
    with final.p.roll.baseline.simulator() as env:
        episodes=TrainingEpisodes(env,arm,"ppo",snapshot_inside_rollout)
        obs,_=episodes.reset()
        stats=[0.,0]
        while agent.num_timesteps<target:
            env.sums.clear()
            size=min(2048,target-agent.num_timesteps)
            obs,last,_=agent.collect_rollout(episodes,obs,ep_stats=stats,steps=size,timeout_is_failure=True)
            adv,ret=agent.buffer.compute_gae(last,agent.gamma,agent.gae_lambda)
            metrics=agent.update(adv,ret,reference_policy=reference,reference_coef=BETA[arm])
            step=agent.num_timesteps
            append(folder/"updates.jsonl",dict(step=step,metrics=metrics,rewards=dict(env.sums),
                                              completed=dict(episodes.outcomes)))
            if (arm in SOURCE and step%MARK==0) or step==target:
                store_checkpoint(agent,arm,step)
            if arm in SOURCE and step==32768 and not fixed_budget:
                decision=folder/"extension.json"
                while not decision.exists():
                    time.sleep(2)
                if read(decision)["extend"]:
                    target=65536
    save(folder/"done.json",dict(step=target,source_hashes=check_sources()))


def evaluate(task):
    arm,step,panel,rep=task
    base,reps=PANELS[panel]
    assert rep<reps
    cases=[(route,rep,base) for route in final.ROUTES]
    return final.run_episodes((str(ROOT/"evaluation"),cases,True,path_for(arm,step),False,False),
                              record_terminal=(arm=="C" and step==200000))


def paired(a,b):
    def identity(row):
        return tuple(row[k] for k in ("route","rep","seed","lane","initial_sha256"))
    assert list(map(identity,a))==list(map(identity,b))
    counts=Counter((x["outcome"]=="arrival",y["outcome"]=="arrival") for x,y in zip(a,b))
    return dict(kept=counts[True,True],lost=counts[True,False],gained=counts[False,True])


def rank(arm,step):
    c=final.tally(read(data_path(arm,step,"screen")))
    return c["arrival"],-c["collision"],-c["stall"]-c["timeout"],-c["off_route"],-step


def run(workers, experiment=None):
    # All arms share one evaluation pool, including when training runs concurrently.
    exp = experiment or sys.modules[__name__]
    ROOT, evaluate, rank = exp.ROOT, exp.evaluate, exp.rank
    path_for, data_path = exp.path_for, exp.data_path
    ROOT.mkdir(parents=True,exist_ok=True)
    ev=ROOT/"evaluation"
    ev.mkdir(exist_ok=True)
    save(ROOT/"protocol.json",exp.preflight())
    processes={}
    outputs=[]
    tasks={}
    grouped={}
    scheduled=set()
    choices={}
    def schedule(pool,arm,step,panel):
        key=(arm,step,panel)
        if key in scheduled:
            return
        scheduled.add(key)
        out=data_path(*key)
        if out.exists():
            return
        cached=ORIGINAL_ROWS.get(arm,{}).get(panel)
        if cached:
            rows=read(final.ROOT/cached/"episodes.json")
            assert [r["seed"] for r in rows]==final.planned(*PANELS[panel])
            save(out,rows)
            return
        grouped[key]={}
        for rep in range(PANELS[panel][1]):
            tasks[pool.submit(evaluate,(*key,rep))]=(key,rep)
    try:
        for arm in RATE:
            if (ROOT/arm/"done.json").exists():
                continue
            f=(ROOT/f"{arm}.log").open("a",encoding="utf8")
            outputs.append(f)
            processes[arm]=subprocess.Popen([sys.executable,"-u","-m",exp.__spec__.name,"--train",arm],
                                           stdout=f,stderr=subprocess.STDOUT)
        with ProcessPoolExecutor(workers) as pool:
            for arm in ORIG if experiment is None else ():
                for panel in PANELS:
                    schedule(pool,arm,0,panel)
            while True:
                for arm,proc in processes.items():
                    if proc.poll() not in (0,None):
                        raise RuntimeError(f"{arm} exited {proc.returncode}, see {arm}.log")
                for arm in RATE:
                    marks=range(50000,200001,50000) if arm=="C" else range(MARK,65537,MARK)
                    for step in marks:
                        if path_for(arm,step).exists():
                            schedule(pool,arm,step,"screen")
                    if arm in SOURCE and not (ROOT/arm/"extension.json").exists():
                        first=list(range(MARK,32769,MARK))
                        if all(data_path(arm,s,"screen").exists() for s in first):
                            counts=[rank(arm,s)[0] for s in first]
                            up=counts[-1]>counts[-2] and counts[-1]>counts[0]
                            threshold=107 if arm=="A" else 125
                            extend=up and (counts[-1]>=threshold if arm=="A" else counts[-1]>threshold)
                            if experiment is not None:
                                extend=exp.extend(arm,counts[-1],up)
                            save(ROOT/arm/"extension.json",dict(counts=counts,up=up,extend=extend))
                    done=ROOT/arm/"done.json"
                    if done.exists() and arm not in choices:
                        end=read(done)["step"]
                        scored=[s for s in marks if s<=end]
                        if all(data_path(arm,s,"screen").exists() for s in scored):
                            best=max(scored,key=lambda s:rank(arm,s))
                            choices[arm]=best
                            for panel in list(PANELS)[1:]:
                                schedule(pool,arm,best,panel)
                if len(choices)==3 and not tasks:
                    break
                if tasks:
                    finished,_=wait(tasks,timeout=1,return_when=FIRST_COMPLETED)
                    for fut in finished:
                        key,rep=tasks.pop(fut)
                        grouped[key][rep]=fut.result()
                        if len(grouped[key])==PANELS[key[2]][1]:
                            rows=sum((grouped[key][r] for r in sorted(grouped[key])),[])
                            assert [r["seed"] for r in rows]==final.planned(*PANELS[key[2]])
                            save(data_path(*key),rows)
                            print(key,final.tally(rows),flush=True)
                else:
                    time.sleep(1)
        save(ROOT/"selection.json",choices)
        exp.report(choices)
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for f in outputs:
            f.close()

def report(choices):
    torch.set_num_threads(1)
    source_hashes=check_sources()
    names={"C":"순수 PPO","BC":"BC","A":"BC + KL-PPO","R5":"BC + DAgger R5","B":"BC + DAgger R5 + KL-PPO"}
    cols=(*final.OUTCOMES,"slow10")
    counts={}
    rows_by_model={}
    for arm in names:
        rows_by_model[arm]={panel:read(data_path(arm,choices.get(arm,0),panel)) for panel in PANELS}
        counts[arm]={panel:final.tally(rows) for panel,rows in rows_by_model[arm].items()}
    for panel in PANELS:
        for arm in names:
            paired(rows_by_model["R5"][panel],rows_by_model[arm][panel])
    r5=counts["R5"]
    assert r5["additional"]["arrival"]==252 and r5["designated"]["arrival"]==247
    def meets(c):
        n=sum(c[k] for k in final.OUTCOMES)
        return c["arrival"]/n>=.95 and c["collision"]/n<.05
    # A replacement must meet the project target in all holdouts and improve at least
    # one without losing completions or increasing collisions on another.
    confirm=list(PANELS)[1:]
    promote=all(meets(counts["B"][p]) and counts["B"][p]["arrival"]>=r5[p]["arrival"]
                and counts["B"][p]["collision"]<=r5[p]["collision"] for p in confirm)
    promote=promote and any(counts["B"][p]["arrival"]>r5[p]["arrival"] or
                            counts["B"][p]["collision"]<r5[p]["collision"] for p in confirm)
    fix8=[]
    for orig,new in zip(rows_by_model["R5"]["designated"],rows_by_model["B"]["designated"]):
        if orig["route"]==8 and orig["outcome"]=="stall":
            fix8.append(dict(seed=orig["seed"],lane=orig["lane"],after=new["outcome"]))
    assert len(fix8)==2
    vx,vy=final.dataset(read(final.ROOT/"valid/episodes.json"),"valid")
    pure=final.network(path_for("C",200000))
    with torch.no_grad():
        accel=torch.cat([pure.det_action(chunk)[:,0] for chunk in vx.split(2048)])
    probe={}
    for delta in (0.,.1):
        mask=vy[:,0]>delta
        probe[str(delta)]=[int(((accel < -delta)&mask).sum()),int(mask.sum())]
    detail=read(data_path("C",200000,"screen"))
    text=["# 최종 환경의 PPO와 모방학습 5종 비교","",
          "발표 모델 추천은 **"+("보강형 R5 + KL-PPO 승격" if promote else "보강형 DAgger R5 유지")+"**다. 각 후보를 같은 도로·출발 차로·교통 시드에서 평가하고 최초 관측 해시를 주행별로 대조했다.","",
          "## 동일 조건 결과","","각 셀은 완주 / 충돌 / 끝 정지 / 일반 시간초과 / 실제 이탈 / 10초 이상 저속 합류 대기 순의 건수다. 대기는 종료 사유와 중복될 수 있다.","",
          "|모델|선정 스텝|선정 128회|64회|지정 256회|추가 256회|","|---|---:|---|---|---|---|"]
    for arm in names:
        cells=[" / ".join(str(counts[arm][p][k]) for k in cols) for p in PANELS]
        text.append("|"+"|".join([names[arm],str(choices.get(arm,"원본")),*cells])+"|")
    text+=["","|모델 · 완주율 / 충돌률|선정 128회|64회|지정 256회|추가 256회|","|---|---|---|---|---|"]
    for arm in names:
        cells=[]
        for p,(_,reps) in PANELS.items():
            c=counts[arm][p]
            cells.append(f"{100*c['arrival']/(reps*16):.2f}% / {100*c['collision']/(reps*16):.2f}%")
        text.append("|"+"|".join([names[arm],*cells])+"|")
    text+=["","104320000은 이번 PPO의 학습과 선정에서 분리했으며 원본 R5를 평가한 적이 있는 시드 묶음이다. 이 묶음에서 원본 252/256에 대해 KL-PPO는 "+str(counts["B"]["additional"]["arrival"])+"/256을 완주했다.",
           "","지정 256회의 8번 끝 정지 두 건은 "+", ".join(f"시드 {r['seed']} → {r['after']}" for r in fix8)+"였다.",
           "","## 학습 단계별 주행","","완주 유지·상실·새 완주는 같은 128회에서 해당 출발 모방 모델과 비교했다.",
           "","|조건|PPO 스텝|완주|충돌|끝 정지|시간초과|이탈|10초 대기|유지|상실|새 완주|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("A","B","C"):
        marks=sorted(int(p.stem.split("_")[1]) for p in (ROOT/"evaluation").glob(f"{arm}_*_screen.json"))
        for m in marks:
            rows=read(data_path(arm,m,"screen"))
            c=final.tally(rows)
            transitions=list(paired(rows_by_model[SOURCE[arm]]["screen"],rows).values()) if arm in SOURCE else ["—"]*3
            text.append("|"+"|".join(map(str,[arm,m,*[c[k] for k in cols],*transitions]))+"|")
    text+=["","연장 판정은 32,768스텝 완주 수가 직전과 첫 중간 평가보다 모두 높고 A는 107 이상, B는 125 초과일 때로 사전 정의했다. "+
           " · ".join(f"{arm} "+("연장" if read(ROOT/arm/"extension.json")["extend"] else "32,768 종료") for arm in SOURCE)+".",
           "","## 순수 PPO의 실패 위치와 가감속",""]
    early=read(ROOT/"C/early_50k.json")
    text+=["첫 50,000 학습 스텝의 종료 건수는 "+", ".join(f"{k} {early['outcomes'].get(k,0)}" for k in (*final.OUTCOMES,"clear_stop"))+"였다. 마지막에 진행 중인 주행의 보상도 누적했다.",
           "","|보상 항목|누적값|","|---|---:|"]
    text += [f"|{k}|{v:.2f}|" for k,v in early["rewards"].items()]
    text+= [f"|합계|{early['return_sum']:.2f}|",""]
    for delta,(opposed,total) in probe.items():
        text.append(f"검증 시범에서 전문가 가속 > {delta}, 모델 가속 < −{delta}인 상태는 {opposed}/{total} ({100*opposed/total:.2f}%)였다.")
    text+=["","200,000스텝 선정 패널의 실패 위치다. 이탈은 실제 진입한 도로, 충돌 위치는 마지막으로 관측된 도로다. 시간초과는 종료 순간 속도가 0.5m/s 이하일 때 정지로 표시했다.",
           "","|경로|실패 상황과 도로 (건수)|","|---:|---|"]
    failures={}
    for route in final.ROUTES:
        loc=Counter()
        for row in detail:
            if row["route"]!=route or row["outcome"]=="arrival":
                continue
            edge=row["end_edge"]
            kind={"stall":"차로 끝 정지","collision":"충돌","off_route":"분기 이탈","timeout":"시간초과"}[row["outcome"]]
            if row["outcome"]=="off_route" and row["off_route_detail"]:
                off=row["off_route_detail"]
                edge=off["src"]+" → "+off["actual"]
            if row["outcome"]=="timeout" and row["last_speed"]<=.5:
                kind+=("·출발 도로 정지" if row["end_edge"]==row["start_edge"] else "·주행 중 정지")
            loc[f"{kind} {edge}"]+=1
        failures[route]=dict(loc)
        text.append(f"|{route}|"+(", ".join(f"{k} ({v})" for k,v in loc.most_common()) or "전부 완주")+"|")
    text+=["","## 실행 조건과 선정 기준","",
           "수정 분기, 고가 교차 20쌍 보정, 50차원 관측, 고교통 15,000대/시, 안전게이트 OFF와 256×256 ReLU를 공통 적용했다. 학습은 전진 상한 +6, 분기 +2.5, 도착 +10, 충돌 −10, 이탈·끝 정지·시간초과 −8, 앞이 빈 저속 −0.02/스텝과 50스텝 연속 −7이다. 평가에는 학습용 저속 조기종료를 적용하지 않았다.",
           "","할인율 0.995, GAE 0.95, 롤아웃 2,048, 10에포크, 배치 64를 사용했다. A·B는 가치 출력층을 20,000스텝 워밍업하고 행동 파라미터의 바이트 일치를 검사했다. A는 학습률 2e−5와 KL 계수 0.5, B는 1e−5와 1.0, 엔트로피 보너스는 둘 다 0이다. C는 무작위 시드 11, 학습률 1e−4, 차로 엔트로피 0.002, 유지 편향 1이다. C 중간 체크포인트는 정확히 5만 스텝마다 그때 적용 중인 정책이고 20만은 마지막 부분 롤아웃 업데이트까지 반영했다.",
           "","학습 후 체크포인트 중 128회에서 완주 최대, 충돌 최소, 끝 정지와 시간초과 합 최소, 이탈 최소, 이른 시점 순으로 선정했다. R5 교체 기준은 세 홀드아웃에서 완주율 95% 이상·충돌률 5% 미만, 각각 기존 이상 완주·기존 이하 충돌, 적어도 한 곳에서 개선이다. 학습 초기화는 조건당 한 번이다.",
           "","[선정 스텝](../../results/jys3_relu_ppo_comparison/selection.json) · [집계와 원본 해시](../../results/jys3_relu_ppo_comparison/summary.json) · [주행별 기록](../../results/jys3_relu_ppo_comparison/evaluation)"]
    timed_out=[row for row in detail if row["outcome"]=="timeout"]
    stationary=[row for row in timed_out if row["last_speed"]<=.5]
    start_stops=sum(row["end_edge"]==row["start_edge"] for row in stationary)
    early_completed=sum(early["outcomes"].values())
    text[3:3]=["",
        f"순수 PPO의 첫 5만 학습 스텝에서 끝난 {early_completed}회 중 충돌은 {early['outcomes'].get('collision',0)}회였다. 20만 모델의 결정론적 128회 평가에서는 일반 시간초과 {len(timed_out)}건 중 {len(stationary)}건이 종료 순간 0.5m/s 이하였고 그중 {start_stops}건은 출발 도로에 있었다. 차로 끝 정지와 실제 오진입은 아래에서 경로별로 집계했다.",
        "",
        "BC에서 KL-PPO로 바꾼 뒤 완주 건수 차이는 "+", ".join(
            f"{label} {counts['A'][panel]['arrival']-counts['BC'][panel]['arrival']:+d}건"
            for panel,label in (("screen","선정"),("hold64","64회"),("designated","지정 256회"),("additional","추가 256회")))+
        f"이었다. R5의 지정 8번 끝 정지 두 건 중 {sum(r['after']=='arrival' for r in fix8)}건이 KL-PPO에서 완주로 바뀌었다."]
    save(ROOT/"summary.json",dict(selection=choices,counts=counts,route8=fix8,source_hashes=source_hashes,
                                  pure_acceleration_probe=probe,pure_failure_locations=failures,promote=promote))
    (final.DOCS/"jys3-relu-ppo-comparison.md").write_text("\n".join(text)+"\n",encoding="utf8")



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",choices=RATE)
    parser.add_argument("--workers",type=int,default=7)
    parser.add_argument("--report",action="store_true")
    args=parser.parse_args()
    if args.train:
        train(args.train)
    elif args.report:
        report(read(ROOT/"selection.json"))
    else:
        run(args.workers)
