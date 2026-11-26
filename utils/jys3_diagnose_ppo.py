"""Read-only checkpoint probes and matched SUMO replay. No optimizer is constructed."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
from itertools import chain
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("IC_LIBSUMO","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import torch
from utils import jys3_relu_ppo_comparison as run

final=run.final
ROOT=run.ROOT
OUT=ROOT/"diagnosis"
PANELS=("screen","designated","additional")
SEL=run.read(ROOT/"selection.json")
OUTCOMES=final.OUTCOMES


def dump(name,value):
    path=OUT/name
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf8")
    tmp.replace(path)


@lru_cache(None)
def policy(arm,step):
    return final.network(run.path_for(arm,step))


def rows(arm,step,panel):
    return run.read(run.data_path(arm,step,panel))


def lost_pairs(arm,panel):
    original=rows(run.SOURCE[arm],0,panel)
    tuned=rows(arm,SEL[arm],panel)
    run.paired(original,tuned)
    return [(a,b) for a,b in zip(original,tuned) if a["outcome"]=="arrival" and b["outcome"]!="arrival"]


def output(model,x):
    mu,_,z,_=model(x)
    return [float(mu.item()),*z.softmax(-1)[0].tolist()]


def replay_one(env,arm,step,row,start,finish):
    """Use final evaluation actions. Store policy pairs on the same failed-trajectory states."""
    model=policy(arm,step)
    route=f"route_{row['route']:02d}"
    env.ego_depart_lane=str(row["lane"])
    obs,_=env.reset(seed=row["seed"],options={"route":route})
    assert hashlib.sha256(obs.tobytes()).hexdigest()==row["initial_sha256"]
    trace=[]
    cmds=Counter()
    resets=Counter()
    clear=longest=slow_run=max_slow=0
    shadow_first=None
    censored_gap=blocked=fast=0
    for t in range(1,1001):
        state=torch.from_numpy(obs[:50]).unsqueeze(0)
        before=env.snapshot
        action=model.det_action(state)[0].numpy()
        cmds[int(action[1])]+=1
        if start<=t<=finish:
            item=dict(t=t,edge=before[0],lane=before[1],own=output(model,state))
            if arm in run.SOURCE:
                item["ref_same_state"]=output(policy(run.SOURCE[arm],0),state)
            trace.append(item)
        obs,_,done,cut,info=env.step(action)
        speed=info.get("speed")
        gap=info["gap"]
        low=speed is not None and speed<2
        open_road=gap is None or gap>25
        max_slow=max(max_slow,slow_run+1 if low else 0)
        slow_run=slow_run+1 if low else 0
        if low and open_road:
            clear+=1
            longest=max(longest,clear)
            if clear==50 and not (done or cut) and shadow_first is None:
                shadow_first=t
        else:
            if clear:
                cause="both" if not low and not open_road else "speed" if not low else "gap"
                resets[cause]+=1
            clear=0
        blocked+=int(low and not open_road)
        fast+=int(not low)
        censored_gap+=int(low and open_road)
        if done or cut:
            break
    assert (done or cut) and (t,info["outcome"])==(row["steps"],row["outcome"]),(arm,row["seed"],t,info["outcome"],row)
    after=env.snapshot
    loc=after if after is not None else before
    if "end_edge" in row:
        assert (loc[0],loc[1])==(row["end_edge"],row["end_lane"])
    return dict(trace=trace,edge=loc[0],lane=int(loc[1]),location_source="terminal" if after else "last_observed",
                before_edge=before[0],before_lane=int(before[1]),off_route=info.get("off_route_detail"),
                left=cmds[1],right=cmds[-1],keep=cmds[0],shadow_first=shadow_first,
                longest_clear=longest,longest_low=max_slow,low_blocked=blocked,speed_at_least_2=fast,
                low_clear=censored_gap,resets=dict(resets))


def replay_batch(job):
    arm,step,panel,group=job
    torch.set_num_threads(1)
    result=[]
    with final.p.roll.baseline.simulator() as env,torch.inference_mode():
        for old,row in group:
            ident={k:row[k] for k in ("seed","rep","route","lane","steps","outcome")}
            if old:
                high=row["steps"]
                low=max(1,high-19)
                failed=replay_one(env,arm,step,row,low,high)
                ref=replay_one(env,run.SOURCE[arm],0,old,low,high)
                timeline={s["t"]:s["own"] for s in ref["trace"]}
                for s in failed["trace"]:
                    s["ref_own_trajectory"]=timeline.get(s["t"])
                ident["original_steps"]=old["steps"]
            else:
                failed=replay_one(env,arm,step,row,1,0)
            result.append(dict(**ident,arm=arm,checkpoint=step,panel=panel,replay=failed))
    return result


def collect(workers):
    jobs=[]
    for arm in run.SOURCE:
        for panel in PANELS:
            pairs=lost_pairs(arm,panel)
            jobs.extend((arm,SEL[arm],panel,pairs[i:i+8]) for i in range(0,len(pairs),8))
    for step,panel in [(SEL["C"],p) for p in PANELS]+[(200000,"screen")]:
        if step==SEL["C"]==200000 and panel=="screen" and any(x[:3]==("C",step,panel) for x in jobs):
            continue
        group=[(None,r) for r in rows("C",step,panel)]
        jobs.extend(("C",step,panel,group[i:i+8]) for i in range(0,len(group),8))
    shard_dir=OUT/"shards"
    shard_dir.mkdir(exist_ok=True)
    def shard(job):
        arm,step,panel,g=job
        return shard_dir/f"{arm}_{step}_{panel}_{g[0][1]['seed']}.json"
    missing=[j for j in jobs if not shard(j).exists()]
    with ProcessPoolExecutor(workers) as pool:
        for job,result in zip(missing,pool.map(replay_batch,missing)):
            dump(shard(job).relative_to(OUT),result)
    combined=[]
    for job in jobs:
        arm,step,panel,pairs=job
        result=run.read(shard(job))
        assert len(result)==len(pairs)
        for row,(old,expected) in zip(result,pairs):
            assert (row["arm"],row["checkpoint"],row["panel"])==(arm,step,panel)
            assert all(row[k]==expected[k] for k in ("route","rep","lane","seed","steps","outcome"))
            if old:
                assert row["original_steps"]==old["steps"]
        combined.extend(result)
    return combined


def jsonlines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def fmetrics(y,pred):
    matrix=torch.bincount(y*3+pred,minlength=9).reshape(3,3).numpy()
    tp=np.diag(matrix)
    recalls=tp/np.maximum(1,matrix.sum(1))
    f1=2*tp/np.maximum(1,matrix.sum(0)+matrix.sum(1))
    changes=matrix[[0,2],:].sum()
    binary_tp=matrix[np.ix_([0,2],[0,2])].sum()
    return dict(matrix=matrix.tolist(),side_macro_f1=float(f1[[0,2]].mean()),
                side_recall=float(tp[[0,2]].sum()/changes),recalls=recalls.tolist(),
                binary_recall=float(binary_tp/changes),binary_f1=float(2*binary_tp/(changes+matrix[:,[0,2]].sum())))


@torch.inference_mode()
def probes():
    x,y=final.dataset(run.read(final.ROOT/"valid/episodes.json"),"valid")
    cls=(1-y[:,1]).long()
    result={}
    for arm,source in run.SOURCE.items():
        ref=policy(source,0)
        mu0,s0,z0,_=ref(x)
        logp0=z0.log_softmax(1)
        preds0=z0.argmax(1)
        records=[]
        for p in sorted((ROOT/arm).glob("policy_*.pt"),key=lambda v:int(v.stem.split("_")[1])):
            step=int(p.stem.split("_")[1])
            model=policy(arm,step)
            mu,s,z,_=model(x)
            logp=z.log_softmax(1)
            pred=z.argmax(1)
            lane_kl=(logp.exp()*(logp-logp0)).sum(1)
            accel_kl=(torch.log(s0/s)+(s.square()+(mu-mu0).square())/(2*s0.square())-.5)[:,0]
            groups={}
            for name,mask in (("keep",cls==1),("change",cls!=1)):
                groups[name]=dict(n=int(mask.sum()),lane=float(lane_kl[mask].mean()),
                       accel=float(accel_kl[mask].mean()),joint=float((lane_kl+accel_kl)[mask].mean()),
                       ref_accuracy=float((preds0[mask]==cls[mask]).float().mean()),
                       tuned_accuracy=float((pred[mask]==cls[mask]).float().mean()),
                       flips=int((pred[mask]!=preds0[mask]).sum()))
            rec=dict(step=step,groups=groups,metrics=fmetrics(cls,pred),
                     log_std=model.log_std.item(),sigma=float(s))
            if step==SEL[arm]:
                shifts={}
                for label,prefix in (("fc1","trunk.0"),("fc2","trunk.2"),("actor_accel","mu_head"),
                                     ("actor_lane","lane_head"),("critic","v_head")):
                    base=torch.cat([v.flatten().double() for k,v in ref.named_parameters() if k.startswith(prefix+".")])
                    dest=torch.cat([v.flatten().double() for k,v in model.named_parameters() if k.startswith(prefix+".")])
                    delta=(dest-base).norm().item()
                    shifts[label]=[delta,100*delta/base.norm().item()]
                rec["shifts"]=shifts
                swaps={}
                for name,key in (("trunk_only","trunk"),("lane_head_only","lane_head")):
                    hybrid=deepcopy(ref)
                    getattr(hybrid,key).load_state_dict(getattr(model,key).state_dict())
                    hp=hybrid(x)[2].argmax(1)
                    swaps[name]=dict(metrics=fmetrics(cls,hp),flips_change=int(((hp!=preds0)&(cls!=1)).sum()))
                rec["swaps"]=swaps
            records.append(rec)
        warm=jsonlines(ROOT/arm/"warm_episodes.jsonl")
        updates=jsonlines(ROOT/arm/"updates.jsonl")
        first=jsonlines(ROOT/arm/"ppo_episodes.jsonl")
        original_rollouts=[r for r in first if r["step"]<=2048]
        result[arm]=dict(base_metrics=fmetrics(cls,preds0),base_log_std=ref.log_std.item(),
             base_sigma=float(s0),warm=dict(Counter(r["outcome"] for r in warm)),warm_n=len(warm),
             selected_update=next(r for r in updates if r["step"]==SEL[arm]),
             initial_ppo=dict(Counter(r["outcome"] for r in original_rollouts)),trajectory=records)
    mask=y[:,0]>.1
    for step in (50000,100000,150000,200000):
        model=policy("C",step)
        mu,_,z,_=model(x)
        result[f"C{step}"]=dict(metrics=fmetrics(cls,z.argmax(1)),
            opposite=int(((mu[:,0]<-.1)&mask).sum()),expert_accel=int(mask.sum()))
        if step==200000:
            gap_groups={}
            for label,gap_mask in (("over_25m",x[:,9]*120>25),("at_most_25m",x[:,9]*120<=25)):
                subset=mask&gap_mask
                gap_groups[label]=[int(((mu[:,0]<-.1)&subset).sum()),int(subset.sum())]
            result[f"C{step}"]["by_observed_lead_gap"]=gap_groups
    return result


def geometry():
    folder=ROOT.parents[1]/"scenarios"/"hv_high15000_entryprep"
    net=ET.parse(folder/"network.net.xml")
    edges={e.get("id") for e in net.findall("edge") if not e.get("function")}
    links=defaultdict(lambda:defaultdict(list))
    for c in net.findall("connection"):
        src,dst=c.get("from"),c.get("to")
        if src in edges and dst in edges:
            links[src][dst].append((int(c.get("fromLane")),int(c.get("toLane"))))
    result={}
    for r in ET.parse(folder/"highway.rou.xml").findall("route"):
        if not r.get("id","").startswith("route_"):
            continue
        route=int(r.get("id").split("_")[1]); path=r.get("edges").split()
        for lane in (0,1):
            reachable={lane}
            group="no_fork"
            for src,dst in zip(path,path[1:]):
                reachable={b for a,b in links[src][dst] if a in reachable}
                if len(links[src])>1:
                    group="connected" if reachable else "needs_change"
                    break
            result[route,lane]=group
    return result


def returns():
    episodes=jsonlines(ROOT/"C/ppo_episodes.jsonl")
    previous=0
    buckets=defaultdict(list)
    base={"arrival":10,"off_route":-8,"collision":-10,"clear_stop":-7,"stall":-8,"timeout":-8}
    for r in episodes:
        n=r["step"]-previous
        previous=r["step"]
        last=.995**(n-1)
        weight_sum=(1-.995**n)/(1-.995)
        terminal=base[r["outcome"]]
        residue=r["episode_return"]-terminal
        # Each nonterminal component, including at the last step, is >= -0.02.
        # Bound timestamp-unknown shaping using 0.995^(n-1) <= w <= 1.
        lo=last*(terminal+residue)-.02*(weight_sum-last*n)
        hi=last*terminal+residue+.02*(n-weight_sum)
        b=0 if previous<=50000 else 1 if previous<=100000 else 2
        buckets[b,r["outcome"]].append([n,r["episode_return"],terminal*last,lo,hi])
    out=[]
    for (b,k),values in sorted(buckets.items()):
        means=np.array(values).mean(0).tolist()
        out.append(dict(period=b,outcome=k,n=len(values),length=means[0],undiscounted=means[1],
                        discounted_terminal=means[2],discounted_lower=means[3],discounted_upper=means[4]))
    return out


@torch.inference_mode()
def recover_training_segments():
    """Only episodes fully inside a rollout for which the exact policy can be recovered."""
    torch.set_num_threads(1)
    snapshots={k:policy("C",k) for k in (50000,100000,150000)}
    torch.manual_seed(11)
    snapshots[0]=final.network()  # identical seed and constructor to the initial pure PPO actor
    logs=jsonlines(ROOT/"C/ppo_episodes.jsonl")
    intervals={k:(k//2048*2048,k//2048*2048+2048) for k in snapshots}
    candidates=[]
    start=0
    for idx,r in enumerate(logs):
        for k,(lower,upper) in intervals.items():
            if lower<=start and r["step"]<=upper:
                candidates.append((k,idx,start,r))
        start=r["step"]
    normal_mu=torch.zeros(1,1)
    normal_std=torch.ones(1,1)
    lane_probs=torch.full((1,3),1/3)
    clock=0
    recovered=[]
    with final.p.roll.baseline.simulator() as env:
        env.progress_cap,env.off_route_penalty=6.,-8.
        run.driving.ARRIVAL_REWARD,run.driving.PASS_REWARD=10.,2.5
        run.driving.COLLISION_REWARD,run.driving.TIMEOUT_REWARD=-10.,-8.
        for k,idx,start,r in candidates:
            # The original CPU sampler draws one normal and one categorical variate per step.
            # PPO shuffles minibatches with NumPy and consumes no Torch exploration draws.
            for _ in range(start-clock):
                torch.normal(normal_mu,normal_std)
                torch.multinomial(lane_probs,1,True)
            route=f"route_{r['route']:02d}"
            env.ego_depart_lane=str(min(idx//16%2,env.counts[env.routes[route][0]]-1))
            obs,_=env.reset(seed=r["seed"],options={"route":route})
            idle=0;rewards=[];last_cmds=deque(maxlen=25)
            for t in range(r["step"]-start):
                act,_,_=snapshots[k].sample(torch.from_numpy(obs[:50]).unsqueeze(0))
                last_cmds.append(int(act[0,1]))
                obs,reward,done,cut,info=env.step(act[0].numpy())
                clear=info.get("speed") is not None and info["speed"]<2 and (info["gap"] is None or info["gap"]>25)
                idle=idle+1 if clear else 0
                reward-=.02*clear
                if idle>=50 and not(done or cut):
                    info["outcome"]="clear_stop";reward-=7.;done=True
                rewards.append(reward)
                if done or cut:break
            clock=start+len(rewards)
            matched=(bool(done or cut) and clock==r["step"] and info["outcome"]==r["outcome"]
                     and abs(sum(rewards)-r["episode_return"])<1e-6)
            if not matched:
                raise RuntimeError(f"Stochastic replay mismatch at {r['seed']}: {clock}, {info['outcome']}, {sum(rewards)} vs {r}")
            recovered.append(dict(seed=r["seed"],end_step=clock,route=r["route"],outcome=r["outcome"],
                                  length=len(rewards),discounted=sum(.995**t*v for t,v in enumerate(rewards)),
                                  undiscounted=sum(rewards),last_command=last_cmds[-1],
                                  changed_command_last_5s=any(last_cmds),snapshot=k))
    dump("discounted_verified.json",recovered)
    return recovered


def analyze(records):
    stats=probes()
    mapping=geometry()
    losses={}
    for arm,source in run.SOURCE.items():
        changes=[]
        for p in PANELS:
            a,b=rows(source,0,p),rows(arm,SEL[arm],p)
            trans=run.paired(a,b)
            changes.append(dict(panel=p,old_off=sum(r["outcome"]=="off_route" for r in a),
                new_off=sum(r["outcome"]=="off_route" for r in b),**trans))
        group=[r for r in records if r["arm"]==arm]
        assert len(group)==sum(p["lost"] for p in changes)
        trace=[s for r in group for s in r["replay"]["trace"]]
        own=np.asarray([s["own"] for s in trace])
        ref=np.asarray([s["ref_same_state"] for s in trace])
        valid=[s for s in trace if s["ref_own_trajectory"] is not None]
        old_pred=ref[:,1:].argmax(1);new_pred=own[:,1:].argmax(1)
        losses[arm]=dict(trans=changes,n=len(group),outcomes=dict(Counter(r["outcome"] for r in group)),
            routes=dict(Counter(r["route"] for r in group)),window_states=len(trace),
            actual_matched_steps=len(valid),own=own.mean(0).tolist(),ref=ref.mean(0).tolist(),
            ref_own_mean=np.mean([s["ref_own_trajectory"] for s in valid],axis=0).tolist(),
            paired_tuned_mean=np.mean([s["own"] for s in valid],axis=0).tolist(),
            flips=int((old_pred!=new_pred).sum()),lane_to_keep=int(((old_pred!=1)&(new_pred==1)).sum()),
            keep_to_lane=int(((old_pred==1)&(new_pred!=1)).sum()),
            mu_abs_delta=float(np.abs(own[:,0]-ref[:,0]).mean()),
            sites=Counter(f"{r['route']} / {r['replay']['edge']} / {r['replay']['lane']} / {r['outcome']}" for r in group).most_common())
    pure=[r for r in records if r["arm"]=="C"]
    subsets=defaultdict(list)
    for r in pure:
        subsets[r["checkpoint"],r["panel"],mapping[r["route"],r["lane"]]].append(r)
    groups=[]
    for (step,panel,key),part in sorted(subsets.items()):
        c=Counter(r["outcome"] for r in part)
        l=sum(r["replay"]["left"] for r in part);rr=sum(r["replay"]["right"] for r in part)
        t=sum(r["steps"] for r in part)
        groups.append(dict(step=step,panel=panel,group=key,n=len(part),outcomes=dict(c),
             left=l,right=rr,commands_1000=1000*(l+rr)/t,command_episodes=sum(r["replay"]["left"]+r["replay"]["right"]>0 for r in part)))
    times=[r for r in pure if r["checkpoint"]==200000 and r["outcome"]=="timeout"]
    assert len(times)==36
    for r in times:
        v=r["replay"]
        if v["shadow_first"] is not None: kind="would_stop"
        elif v["longest_low"]<50: kind="speed_limits_runs"
        else: kind="gap_breaks_long_low"
        v["timeout_class"]=kind
    stop=Counter(r["replay"]["timeout_class"] for r in times)
    # Preserve one row per lost journey, including the checkpoint pair and exact terminal location.
    with (OUT/"lost.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["arm","panel","route","seed","start_lane","outcome","step","end_edge","end_lane","location_source","before_edge","before_lane","intended_next","actual_next","window","ref_mu","ppo_mu","ref_left","ppo_left","ref_keep","ppo_keep","ref_right","ppo_right"])
        for r in records:
            if r["arm"] not in run.SOURCE:continue
            v=r["replay"];off=v["off_route"] or {}
            tr=v["trace"];a=np.mean([s["ref_same_state"] for s in tr],0);b=np.mean([s["own"] for s in tr],0)
            w.writerow([r["arm"],r["panel"],r["route"],r["seed"],r["lane"],r["outcome"],r["steps"],v["edge"],v["lane"],v["location_source"],v["before_edge"],v["before_lane"],off.get("expected"),off.get("actual"),len(tr),*chain.from_iterable(zip(a,b))])
    with (OUT/"timeout36.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["route","seed","edge","lane","class","first_train_stop_step","max_clear_slow","max_low","low_with_close_front","low_with_clear_front","other_steps","gap_resets","speed_resets","combined_resets"])
        for r in times:
            v=r["replay"]
            w.writerow([r["route"],r["seed"],v["edge"],v["lane"],v["timeout_class"],v["shadow_first"],v["longest_clear"],v["longest_low"],v["low_blocked"],v["low_clear"],v["speed_at_least_2"],*[v["resets"].get(k,0) for k in ("gap","speed","both")]])
    dump("analysis.json",dict(checkpoints=stats,losses=losses,groups=groups,returns=returns(),timeouts=stop,
                              hashes=run.check_sources(),verified_results=len(records)))


def report(records):
    d=run.read(OUT/"analysis.json")
    loss,probes=d["losses"],d["checkpoints"]
    short={"screen":"선정 128","designated":"지정 256","additional":"추가 256"}
    reason={"arrival":"완주","collision":"충돌","off_route":"이탈","stall":"차로 끝 정지",
            "clear_stop":"10초 정지","timeout":"일반 시간초과"}
    periods=["0~5만","5~10만","10~20만"]
    fmt=lambda a:" / ".join(f"{v:.3f}" for v in a)
    pct=lambda a,b:f"{100*a/b:.1f}%"
    selected={arm:next(r for r in probes[arm]["trajectory"] if r["step"]==SEL[arm]) for arm in run.SOURCE}
    text=["# 세 PPO 모델의 실패 원인과 재설계 근거","",
          "분석 대상은 BC + KL-PPO(A, 32,768스텝), 보강형 DAgger R5 + KL-PPO(B, 32,768스텝), 순수 PPO(C, 선정 10만·최종 20만 스텝)다. 가중치를 고정했다. A·B는 선정 128회와 지정·추가 각 256회를 원본과 짝지었다. 64회 패널은 이 분석의 분모에서 제외했다.",
          "","## 1. 기존 완주를 잃은 주행 전수","",
          "|모델|패널|원본 → PPO 이탈|기존 완주 유지|완주 상실|새 완주|","|---|---|---:|---:|---:|---:|"]
    for arm,v in loss.items():
        for p in v["trans"]:
            text.append(f"|{arm}|{short[p['panel']]}|{p['old_off']} → {p['new_off']}|{p['kept']}|{p['lost']}|{p['gained']}|")
        sums={k:sum(p[k] for p in v["trans"]) for k in ("old_off","new_off","kept","lost","gained")}
        text.append(f"|**{arm}**|**합계 640**|**{sums['old_off']} → {sums['new_off']}**|{sums['kept']}|{sums['lost']}|{sums['gained']}|")
    text+=["","|모델|상실 사례의 종료 사유|반복된 실패 위치 · 경로 / 도로 ID / 차로 / 종료 사유 (건수)|","|---|---|---|"]
    for arm,v in loss.items():
        text.append("|"+arm+"|"+", ".join(f"{reason[k]} {n}" for k,n in v["outcomes"].items())+"|"+
                    "<br>".join(f"{site} ({n})" for site,n in v["sites"][:5])+"|")
    text+=["","실패 직전 최대 20번의 행동을 세 방식으로 비교했다. ① 실패한 학생의 실제 출력, ② 학생이 본 동일 관측에 원본을 넣은 출력, ③ 원본이 독립적으로 완주할 때 같은 시각에 실제로 낸 출력이다. ①과 ②는 입력이 같고, ①과 ③은 교통 시드와 시각이 같다. ③은 원본이 먼저 완주한 뒤의 시각을 제외했다. 도로 ID가 사라진 충돌은 직전 관측 도로·차로로 표시했다.",
           "","|모델|관측 위치|스텝 수|평균 가감속 μ|평균 P(좌) / P(유지) / P(우)|","|---|---|---:|---:|---|"]
    for arm,v in loss.items():
        for key,n,label in (("ref","window_states","② 실패 관측에서 원본"),("own","window_states","① 실패 관측에서 PPO"),
                            ("ref_own_mean","actual_matched_steps","③ 원본 자기 궤적"),("paired_tuned_mean","actual_matched_steps","① 같은 시각의 PPO")):
            a=v[key]
            text.append(f"|{arm}|{label}|{v[n]}|{a[0]:+.3f}|{fmt(a[1:])}|")
    for arm,v in loss.items():
        text.append(f"\n{arm}의 같은 실패 관측 {v['window_states']}개에서 차로 argmax가 바뀐 것은 {v['flips']}개다. 원본 변경→PPO 유지는 {v['lane_to_keep']}개, 유지→변경은 {v['keep_to_lane']}개였고 가감속의 평균 절대 변화는 {v['mu_abs_delta']:.3f}였다.")
    text+=["","상실한 73건의 원본·PPO 146회와 순수 PPO 768회를 재생했다. 모든 주행의 최초 관측 SHA-256, 종료 스텝, 종료 사유를 기존 기록과 대조했다. 마지막 20스텝 평균은 앞선 분기 준비 시점의 행동까지 설명하지 않는다. [상실한 모든 주행과 종점·차로·평균 출력](../../results/jys3_relu_ppo_comparison/diagnosis/lost.csv) · [매 스텝의 세 정책 출력과 재생 기록](../../results/jys3_relu_ppo_comparison/diagnosis/replay.json)",
           "","## 2. 탐험 분산, 공유층, 희귀한 차로변경 결정","","### 원본 행동 파라미터가 고정된 2만 스텝 워밍업","","|출발 모델|확률 주행 · 완주 / 충돌|결정론 선정 128회 · 완주 / 충돌|완주율 차이|원본 → 선정 log_std|표준편차 σ|","|---|---|---|---:|---:|---:|"]
    for arm,src in run.SOURCE.items():
        v=probes[arm];n=v["warm_n"];a=v["warm"].get("arrival",0);c=v["warm"].get("collision",0)
        rc=final.tally(rows(src,0,"screen"))
        text.append(f"|{src}|{a}/{n} ({pct(a,n)}) / {c}/{n} ({pct(c,n)})|{rc['arrival']}/128 ({pct(rc['arrival'],128)}) / {rc['collision']}/128 ({pct(rc['collision'],128)})|{100*(a/n-rc['arrival']/128):+.1f}%p|{v['base_log_std']:.4f} → {selected[arm]['log_std']:.4f}|{v['base_sigma']:.4f} → {selected[arm]['sigma']:.4f}|")
    text+=["","워밍업 동안 행동 파라미터는 비트 단위로 고정됐고 가감속과 차로를 모두 샘플링했다. 완료된 에피소드만 집계했다. 훈련 교통 시드와 10초 종료 조건도 평가와 다르다. σ는 약 0.61로 거의 일정했다. A의 확률 주행 충돌률은 결정론 평가보다 낮고 B에서는 한 건이었다. 공통 원인을 가감속 노이즈로 특정할 대조는 없다. 시뮬레이터가 기록한 충돌은 실제 충돌 건수다.",
           "","### 어떤 층이 움직였나","","각 셀은 가중치·편향을 합친 L2 변화량 / 원본 L2 대비 변화율이다. critic은 워밍업과 PPO를 모두 거친 값이다.",
           "","|모델|fc1|fc2|가감속 출력|차로 출력|가치 출력|","|---|---:|---:|---:|---:|---:|"]
    for arm,v in selected.items():
        text.append("|"+arm+"|"+"|".join(f"{v['shifts'][k][0]:.4f} / {v['shifts'][k][1]:.2f}%" for k in ("fc1","fc2","actor_accel","actor_lane","critic"))+"|")
    text+=["","검증 시범에서 원본 네트워크의 일부를 PPO 후 가중치로 잠시 교체해 순전파했다. 두 공유층만 교체하면 변경 방향 정답이 **A 22→15/29, B 25→18/29**였다. 차로 출력층만 교체하면 **A 23/29, B 26/29**였다. 공유층 변화만으로 재현율 하락이 재현됐다. 가치 손실과 정책 손실의 공유층별 기울기는 저장되지 않아 어느 손실이 얼마만큼 움직였는지는 분해할 수 없다.",
           "","### 전문가 유지와 변경에서 기준 정책 KL이 어떻게 달랐나","","KL은 학습 코드와 같은 방향인 KL(PPO ∥ 원본)이며 단위는 nat다. 검증 시범 32회·11,737상태 중 변경은 29개(0.247%)다.",
           "","|모델|전문가 정답|상태 수|차로 범주 KL|가감속 정규 KL|합산 KL|해당 그룹 정답률 원본 → PPO|","|---|---|---:|---:|---:|---:|---:|"]
    for arm,v in selected.items():
        for label,g in v["groups"].items():
            text.append(f"|{arm}|{label}|{g['n']}|{g['lane']:.5f}|{g['accel']:.5f}|{g['joint']:.5f}|{100*g['ref_accuracy']:.1f}% → {100*g['tuned_accuracy']:.1f}%|")
    text+=["","|모델|희귀/유지 차로 KL 배수|좌·우 방향 평균 F1|변경 방향 재현율|변경/유지 이진 F1|","|---|---:|---:|---:|---:|"]
    for arm,v in selected.items():
        a=probes[arm]["base_metrics"];b=v["metrics"];g=v["groups"]
        text.append(f"|{arm}|{g['change']['lane']/g['keep']['lane']:.1f}배|{a['side_macro_f1']:.3f} → {b['side_macro_f1']:.3f}|{100*a['side_recall']:.1f}% → {100*b['side_recall']:.1f}%|{a['binary_f1']:.3f} → {b['binary_f1']:.3f}|")
    text+=["","B에서는 유지 정답에 잘못 변경을 명령한 상태가 17→2개로 줄어 F1이 올랐다. 실제 변경 정답을 맞힌 횟수는 25→19개로 줄었다. 유지가 압도적인 이 검증 표본에서 전체 평균은 변경 상태의 큰 KL을 가린다. 이 29개는 검증용 시범 상태이며, 매 업데이트 학습 배치의 전문가 정답 비율은 기록되지 않았다.",
           "","## 3. 순수 PPO가 완주한 출발과 실패한 출발","","출발 차로에서 임무 경로의 첫 분기 출구까지 실제 지도 연결을 따라갔다. 중간에서 차로 명령을 한 번도 쓰지 않아도 연결이 하나 이상 남으면 ‘연결’, 모두 끊기면 ‘변경 필요’다. 외부 도로 분기가 없는 경로는 따로 셌다. 연결 출발도 이후 분기나 합류에는 변경이 필요할 수 있다. 명령은 매 스텝 정책이 선택한 횟수다.",
           "","|가중치 · 패널|출발 그룹|N|완주|이탈|끝 정지|일반 시간초과|충돌|좌/우 명령|","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys={"connected":"연결","needs_change":"변경 필요","no_fork":"분기 없음"}
    for g in d["groups"]:
        n=g["n"];c=g["outcomes"]
        text.append(f"|{g['step']//10000}만 · {short[g['panel']]}|{keys[g['group']]}|{n}|"+
                    "|".join(f"{c.get(k,0)} ({pct(c.get(k,0),n)})" for k in ("arrival","off_route","stall","timeout","collision"))+f"|{g['left']}/{g['right']}|")
    text+=["","20만은 128회, 두 256회는 당시 선정된 10만 가중치로 재생했다. 네 중간 체크포인트 모두 검증 시범의 실제 차로변경 정답 29개 중 0개를 결정론적으로 맞혔다. 전문가 가속 >0.1 상태 2,547개에서 학생 출력 <−0.1은 5만·10만·15만·20만에 각각 "+
           " / ".join(str(probes[f"C{k}"]["opposite"]) for k in (50000,100000,150000,200000))+"개였다. 20만의 56.7%는 같은 상태에서 직접 측정한 감속이다. C 학습 배치의 평균 차로변경 확률은 첫 업데이트 뒤 33.7%, 마지막 업데이트 뒤 6.0%였다. 결정론 주행 768회에서는 좌·우 명령이 각각 0회였다. 확률적 탐색과 실제 argmax 선택은 서로 다른 지표다. 전문가가 가속한 상태의 관측 전방 간격을 나누면 25m 초과에서 1,095/1,920개(57.0%), 25m 이하에서 348/627개(55.5%)가 반대 감속이었다. 감속 편향은 가까운 앞차를 본 상태에만 집중되지 않았다.",
           "","### 일반 시간초과 36건의 10초 카운터 재구성","",
           "평가에서는 10초 종료를 사용하지 않았다. 원래 1,000스텝을 끝까지 재생하며 학습 코드와 같은 후속 관측 속도·앞차 간격으로 별도 카운터를 계산했다. 50스텝을 채운 주행부터 분류하고, 나머지는 연속 저속 자체가 짧은 경우와 저속이 50스텝 이상인데 간격 조건이 끊긴 경우로 나눴다.",
           "","|분류|주행 수|","|---|---:|"]
    for k,label in (("would_stop","평가 중 50스텝 충족. 학습 규칙이면 그 시점에 종료"),("speed_limits_runs","속도 <2인 연속 구간이 모두 50스텝 미만"),("gap_breaks_long_low","50스텝 이상 저속 구간이 있었으나 앞차 간격 ≤25m로 무장애 연속 50스텝 미충족")):
        text.append(f"|{label}|{d['timeouts'].get(k,0)}|")
    text+=["","나머지 한 건은 2번 경로·시드 95380302, 도로 169728849의 0차로였다. 연속 저속은 최장 363스텝이었으나 앞이 빈 연속 저속은 최장 27스텝이었다. 속도와 간격에 의한 초기화가 한 주행에 함께 나타날 수 있다. 개별 주행별 첫 발동 스텝, 최장 저속·무장애 저속, 간격·속도·동시 초기화 횟수는 [36건 대조표](../../results/jys3_relu_ppo_comparison/diagnosis/timeout36.csv)에 있다.",
           "","## 4. 충돌·정지·이탈의 실제 수익 순서","","학습 로그의 episode_return은 할인 전 합이다. 종료 스텝 차이로 주행 길이를 구해 마지막 보상의 할인값은 정확히 계산할 수 있다. 매 2,048스텝의 정책과 스텝별 보상은 전부 저장되지 않았다. 전체 표에는 확인할 수 있는 할인 수익의 상·하한을, 바로 아래에는 같은 확률 행동을 복원한 일부 주행의 정확한 할인 수익을 표시했다. 구간은 에피소드가 끝난 스텝 기준이다.",
           "","|학습 구간|종료|건수|평균 길이|평균 할인 전 수익|할인된 마지막 보상 항|전체 평균 할인 수익의 가능 구간|","|---|---|---:|---:|---:|---:|---:|"]
    for r in d["returns"]:
        text.append(f"|{periods[r['period']]}|{reason[r['outcome']]}|{r['n']}|{r['length']:.1f}|{r['undiscounted']:+.2f}|{r['discounted_terminal']:+.2f}|[{r['discounted_lower']:+.2f}, {r['discounted_upper']:+.2f}]|")
    text+=["","종료 항 이외의 스텝당 보상 하한은 −0.02다. 각 스텝의 나머지 보상에 0.02를 더하면 음수가 없으므로, 그 합에 가장 작은 할인 가중치와 가장 큰 가중치를 각각 곱해 경계를 구할 수 있다. 이후 매 스텝 보탠 0.02의 할인 합을 빼고 할인된 종료 항을 더했다. 상·하한은 통계적 신뢰구간이 아니다.",
           "","초반에는 이탈의 하한 −3.81과 10초 정지의 하한 −3.26이 충돌의 상한 −6.27보다 높다. 따라서 **첫 5만 스텝에 관측된 평균 할인 수익은 이탈과 정지가 각각 충돌보다 높았다.** 완주 평균의 하한은 +5.06이었다. 종료 사유별 관측값으로 같은 상태에서 다른 행동을 선택했을 때의 기대 수익까지 계산할 수는 없다."]
    exact=run.read(OUT/"discounted_verified.json")
    bins=defaultdict(list)
    for r in exact:
        b=0 if r["end_step"]<=50000 else 1 if r["end_step"]<=100000 else 2
        bins[b,r["outcome"]].append(r)
    text+=["","**확률 주행 복원 표본**. 시드 11 초기 정책과 5만·10만·15만 시점에 저장한 정책이 실제로 적용된 2,048스텝 롤아웃 안에서 시작·종료한 에피소드다. CPU 난수 순서를 복원하고 원래 교통 시드, 길이, 종료 사유, 할인 전 수익(허용 오차 0.000001)을 모두 대조했다. 아래 평균은 해당 표본에만 해당한다.",
           "","|구간|종료|복원 건수|실제 평균 할인 수익|종료 전 최대 5초에 차로변경 명령이 있었던 건수|","|---|---|---:|---:|---:|"]
    for (b,k),rr in sorted(bins.items()):
        text.append(f"|{periods[b]}|{reason[k]}|{len(rr)}|{np.mean([v['discounted'] for v in rr]):+.3f}|{sum(v['changed_command_last_5s'] for v in rr)}|")
    text+=["","종료 직전에 변경 명령이 있었다는 기록만으로 충돌 원인을 측면 변경으로 분류할 수는 없다. [복원한 학습 주행별 할인 수익](../../results/jys3_relu_ppo_comparison/diagnosis/discounted_verified.json)",
           "","## 5. 재설계 우선순위","","아래 수치는 다음 대조 실험의 제안값이다. 이번 진단의 관측 성능과 구별한다.",
           "","|순위 · 적용 모델|실측 근거|먼저 바꿀 한 가지와 판정 기준|","|---|---|---|",
           "|1 · BC/R5 + PPO|공유층만 바꿔도 A·B가 전문가 변경 정답을 각각 7개씩 잃었다. 두 원본의 공유층 변화는 fc1 기준 9.37%·5.16%였다.|첫 8,192스텝에서 원본 trunk를 복제해 critic 전용으로 두고 가치 손실의 기울기가 actor로 흐르는 경로 하나만 끊는다. actor는 기존처럼 정책 손실로 갱신한다. 원래 lr(A 2e−5, B 1e−5), 보상과 KL 계수는 고정해 분리 효과를 본다. 첫 판정은 원본의 변경 정답 A 22/29·B 25/29 보존과 같은 출발의 기존 완주 상실 건수다.|",
           "|2 · BC/R5 + PPO|차로변경 상태의 차로 KL이 유지 상태보다 A 약 27배·B 약 198배였다. R5는 F1 상승에도 필요한 변경을 6개 더 놓쳤다.|1번과 별도 조건으로 유지·변경 KL을 따로 평균해 차로 기준 KL에 0.9·0.1 가중치를 적용한다. 공통 기준은 기존 A β=0.5, B β=1.0. 정답 그룹은 학습용 시범/학습용 라벨로만 구성하고 검증 29개는 채점 전용이다. 방향별 재현율·유지 오경보·분기 이탈을 함께 비교한다.|",
           "|3 · 순수 PPO|초기 579회 중 432회 충돌했고, 실제 변경이 필요한 출발의 행동과 종료가 위 표에서 갈렸다. 20만에 전문가 가속의 56.7%에서 반대 감속했다.|우선 첫 분기 전부터 연결/변경 필요 출발을 1:1로 맞춘 짧은 구간에서 학습한 뒤 전체 경로로 확장한다. 각 방향의 실제 통과율과 기존 완주 유지로 승격한다. 별도 대조로 첫 5만 스텝을 저교통 3,000대/시로 바꾸고 최종 평가는 계속 15,000대/시에서 한다. 구간 단축·교통량 효과를 한 번에 섞지 않는다.|",
           "","### 보상과 탐험 파라미터의 후순위 대조",
           "",
           "- **미세조정의 탐험 분산.** 원본 log_std −0.5를 고정한 대조군과 초기값·상한 모두 −1.0(σ 0.368)인 조건을 같은 시드에서 비교한다. 차로 범주 샘플링은 동일하게 두고 가감속 분산만 바꾼다. 워밍업에서 충돌이 일관되게 늘지 않았으므로 이 변경에는 앞선 두 조정보다 낮은 우선순위를 준다.",
           "- **미세조정의 종료 보상.** 초기 대조는 충돌/이탈/끝 정지 −10/−8/−8을 유지한다. B의 실패에서는 이탈과 충돌이 함께 늘었다. 정책 보존을 확인한 뒤 이탈만 −8→−9로 바꾸는 단독 시험에서 기존 완주가 정지로 전환되는지도 센다.",
           "- **순수 PPO의 분기 신용 할당.** 구간 커리큘럼의 대조안은 분기당 +2.5 예산에서 올바른 연결 차로를 처음 확보한 뒤 실제 통과했을 때의 준비분 +0.5와 통과분 +2.0을 구별하는 것이다. 분기 접근 구간에 부적합 차로로 진입한 주행만 준비 시 +0.5를 한 번 선지급하고 통과 시 +2.0을 준다. 처음부터 적합한 차로였던 주행은 이탈·복귀해도 준비분을 지급하지 않고 정상 통과 시 +2.5를 준다. 이탈 후 복귀로 준비분을 반복 수령하는 횟수와 실제 분기 통과율을 기록한다.",
           "- **순수 PPO의 정지 처리.** 36건 중 35건은 10초 조건이 실제로 성립했다. 다음 대조에서는 첫 발동 시 −7을 한 번 주고 −0.02/스텝도 유지하되, 그 시점에 종료하는 조건과 원래의 물리적 종료 또는 1,000스텝까지 계속하는 조건을 비교한다. 실제 앞차가 막는 정체에는 추가 벌점을 부과하지 않는다. 완주와 실제 이탈, 30초 전진 거리, 10초 종료에 해당했던 케이스의 최종 결과를 함께 본다. 충돌 벌점을 한꺼번에 낮추면 이 효과와 구별하기 어렵다.",
           "","[진단 스크립트](../../utils/jys3_diagnose_ppo.py) · [집계·층별 변화·체크포인트별 지표](../../results/jys3_relu_ppo_comparison/diagnosis/analysis.json) · [원래의 5종 평가](jys3-relu-ppo-comparison.md)"]
    (final.DOCS/"jys3-ppo-failure-diagnosis.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    run.check_sources()
    records=collect(args.workers)
    dump("replay.json",records)
    recover_training_segments()
    analyze(records)
    report(records)
