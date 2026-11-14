"""Repeated seeds and frozen-policy traffic transfer; no action intervention."""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
from pathlib import Path
from utils import run_route_learning_trials as x
from utils import run_observation_reward_trials as t

OUT=t.ROOT/'results/route_repeat_transfer_20261114'
DOC=t.ROOT/'docs/experiments/ic-route-repeat-transfer-20261114.md'
CONDITIONS=('baseline','guidance','dwell')
SEEDS=(22,33)
EVAL_SEEDS=(29101,29102,29103)
SOURCE_FILES=(*t.SOURCE_FILES,'utils/run_route_repeat_transfer.py')

def make_env(condition,evaluation=False,none=False):
    return x.original_make(condition,evaluation=evaluation,none=none if evaluation else True)

def job(seed,condition,smoke=False):
    t.OUT=OUT/f'seed{seed}';t.SEED=seed;t.EVAL_SEEDS=EVAL_SEEDS
    t.make_env=make_env;t.evaluate=evaluate
    t.a.b.torch.set_num_threads(1)
    if seed!=11 or smoke:return t.train(condition,smoke)
    # Existing seed11 is evaluated without training or replacing its provenance.
    dest=t.OUT/'runs'/condition
    if (dest/'complete.json').exists():return t.read(dest/'complete.json')
    dest.mkdir(parents=True,exist_ok=True)
    source=x.OUT/'runs'/condition
    shutil.copy2(source/'manifest.json',dest/'manifest.json')
    rows=[]
    for policy in sorted(source.glob('policy-*.pt'),key=lambda p:int(p.stem.split('-')[1])):
        rows.extend(evaluate(condition,policy,dest))
    result=dict(stage='complete',condition=condition,training_seed=seed,evaluations=len(rows),mode='frozen_policy_transfer')
    t.save(dest/'complete.json',result);t.save(dest/'progress.json',result)
    return result

def publish():
    lines=['# 경로학습 반복 seed 및 교통 전이 평가','','학습: 무교통, seed22/33, 각20만 스텝. seed11은 기존 정책 재사용. 평가 seed29101/29102/29103은 이전 후보 평가와 분리. 모든 중간 모델을 무교통16회+기본교통48회로 비교. 교통 추가 재학습은 아직 하지 않는다. 강제보정 없음.','','|학습 seed|조건|상태|스텝|','|---|---|---|---:|']
    for seed in (22,33,11):
        for c in CONDITIONS:
            p=OUT/f'seed{seed}/runs'/c/'progress.json';v=t.read(p) if p.exists() else {}
            lines.append(f"|{seed}|{c}|{v.get('stage','대기')}|{v.get('steps','')}|")
    lines+=['','|학습 seed|조건|중간 모델|교통|완주|충돌|기타|','|---|---|---|---|---:|---:|---:|']
    for p in sorted(OUT.glob('seed*/runs/*/policy-*-evaluation.json')):
        rows=t.read(p)
        for mode in ('none','reference'):
            g=[r for r in rows if r['traffic']==mode]
            a=sum(r['outcome']=='arrival' for r in g);b=sum(r['outcome']=='collision' for r in g)
            lines.append(f'|{p.parents[2].name}|{p.parent.name}|{p.stem}|{mode}|{a}/{len(g)}|{b}|{len(g)-a-b}|')
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        assert shutil.disk_usage(OUT).free>20*1024**3
        assert subprocess.check_output(['git','branch','--show-current'],cwd=t.ROOT,text=True).strip()==t.BRANCH
        manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=t.ROOT,text=True).strip(),source_hashes={f:t.a.b.digest(t.ROOT/f) for f in SOURCE_FILES},conditions=CONDITIONS,train_seeds=SEEDS,eval_seeds=EVAL_SEEDS,workers=3,training_traffic='none',evaluation_traffic=['none','reference'])
        for seed in (11,22,33):
            root=OUT/f'seed{seed}'
            for mode in ('none','reference'):
                dest=root/'inputs'/mode;dest.mkdir(parents=True,exist_ok=True)
                for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):shutil.copy2(x.OUT/'inputs'/mode/name,dest/name)
            t.save(root/'manifest.json',manifest)
        if not args.smoke:
            for seed in SEEDS:
                hashes=set()
                for c in CONDITIONS:
                    folder=OUT/f'seed{seed}/smoke'/c
                    assert (folder/'complete.json').exists()
                    m=t.read(folder/'manifest.json');assert m['source_hashes']==manifest['source_hashes'];hashes.add(m['initial_tensor_sha'])
                assert len(hashes)==1
        t.save(OUT/'manifest.json',manifest);t.save(OUT/'status.json',dict(stage='smoke' if args.smoke else 'running',pid=os.getpid()))
        specs=[(s,c,args.smoke) for s in SEEDS for c in CONDITIONS]
        if not args.smoke:specs.extend((11,c,False) for c in CONDITIONS)
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(job,*spec) for spec in specs]
            for f in cf.as_completed(futures):
                print(f.result(),flush=True)
                if not args.smoke:publish()
        t.save(OUT/'status.json',dict(stage='smoke_complete' if args.smoke else 'complete'))
    except Exception as exc:
        for seed in (11,22,33):(OUT/f'seed{seed}'/'STOP').write_text(str(exc),encoding='utf8')
        t.save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

import collections
def evaluate(condition,checkpoint,directory,smoke=False):
    m=t.read(directory/'manifest.json')
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(checkpoint))
    rows=[];tr=t.a.b.traci;v=tr.vehicle
    for none in (True,False):
        env=t.make_env(condition,evaluation=True,none=none)
        if smoke:env.max_steps=50
        try:
            for route in (['route_13'] if smoke else sorted(env.routes)):
                for seed in (t.EVAL_SEEDS[:1] if none or smoke else t.EVAL_SEEDS):
                    row=dict(condition=condition,checkpoint=checkpoint.name,route=route,seed=seed,traffic='none' if none else 'reference')
                    try:
                        env.reward_counts.clear()
                        obs,_=env.reset(seed=seed,options={'route':route})
                    except RuntimeError as exc:
                        if '삽입 실패' not in str(exc):raise
                        rows.append(dict(row,outcome='insertion_failure',error=str(exc)));continue
                    trace=t.a.b.TimeoutTrace();counts=collections.Counter();teleports=0;bonus=0.;decisions={}
                    while True:
                        if (t.OUT/'STOP').exists():raise RuntimeError('Stopped after peer failure')
                        if env.step_count%5==0:trace.sample(env)
                        nav=env.navigation_context();edge=v.getRoadID(env.ego_id);lane=v.getLaneIndex(env.ego_id)
                        pos=v.getLanePosition(env.ego_id);remaining=tr.lane.getLength(v.getLaneID(env.ego_id))-pos
                        speed=v.getSpeed(env.ego_id);targets=nav['current_targets'];wrong=bool(targets) and lane not in targets
                        action=agent.predict(obs,deterministic=True);direction=env._quantize_lane_change(action[1]);needed=env.navigation.offset(lane,targets)
                        constrained=bool(targets) and len(targets)<env.navigation.counts.get(edge,0) and remaining<=300
                        if constrained:
                            counts['decision_steps']+=1;counts['correct_lane_steps']+=int(not wrong)
                            counts['moving_decision_steps']+=int(speed>.5);counts['moving_correct_lane_steps']+=int(speed>.5 and not wrong)
                            if wrong:
                                counts['wrong_lane_steps']+=1;counts['toward_request_steps']+=int(direction*needed>0)
                                counts['moving_wrong_lane_steps']+=int(speed>.5)
                                counts['moving_toward_request_steps']+=int(speed>.5 and direction*needed>0)
                            decisions.setdefault((edge,v.getRouteIndex(env.ego_id)),dict(edge=edge,entry_remaining=remaining,entry_correct=not wrong))
                        obs,_,done,cut,info=env.step(action)
                        bonus+=info.get('route_selection_reward',0);teleports+=tr.simulation.getStartingTeleportNumber()
                        if constrained and env.ego_id in v.getIDList():
                            after_edge=v.getRoadID(env.ego_id);after_lane=v.getLaneIndex(env.ego_id)
                            if after_edge==edge:
                                counts['toward_actual_changes']+=int(wrong and (after_lane-lane)*needed>0)
                                counts['left_correct_lane']+=int(not wrong and after_lane not in targets)
                            else:counts['passed_decisions']+=1;counts['passed_from_correct_lane']+=int(not wrong)
                        if done or cut:break
                    rows.append(dict(row,outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit',seconds=env.step_count*.2,teleports=teleports,route_overrides=env.route_override_count,diagnostic=trace.result() if cut else None,following=dict(counts),selection_reward_total=bonus,decisions=list(decisions.values()),reward_components=dict(env.reward_counts)))
                    t.save(directory/'progress.json',dict(stage='evaluation',pid=t.os.getpid(),checkpoint=checkpoint.name,completed=len(rows),target=2 if smoke else 64))
        finally:env.close()
    t.save(directory/(checkpoint.stem+'-evaluation.json'),rows)
    return rows

if __name__=="__main__":main()
