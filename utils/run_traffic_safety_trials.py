"""Warm-start existing traffic policy; independent rewards plus candidate search."""
import collections
from utils import run_route_learning_trials as x
from utils import run_observation_reward_trials as t
from env.traffic_safety_trials import TrafficSafetyEnv
OUT=t.ROOT/'results/traffic_safety_warmstart_20261115'
DOC=t.ROOT/'docs/experiments/ic-traffic-safety-results-20261115.md'
CONDITIONS=('baseline','combo','gap','lane')
SOURCE=t.ROOT/'results/route_following_trials_20261114/runs/baseline/policy-200000.pt'

def make_env(condition,evaluation=False,none=False):
    return x.original_make(condition,evaluation=evaluation,none=none)

Log=x.Log
def train(condition,smoke=False):
    a.b.torch.set_num_threads(1)
    directory=OUT/('smoke' if smoke else 'runs')/condition
    if (directory/'complete.json').exists():return read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Incomplete run preserved; no implicit resume: '+str(directory))
    directory.mkdir(parents=True)
    manifest=read(OUT/'manifest.json')
    for f,h in manifest['source_hashes'].items():assert a.b.digest(ROOT/f)==h,('source changed',f)
    budget=32 if smoke else 200000
    hp=dict(read(a.r.sources('baseline',22)/'manifest.json')['hp'],seed=SEED,device='cpu')
    env=make_env(condition);agent=a.b.PPO(obs_dim=63,act_dim=2,buffer=RolloutBuffer(),**hp)
    agent.load(str(SOURCE))
    save(directory/'manifest.json',dict(condition=condition,hp=hp,budget=budget,parent_policy=str(SOURCE),parent_sha=a.b.digest(SOURCE),optimizer_reset=True,steps_are_additional=True,initial_tensor_sha=policy_hash(agent),simulation=dict(step_length=.2,warmup_steps=4500,max_steps=4500),reward=env.rw,action_safety=env.lc_safety_enabled,observation_dim=63,source_commit=manifest['commit'],source_hashes=manifest['source_hashes']))
    agent.save(str(directory/'initial.pt'))
    agent.save(str(directory/'policy-0.pt'))
    logger=Log(directory,agent,budget,env)
    try:
        agent.learn(env,budget,logger=logger,eval_interval=0,exact_budget=True)
        assert agent.num_timesteps==budget and env.route_override_count==0
        agent.save(str(directory/'model.pt'))
        save(directory/'training-complete.json',dict(steps=budget,episode_seeds=env.episode_seeds,model_sha=a.b.digest(directory/'model.pt')))
    finally:env.close();logger.close()
    rows=[]
    for checkpoint in sorted(directory.glob('policy-*.pt'),key=lambda p:int(p.stem.split('-')[1])):
        rows.extend(evaluate(condition,checkpoint,directory,smoke))
    result=dict(condition=condition,steps=budget,model_sha=a.b.digest(directory/'model.pt'),evaluations=len(rows))
    save(directory/'complete.json',result);save(directory/'progress.json',dict(stage='complete',**result))
    return result
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
                    trace=t.a.b.TimeoutTrace();counts=collections.Counter();teleports=0;bonus=0.;decisions={};tail=collections.deque(maxlen=25);first_state=None
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
                        safety=env.safety_context(action)
                        state=dict(t=env.step_count*.2,edge=edge,lane=lane,speed=speed,action=action.tolist(),safety=safety,obs=obs.tolist())
                        if first_state is None:first_state=state.copy()
                        risk=safety['ttc'] is not None and safety['ttc']<3
                        counts['risk_steps']+=int(risk);counts['risk_brake_requests']+=int(risk and action[0]<0)
                        counts['risky_lane_requests']+=int(safety['lane']>.5)
                        obs,_,done,cut,info=env.step(action)
                        if env.ego_id in v.getIDList():
                            state['after_speed']=v.getSpeed(env.ego_id)
                            counts['risk_actual_decelerations']+=int(risk and state['after_speed']<speed-1e-5)
                        tail.append(state)
                        bonus+=info.get('route_selection_reward',0);teleports+=tr.simulation.getStartingTeleportNumber()
                        if constrained and env.ego_id in v.getIDList():
                            after_edge=v.getRoadID(env.ego_id);after_lane=v.getLaneIndex(env.ego_id)
                            if after_edge==edge:
                                counts['toward_actual_changes']+=int(wrong and (after_lane-lane)*needed>0)
                                counts['left_correct_lane']+=int(not wrong and after_lane not in targets)
                            else:counts['passed_decisions']+=1;counts['passed_from_correct_lane']+=int(not wrong)
                        if done or cut:break
                    rows.append(dict(row,outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit',seconds=env.step_count*.2,teleports=teleports,route_overrides=env.route_override_count,diagnostic=trace.result() if cut else None,following=dict(counts),selection_reward_total=bonus,decisions=list(decisions.values()),reward_components=dict(env.reward_counts),first_state=first_state,tail=list(tail) if info['collided'] or cut else []))
                    assert env.route_override_count==0
                    t.save(directory/(checkpoint.stem+'-partial.json'),rows)
                    t.save(directory/'progress.json',dict(stage='evaluation',pid=t.os.getpid(),checkpoint=checkpoint.name,completed=len(rows),target=2 if smoke else 64))
        finally:env.close()
    t.save(directory/(checkpoint.stem+'-evaluation.json'),rows)
    return rows

def publish():
    lines=['# 교통 정책 이어학습: 안전보상 독립 비교 및 조합 탐색','', '원본 교통정책20만 스텝 보존. 표의 스텝은 추가학습 스텝이며0은 출발 정책이다. Adam 상태는 새로 시작. 모든 조건 reference교통 학습, seed11. 평가seed39101~39103. 강제보정 없음.','', '|조건|상태|추가 스텝|','|---|---|---:|']
    for c in CONDITIONS:
        p=OUT/'runs'/c/'progress.json';v=read(p) if p.exists() else {}
        lines.append(f"|{c}|{v.get('stage','대기')}|{v.get('steps','')}|")
    lines+=['','|조건|추가학습 모델|교통|완주|충돌|기타|위험시 감속요청/위험표본|실제감속|','|---|---|---|---:|---:|---:|---:|---:|']
    for p in sorted(OUT.glob('runs/*/policy-*-evaluation.json')):
        rows=read(p)
        for mode in ('none','reference'):
            g=[r for r in rows if r['traffic']==mode];a=sum(r['outcome']=='arrival' for r in g);b=sum(r['outcome']=='collision' for r in g);ct=collections.Counter()
            for row in g:ct.update(row.get('following',{}))
            lines.append(f"|{p.parent.name}|{p.stem}|{mode}|{a}/{len(g)}|{b}|{len(g)-a-b}|{ct['risk_brake_requests']}/{ct['risk_steps']}|{ct['risk_actual_decelerations']}|")
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf8')

def configure():
    t.OUT=OUT;t.DOC=DOC;t.CONDITIONS=CONDITIONS;t.SEED=11;t.EVAL_SEEDS=(39101,39102,39103)
    t.TrialEnv=TrafficSafetyEnv;t.make_env=make_env
    t.SOURCE_FILES=(*t.SOURCE_FILES,'env/traffic_safety_trials.py','utils/run_traffic_safety_trials.py')
configure()
argparse=t.argparse;os=t.os;shutil=t.shutil;subprocess=t.subprocess;cf=t.cf
ROOT=t.ROOT;BRANCH=t.BRANCH;SEED=t.SEED;EVAL_SEEDS=t.EVAL_SEEDS;SOURCE_FILES=t.SOURCE_FILES
a=t.a;save=t.save;read=t.read;policy_hash=t.policy_hash;RolloutBuffer=t.RolloutBuffer
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if shutil.disk_usage(OUT).free<20*1024**3:raise RuntimeError('Insufficient free disk')
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH
        for mode in ('none','reference'):
            dest=OUT/'inputs'/mode;dest.mkdir(parents=True,exist_ok=True)
            for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):shutil.copy2(ROOT/'results/neighbor_fix_20261114/inputs'/mode/name,dest/name)
        save(OUT/'manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes={p:a.b.digest(ROOT/p) for p in SOURCE_FILES},inputs={m:{n:a.b.digest(OUT/'inputs'/m/n) for n in ('network.net.xml','highway.rou.xml','run.sumocfg')} for m in ('none','reference')},seed=SEED,eval_seeds=EVAL_SEEDS,conditions=CONDITIONS,workers=3))
        if not args.smoke:
            assert all((OUT/'smoke'/c/'complete.json').exists() for c in CONDITIONS),'Smoke gate required'
            hashes={read(OUT/'smoke'/c/'manifest.json')['initial_tensor_sha'] for c in CONDITIONS};assert len(hashes)==1
            for c in CONDITIONS:assert read(OUT/'smoke'/c/'manifest.json')['source_hashes']==read(OUT/'manifest.json')['source_hashes'],'Smoke used different code'
        save(OUT/'status.json',dict(stage='smoke' if args.smoke else 'running',pid=os.getpid()))
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(train,c,args.smoke):c for c in CONDITIONS}
            for future in cf.as_completed(futures):
                try:result=future.result()
                except Exception:
                    (OUT/'STOP').write_text('Peer trial failed',encoding='utf8')
                    raise
                print(result,flush=True)
                if not args.smoke:publish()
        if not args.smoke:
            hashes={read(OUT/'runs'/c/'manifest.json')['initial_tensor_sha'] for c in CONDITIONS};assert len(hashes)==1
        save(OUT/'status.json',dict(stage='smoke_complete' if args.smoke else 'complete'))
    except Exception as exc:save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=="__main__":main()
