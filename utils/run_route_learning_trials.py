"""Isolated no-traffic campaign, max three concurrent CPU workers."""
import collections
from utils import run_route_following_trials as r
from utils import run_observation_reward_trials as t
from env.route_learning_trials import RouteLearningEnv
from env.route_following_trials import PreparedNavigation
OUT=t.ROOT/'results/route_learning_none_20261114'
DOC=t.ROOT/'docs/experiments/ic-route-learning-none-results-20261114.md'
CONDITIONS=('baseline','compact','route_reward','scale5','dwell','guidance')
original_make=t.make_env
original_log=t.Log

def make_env(condition,evaluation=False,none=True):
    env=original_make(condition,evaluation=evaluation,none=True)
    assert env.route_override_count==0
    return env

class Log(original_log):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.next_checkpoint=min(50000,self.budget)
    def log_train(self,step,metrics,verbose=False):
        boundary=self.next_checkpoint
        for key,value in self.env.reward_counts.items():
            self.writer.add_scalar('route_components/'+key,value,step)
        with (self.directory/'reward-components.jsonl').open('a',encoding='utf8') as f:
            f.write(t.json.dumps(dict(step=step,**self.env.reward_counts))+'\n')
        super().log_train(step,metrics,False)
        if step>=boundary:self.next_checkpoint=boundary+50000

def evaluate(condition,checkpoint,directory,smoke=False):
    m=t.read(directory/'manifest.json')
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(checkpoint))
    rows=[];tr=t.a.b.traci;v=tr.vehicle
    for none in (True,):
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
                    t.save(directory/'progress.json',dict(stage='evaluation',pid=t.os.getpid(),checkpoint=checkpoint.name,completed=len(rows),target=1 if smoke else 16))
        finally:env.close()
    t.save(directory/(checkpoint.stem+'-evaluation.json'),rows)
    return rows

def publish():
    rows=[row for p in (OUT/'runs').glob('*/policy-*-evaluation.json') for row in t.read(p)]
    lines=['# 무교통 경로학습 독립 비교','', '16경로, seed11, 각20만 스텝, 강제보정 없음. 5만 스텝 경계 이후 첫 PPO 업데이트에서 중간 저장. 입력63차원·초기가중치·PPO 동일. 학습 보상 성분은 각 run의 reward-components.jsonl 및 TensorBoard에 누적 기록.','', '|조건|상태|스텝|','|---|---|---:|']
    for c in CONDITIONS:
        p=OUT/'runs'/c/'progress.json';v=t.read(p) if p.exists() else {}
        lines.append(f"|{c}|{v.get('stage','대기')}|{v.get('steps','')}|")
    lines+=['','|조건|모델|완주/16|충돌|기타|','|---|---|---:|---:|---:|']
    for c,cp in sorted({(x['condition'],x['checkpoint']) for x in rows}):
        group=[x for x in rows if (x['condition'],x['checkpoint'])==(c,cp)]
        a=sum(x['outcome']=='arrival' for x in group);b=sum(x['outcome']=='collision' for x in group)
        lines.append(f'|{c}|{cp}|{a}/{len(group)}|{b}|{len(group)-a-b}|')
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf8')


def configure():
    t.OUT=OUT;t.DOC=DOC;t.CONDITIONS=CONDITIONS
    t.TrialEnv=RouteLearningEnv;t.RouteNavigation=PreparedNavigation
    t.make_env=make_env;t.Log=Log;t.evaluate=evaluate;t.publish=publish
    t.SOURCE_FILES=(*t.SOURCE_FILES,'env/route_learning_trials.py','utils/run_route_learning_trials.py')

configure()

argparse=t.argparse;os=t.os;shutil=t.shutil;subprocess=t.subprocess;cf=t.cf
ROOT=t.ROOT;BRANCH=t.BRANCH;SEED=t.SEED;EVAL_SEEDS=t.EVAL_SEEDS;SOURCE_FILES=t.SOURCE_FILES
a=t.a;save=t.save;read=t.read;train=t.train
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

