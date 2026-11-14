"""Reuse frozen experiment runner for three isolated route-following conditions."""
import collections
from utils import run_observation_reward_trials as t
from env.route_following_trials import PreparedNavigation, RouteTrialEnv

ORIGINAL_OUT=t.OUT
OUT=t.ROOT/'results/route_following_trials_20261114'

def evaluate(condition,checkpoint,directory,smoke=False):
    m=t.read(directory/'manifest.json')
    agent=t.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(checkpoint))
    rows=[];tr=t.a.b.traci;v=tr.vehicle
    for none in (False,True):
        env=t.make_env(condition,evaluation=True,none=none)
        if smoke:env.max_steps=50
        try:
            for route in (['route_13'] if smoke else sorted(env.routes)):
                for seed in (t.EVAL_SEEDS[:1] if none or smoke else t.EVAL_SEEDS):
                    row=dict(condition=condition,checkpoint=checkpoint.name,route=route,seed=seed,traffic='none' if none else 'reference')
                    try:obs,_=env.reset(seed=seed,options={'route':route})
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
                    rows.append(dict(row,outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit',seconds=env.step_count*.2,teleports=teleports,route_overrides=env.route_override_count,diagnostic=trace.result() if cut else None,following=dict(counts),selection_reward_total=bonus,decisions=list(decisions.values())))
                    t.save(directory/'progress.json',dict(stage='evaluation',pid=t.os.getpid(),checkpoint=checkpoint.name,completed=len(rows),target=2 if smoke else 64))
        finally:env.close()
    t.save(directory/(checkpoint.stem+'-evaluation.json'),rows)
    return rows

def publish():
    t.DOC= t.ROOT/'docs/experiments/ic-route-following-results-20261114.md'
    original_publish()
    rows=[r for p in (OUT/'runs').glob('*/policy-*-evaluation.json') for r in t.read(p)]
    lines=['','## 안내를 따르는 행동 지표','','정지 스텝의 과대표현을 피하기 위해 주행 중(0.5m/s 초과) 비율을 별도 표시한다. 요청률은 실제 변경 성공률이 아니다.','','|조건|모델|교통|주행 중 필요방향 요청/필요 스텝|주행 중 올바른 차로/판정 스텝|목표 방향 실제 변경|맞는 차로 이탈|','|---|---|---|---:|---:|---:|---:|']
    for c,cp,traffic in sorted({(r['condition'],r['checkpoint'],r['traffic']) for r in rows}):
        total=collections.Counter()
        for r in rows:
            if (r['condition'],r['checkpoint'],r['traffic'])==(c,cp,traffic):total.update(r.get('following',{}))
        lines.append(f"|{c}|{cp}|{traffic}|{total['moving_toward_request_steps']}/{total['moving_wrong_lane_steps']}|{total['moving_correct_lane_steps']}/{total['moving_decision_steps']}|{total['toward_actual_changes']}|{total['left_correct_lane']}|")
    with t.DOC.open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')

original_publish=t.publish

def configure():
    t.OUT=OUT;t.DOC=t.ROOT/'docs/experiments/ic-route-following-results-20261114.md'
    t.CONDITIONS=('baseline','guidance','route_reward')
    t.TrialEnv=RouteTrialEnv;t.RouteNavigation=PreparedNavigation
    t.evaluate=evaluate;t.publish=publish
    t.SOURCE_FILES=(*t.SOURCE_FILES,'env/route_following_trials.py','utils/run_route_following_trials.py')

# Windows spawn imports the module before invoking the inherited train function.
configure()
if __name__=='__main__':t.main()
