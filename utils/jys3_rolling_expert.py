"""Rolling-merge expert and paired evaluation on the physical-fork high-traffic panel."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np

if __name__ == '__main__' and '--gui' in sys.argv:
    os.environ['IC_LIBSUMO'] = '0'
from env.sumo_env import traci
from utils import jys3_train_physical_imitation as baseline
from utils.jys3_gap_bc_eval import required_direction

ROOT = baseline.ROOT.parent / 'jys3_rolling_expert'
BASE = 95380000
DT = 0.2
OLD = baseline.data.bc.expert


def _decision(obs, env, mode):
    if obs.shape != (50,):
        raise ValueError('A single 50-dimensional policy observation is required')
    original = OLD(obs, env)
    direction = required_direction(obs)
    if direction == 0 or obs[38] or original[1]:
        return original, False, direction, 'unchanged'
    if mode == 'old':
        return original, True, direction, 'old_wait'

    i = 1 + 4*(2-direction)
    speed = float(obs[0])*env.vmax
    limit = max(1., float(obs[31])*env.vmax)
    remaining = 400.*float(obs[39])
    front = float(obs[i])*env.visibility
    front_closing = -float(obs[i+1])*env.vmax
    rear = -float(obs[i+2])*env.visibility
    rear_closing = float(obs[i+3])*env.vmax
    rear_speed = max(0., speed+rear_closing)
    overlap = bool(obs[37 if direction==1 else 36])
    front_req = 3.+.45*speed+max(0.,front_closing)
    rear_req = 3.+.50*rear_speed+max(0.,rear_closing)
    assert overlap or front<=front_req or rear<=rear_req
    lead = max(0.,float(obs[9])*env.visibility)
    closing = -float(obs[10])*env.vmax

    # Only the conditional second trial changes the lateral decision.
    relaxed_room = (mode=='relaxed' and remaining<25. and speed<4. and not overlap
                    and front>front_req and rear>3.+.45*rear_speed+max(0.,rear_closing))
    if relaxed_room and front<lead:
        lead, closing = front, front_closing
    desired_gap = 2.5+max(0.,1.05*speed+speed*closing/(2*np.sqrt(2.2*3.2)))
    interaction = (desired_gap/max(.2,lead))**2 if lead<env.visibility else 0.
    idm = 2.2*(1.-(speed/limit)**4-interaction)
    if relaxed_room:
        accel, cmd, reason = idm, direction, 'low_speed_gap'
    else:
        # Start from the original blocked-merge target, before its 14 m reserve cap.
        target = min(limit,max(2.,np.sqrt(2.8*remaining)))
        if overlap or front<=front_req:
            front_speed = max(0.,speed-front_closing) if front<env.visibility else speed
            target = min(target,max(5.,front_speed-3.5))
            reason = 'yield_alongside'
        elif lead>15. and remaining>30.:
            target = min(limit,max(target,rear_speed+3.))
            reason = 'advance_ahead'
        else:
            reason = 'approach_end'
        stop_cap = np.sqrt(2.*2.5*max(0.,remaining-10.))
        # Gentle speed alignment; following and the stop envelope can brake harder.
        alignment = max(-2.5,(target-speed)/1.3)
        accel = min(idm,alignment,(stop_cap-speed)/1.3)
        cmd = 0
    scale = env.max_accel if accel>=0 else env.max_decel
    return np.array([np.clip(accel/scale,-1.,1.),cmd],np.float32), not relaxed_room, direction, reason


def rolling_expert(obs, env):
    """Strict-room rolling controller. Runtime inputs are the observation and fixed unit scales."""
    return _decision(obs,env,'strict')[0]


def episode(env, route, rep, mode, gui=False, controller=_decision):
    path = f'route_{route:02d}'
    seed = BASE+rep*100+route
    env.ego_depart_lane = str(min(rep%2,env.counts[env.routes[path][0]]-1))
    obs,_ = env.reset(seed=seed,options={'route':path})
    assert env.sim.physical_forks and not env.sim.lc_safety_enabled and np.isclose(env.sim.step_length,DT)
    initial = [env.snapshot[i] for i in (0,1,7,-1)]
    digest = hashlib.sha256(obs.tobytes()).hexdigest()
    if gui:
        view = traci.gui.getIDList()[0]
        traci.vehicle.setColor(env.ego_id,(255,40,135,255))
        traci.gui.trackVehicle(view,env.ego_id)
        traci.gui.setZoom(view,4300)
    key = None
    total = slow = zero = longest = longest_slow = 0
    moves = []
    reasons = Counter()
    for step in range(1,1001):
        assert obs.shape==(52,) and not obs[50:].any()
        action,wait,direction,reason = controller(obs[:50],env,mode)
        reasons[reason]+=1
        edge,lane = env.snapshot[:2]
        speed = float(env.snapshot[7])
        new_key = (edge,lane,direction) if wait else None
        if new_key != key:
            total = slow = zero = 0
        key = new_key
        total = total+1 if wait else 0
        slow = slow+1 if wait and speed<1. else 0
        zero = zero+1 if wait and speed<.001 else 0
        longest = max(longest,total)
        longest_slow = max(longest_slow,slow)
        obs,_,done,cut,info = env.step(action)
        post = env.snapshot
        if post and edge==post[0] and lane!=post[1]:
            moves.append(dict(step=step,edge=edge,source=int(lane),target=int(post[1]),speed=speed,
                              commanded=int(action[1])))
        if done or cut:
            assert info['outcome']!='arrival' or info['route_intact']
            return dict(route=route,seed=seed,rep=rep,start=initial,obs_sha256=digest,steps=step,
                        outcome=info['outcome'],longest_wait=longest,longest_slow=longest_slow,
                        terminal_zero_steps=zero,moves=moves,reasons=dict(reasons))
    raise AssertionError('missing terminal outcome')


def shard(job):
    mode,rep = job
    with baseline.simulator() as env:
        return [episode(env,r,rep,mode) for r in range(1,17)]


def summary(rows):
    def distribution(key):
        values = np.array([r[key]*DT for r in rows])
        return dict(median=float(np.median(values)),p90=float(np.percentile(values,90)),maximum=float(max(values)))
    moves = [m['speed'] for r in rows for m in r['moves']]
    return dict(outcomes=dict(Counter(r['outcome'] for r in rows)),slow5=sum(r['longest_slow']>=25 for r in rows),
                slow10=sum(r['longest_slow']>=50 for r in rows),wait=distribution('longest_wait'),
                slow=distribution('longest_slow'),moves=len(moves),move_speed=float(np.mean(moves)),
                rolling10_18=sum(10.<=v<=18. for v in moves),
                arrival_median=float(np.median([r['steps']*DT for r in rows if r['outcome']=='arrival'])),
                zero_timeouts=sum(r['outcome']=='timeout' and r['terminal_zero_steps']>=25 for r in rows))


def panel(modes,pool):
    jobs=[(m,i) for i in range(8) for m in modes]
    groups={m:[] for m in modes}
    for (mode,_),rows in zip(jobs,pool.map(shard,jobs)):
        groups[mode].extend(rows)
        baseline.v3.save(ROOT/f'{mode}.json',groups[mode])
    return groups


def evaluate():
    ROOT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(8) as pool:
        groups=panel(['old','strict'],pool)
        stats={m:summary(rows) for m,rows in groups.items()}
        if stats['strict']['zero_timeouts']>=2:
            groups.update(panel(['relaxed'],pool))
            stats['relaxed']=summary(groups['relaxed'])
    before=groups['old']
    history=baseline.v3.read(ROOT.parent/'jys3_classroom_gap_bc'/'eval_expert.json')
    def starts(rows):
        return [(r['route'],r['seed'],r['start'],r['obs_sha256']) for r in rows]
    assert len(set(r['seed'] for r in before))==len(before)==128
    assert starts(before)==starts(history)
    assert [(r['steps'],r['outcome']) for r in history]==[(r['steps'],r['outcome']) for r in before]
    assert all(starts(rows)==starts(before) for rows in groups.values())
    winner='strict'
    if 'relaxed' in stats:
        a,b=stats['strict'],stats['relaxed']
        if (b['outcomes'].get('collision',0)<=a['outcomes'].get('collision',0)
            and b['outcomes'].get('off_route',0)<=a['outcomes'].get('off_route',0)
            and b['outcomes'].get('arrival',0)>=a['outcomes'].get('arrival',0)
            and b['slow10']<a['slow10']):
            winner='relaxed'
    chosen=stats[winner]
    outcome=chosen['outcomes']
    target=(outcome.get('arrival',0)>=125 and outcome.get('collision',0)<=1 and outcome.get('off_route',0)==0)
    top=sorted(before,key=lambda r:(-r['longest_slow'],r['outcome']!='timeout',r['seed']))[:3]
    result=dict(metrics=stats,comparison_candidate=winner,numeric_targets_met=target,
                slow10_delta=chosen['slow10']-stats['old']['slow10'],
                gui_cases=[{k:r[k] for k in ('route','seed','outcome','longest_slow')} for r in top],
                historical_failures=[dict(route=r['route'],seed=r['seed'],**{m:rows[i]['outcome'] for m,rows in groups.items()})
                                     for i,r in enumerate(before) if r['outcome']!='arrival'])
    baseline.v3.save(ROOT/'summary.json',result)
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--mode',choices=('old','new','strict','relaxed'),default='new')
    parser.add_argument('--route',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--delay-ms',type=int,default=30)
    args=parser.parse_args()
    if not args.gui:
        evaluate()
        return
    if args.route not in range(1,17) or args.seed is None or args.delay_ms<0:
        parser.error('GUI needs --route 1..16 and a panel --seed, with nonnegative delay.')
    rep,rem=divmod(args.seed-BASE-args.route,100)
    if rem or rep not in range(8):
        parser.error('Seed must match 95380000 + rep*100 + route for rep=0..7.')
    mode=args.mode
    if mode=='new':
        mode=baseline.v3.read(ROOT/'summary.json')['comparison_candidate']
    with baseline.simulator() as env:
        env.sim.gui=True
        env.sim.gui_delay=args.delay_ms
        result=episode(env,args.route,rep,mode,True)
    reference=next(r for r in baseline.v3.read(ROOT/f'{mode}.json') if r['seed']==args.seed)
    assert result==reference
    print(mode,result['seed'],result['outcome'],result['steps'],flush=True)


if __name__=='__main__':
    main()
