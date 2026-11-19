"""PPO-only adaptation of Siwon's MCW model. No teacher labels or action overrides."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT/'results/entryprep_runtime'
if runtime.exists():
    sys.path.insert(0, str(runtime))
try:
    import libsumo
except ImportError:
    pass
else:
    os.environ['IC_LIBSUMO'] = '1'

import numpy as np
import torch
from algorithms.ppo import PPO
from env.mcw_policy_only import PolicyOnlyEnv
from env.sumo_env import traci
from env.mdp_config import OBSERVATION
from utils.mcw_train import prepare_scenario, load_network_metadata, build_configs, make_env, ROUTES


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def make(args):
    a = SimpleNamespace(stage=args.stage, safety=args.mode=='original', min_front_gap=3.5,
        min_rear_gap=3.5, rear_tau=.35, approach_horizon=1.2, collision_penalty=15.,
        arrival_bonus=5., lc_penalty=.01, smoke=False, seed=args.seed, route_conn=True,
        ttc_reward=True, prep_reward=True, nav_horizon=400., timeout_penalty=15., stall_penalty=.08)
    cfg = prepare_scenario(args.stage, ROOT/'results/mcw_policy_only/scenarios')
    road, nav = load_network_metadata(cfg.parent/'network.net.xml', cfg.parent/'highway.rou.xml')
    ac, rw, sim, ego, traffic = build_configs(a)
    if args.mode in ('original', 'no_gate'):
        ac['lane_change_safety']['enabled'] = args.mode=='original'
        env = make_env(cfg, road, nav, a, ac, rw, sim, ego, traffic)
    else:
        ac['lane_change_safety']['enabled'] = False
        if args.train:
            sim['max_steps'] = 1200
        env = PolicyOnlyEnv(str(cfg), road, ego, sim, OBSERVATION, ac, rw, traffic, (), args.seed,
            navigation=nav, clarity='baseline', ttc_reward=True,
            side_risk_penalty=getattr(args,'side_risk_penalty',.3),
            acceleration_risk_penalty=getattr(args,'acceleration_risk_penalty',0.),
            future_goal_coef=getattr(args,'future_goal_coef',0.),
            invalid_command_penalty=getattr(args,'invalid_command_penalty',.01))
        if args.train and getattr(args,'empty_fraction',0.)>0:
            env.training_traffic_cfg = str(cfg)
            env.training_clear_cfg = str(prepare_scenario('none', ROOT/'results/mcw_policy_only/scenarios'))
    params = dict(lr=args.lr, n_steps=2048, n_epochs=5, minibatch_size=256, gamma=getattr(args,'gamma',.995),
        gae_lambda=getattr(args,'gae_lambda',.95), clip_eps=.15, hidden_sizes=(256,256), policy_type='hybrid',
        value_coef=getattr(args,'value_coef',.5),
        lane_entropy_coef=args.entropy, init_log_std=-1., seed=args.seed, device='cpu')
    agent = PPO(obs_dim=env.observation_space.shape[0], act_dim=2, **params)
    source = Path(args.resume)
    if not source.exists() and source.as_posix()=='results/mcw_runs/stage3_high_v5_b/best_model.pt':
        import zipfile
        with zipfile.ZipFile(ROOT/'results/mcw_runs/mcw-models-20261116.zip') as archive:
            payload=archive.read(source.as_posix())
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_bytes(payload)
    weights = torch.load(source, map_location='cpu', weights_only=True)
    own = agent.policy.state_dict()
    for key, value in weights.items():
        if own[key].shape == value.shape:
            own[key] = value
        elif key=='trunk.0.weight':
            if value.shape[1] > own[key].shape[1]:
                raise ValueError('Refusing to silently truncate observation weights')
            own[key][:, :value.shape[1]] = value
            own[key][:, 31:34] = 0.
            own[key][:, 37:47] = 0.
            if args.train and not getattr(args,'adapt_risk',False):
                own[key][:, value.shape[1]:] *= .15
            else:
                own[key][:, value.shape[1]:] = 0.
        else:
            raise ValueError(f'Checkpoint schema mismatch: {key}')
    agent.policy.load_state_dict(own)
    risk_columns = []
    if getattr(args, 'adapt_risk', False):
        if args.mode != 'road_only':
            raise ValueError('Risk adaptation applies only to road-only policy')
        for parameter in agent.policy.parameters():
            parameter.requires_grad_(False)
        first = agent.policy.trunk[0].weight
        first.requires_grad_(True)
        risk_columns=env.traffic_input_columns()
        mask=torch.zeros_like(first)
        mask[:,risk_columns]=1.
        first.register_hook(lambda grad: grad*mask)
        for parameter in agent.policy.v_head.parameters():
            parameter.requires_grad_(True)
    if getattr(args,'adapt_risk',False) or getattr(args,'detach_value',False):
        # Fit the value head without letting its large early traffic errors
        # rewrite the shared traffic-risk features used by the actor.
        agent.policy.v_head.register_forward_pre_hook(lambda module, inputs: (inputs[0].detach(),))
        agent.optimizer=torch.optim.Adam([
            dict(params=[parameter for name,parameter in agent.policy.named_parameters()
                if parameter.requires_grad and not name.startswith('v_head.')],lr=args.lr),
            dict(params=list(agent.policy.v_head.parameters()),lr=args.lr*getattr(args,'critic_lr_mult',1.))])
    if args.train and args.reset_std:
        with torch.no_grad():
            agent.policy.log_std.fill_(-1.5)
    protocol = dict(base_commit='8eefd3af670e30604f1515096bdd8c0373f3fd63', args=vars(args),
        initial_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), hparams=params,
        action=ac, simulation=sim, observation_dim=env.observation_space.shape[0],
        road_keys=getattr(env,'road_keys',None), backend=traci.__name__, teacher_labels=False,
        scenario_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cfg.parent.iterdir() if p.is_file()})
    protocol['code_hashes'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'env/mcw_policy_only.py','utils/mcw_policy_only.py','utils/mcw_train.py',
        'env/sumo_env.py','algorithms/ppo.py','utils/networks.py','utils/buffer.py')}
    protocol['risk_adaptation_columns']=risk_columns
    protocol['critic_detached'] = bool(getattr(args,'adapt_risk',False) or getattr(args,'detach_value',False))
    if args.mode == 'road_only':
        protocol['observation_schema_half_open'] = env.observation_schema()
        protocol['reward_options'] = dict(side_risk_penalty=env.side_risk_penalty,
            acceleration_risk_penalty=env.acceleration_risk_penalty,
            future_goal_coef=env.future_goal_coef,
            invalid_command_penalty=env.invalid_command_penalty)
        if hasattr(env,'training_clear_cfg'):
            clear=Path(env.training_clear_cfg).parent
            protocol['training_clear_scenario_hashes'] = {
                p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in clear.iterdir() if p.is_file()}
    protocol['versions'] = dict(python=sys.version.split()[0], torch=torch.__version__,
        numpy=np.__version__, sumo='Recorded from running simulator in runtime.json')
    return env, agent, protocol


def evaluate(env, agent, args, out):
    rows=[]
    for ep in range(args.episodes):
        for r, route in enumerate(ROUTES):
            if r+1 not in args.routes:
                continue
            seed=args.eval_seed+ep*100+r
            try:
                obs,_=env.reset(seed=seed, options={'route':route})
            except RuntimeError as exc:
                rows.append(dict(route=route, seed=seed, outcome='insertion_failure', error=str(exc)))
                write(out/'eval.json',dict(status='running',total=len(rows),rows=rows))
                continue
            torch.manual_seed(seed)
            write(out/'runtime.json',dict(sumo_version=traci.getVersion(),backend=traci.__name__))
            edge=None
            recent=collections.deque(maxlen=10)
            while True:
                previous_edge=edge
                edge=traci.vehicle.getRoadID(env.ego_id)
                if args.trace and (edge!=previous_edge or env.step_count%100==0):
                    with torch.no_grad():
                        logits=agent.policy(torch.as_tensor(obs).unsqueeze(0))[2][0]
                        probabilities=torch.softmax(logits,dim=-1).tolist()
                    record=dict(route=route,seed=seed,seconds=env.step_count*.2,edge=edge,
                        lane=traci.vehicle.getLaneIndex(env.ego_id),position=traci.vehicle.getLanePosition(env.ego_id),
                        speed=traci.vehicle.getSpeed(env.ego_id),lane_probabilities_left_keep_right=probabilities,
                        action=agent.predict(obs).tolist(),observation=obs.tolist(),last_inputs=obs[-8:].tolist())
                    position=np.asarray(traci.vehicle.getPosition(env.ego_id))
                    record['nearby_debug']=[]
                    for ident in traci.vehicle.getIDList():
                        if ident!=env.ego_id and np.linalg.norm(np.asarray(traci.vehicle.getPosition(ident))-position)<30.:
                            record['nearby_debug'].append(dict(id=ident,lane=traci.vehicle.getLaneID(ident),
                                position=traci.vehicle.getLanePosition(ident),speed=traci.vehicle.getSpeed(ident)))
                    with (out/'trace.jsonl').open('a',encoding='utf8') as f:
                        f.write(json.dumps(record)+'\n')
                action=agent.predict(obs, deterministic=not args.stochastic)
                if args.trace:
                    recent.append(dict(seconds=env.step_count*env.step_length,edge=edge,
                        lane=traci.vehicle.getLaneID(env.ego_id),
                        position=traci.vehicle.getPosition(env.ego_id),
                        speed=traci.vehicle.getSpeed(env.ego_id),
                        action=action.tolist(),observation=obs.tolist()))
                obs,rew,done,cut,info=env.step(action)
                if done or cut:
                    outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'other'
                    rows.append(dict(route=route, seed=seed, outcome=outcome, seconds=env.step_count*.2,
                        edge=edge, safety_enabled=env.lc_safety_enabled))
                    if args.trace and outcome!='arrival':
                        with (out/'failures.jsonl').open('a',encoding='utf8') as f:
                            f.write(json.dumps(dict(route=route,seed=seed,outcome=outcome,
                                last_two_seconds=list(recent)))+'\n')
                    break
            totals=dict(collections.Counter(x['outcome'] for x in rows))
            write(out/'eval.json',dict(status='running',total=len(rows),totals=totals,rows=rows))
            print(json.dumps(dict(evaluation=len(rows),route=route,outcome=outcome,totals=totals)),flush=True)
    per_route=[]
    for route in ROUTES:
        group=[x for x in rows if x['route']==route]
        if not group:
            continue
        counts=dict(collections.Counter(x['outcome'] for x in group))
        per_route.append(dict(route=route, total=len(group), counts=counts,
            arrival_rate=counts.get('arrival',0)/len(group), collision_rate=counts.get('collision',0)/len(group)))
    totals=dict(collections.Counter(x['outcome'] for x in rows))
    success=totals.get('arrival',0)/len(rows)
    collision=totals.get('collision',0)/len(rows)
    write(out/'eval.json',dict(status='complete',total=len(rows),totals=totals,rows=rows,
        per_route=per_route,arrival_rate=success,collision_rate=collision,
        aggregate_target_met=success>=.95 and collision<.05,
        primary_goal_confirmed=(args.split=='heldout' and args.stage=='high_real_ratio' and len(rows)>=80 and len(per_route)==16
            and success>=.95 and collision<.05),
        every_route_empirical_target_met=(len(per_route)==16 and all(
            r['arrival_rate']>=.95 and r['collision_rate']<.05 for r in per_route))))
    return rows


def train(env, agent, args, out):
    rng=np.random.default_rng(args.seed)
    routes=[]
    stage_outcomes=collections.defaultdict(collections.Counter)
    def reset():
        env.curriculum_stage=args.stage
        if hasattr(env,'training_clear_cfg'):
            empty=rng.random()<args.empty_fraction
            env.cfg_path=env.training_clear_cfg if empty else env.training_traffic_cfg
            env.warmup_steps=5 if empty else 300
            env.curriculum_stage='none' if empty else args.stage
        if not routes: routes.extend(rng.permutation(ROUTES))
        route=str(routes.pop())
        if args.focus_routes and rng.random()<.5:
            route=f'route_{int(rng.choice(args.focus_routes)):02}'
        return env.reset(seed=int(rng.integers(110000,120000)),options={'route':route})[0]
    obs=reset()
    write(out/'runtime.json',dict(sumo_version=traci.getVersion(),backend=traci.__name__))
    started=time.monotonic()
    statsfile=(out/'train.jsonl').open('a',encoding='utf8')
    episodefile=(out/'episodes.jsonl').open('a',encoding='utf8')
    try:
        while agent.num_timesteps<args.steps:
            agent.buffer.clear()
            for _ in range(min(agent.n_steps,args.steps-agent.num_timesteps)):
                action,lp,value=agent.act(obs)
                old_edge=traci.vehicle.getRoadID(env.ego_id)
                nxt,reward,done,cut,info=env.step(action)
                # A deadline is a scored failure in this finite-horizon task.
                # Stop GAE at either ending; never propagate across resets.
                agent.buffer.add(obs,action,lp,reward,value,done or cut)
                agent.num_timesteps+=1
                if done or cut:
                    stage_outcomes[env.curriculum_stage][info['outcome']]+=1
                    episodefile.write(json.dumps(dict(route=env.episode_route, seed=env.sumo_seed,
                        stage=env.curriculum_stage, outcome=info['outcome'], edge=old_edge, steps=env.step_count))+'\n')
                    episodefile.flush()
                obs=reset() if done or cut else nxt
            adv,ret=agent.buffer.compute_gae(agent.get_value(obs),agent.gamma,agent.gae_lambda)
            stats=agent.update(adv,ret)
            row=dict(step=agent.num_timesteps,elapsed=round(time.monotonic()-started,1),
                outcomes=dict(collections.Counter(env.episode_outcomes[-100:])),
                stage_outcomes={k:dict(v) for k,v in stage_outcomes.items()},**stats)
            statsfile.write(json.dumps(row)+'\n'); statsfile.flush()
            print(json.dumps(row),flush=True)
            agent.save(str(out/'model.pt'))
            if agent.num_timesteps%16384==0:
                agent.save(str(out/f'model-{agent.num_timesteps}.pt'))
            if (out/'STOP').exists():
                break
    finally:
        statsfile.close()
        episodefile.close()
        agent.save(str(out/'model.pt'))
        write(out/'training_summary.json',dict(actual_steps=agent.num_timesteps,
            elapsed_seconds=round(time.monotonic()-started,1),
            stop_reason='completed' if agent.num_timesteps>=args.steps else
                'stop_requested' if (out/'STOP').exists() else 'interrupted_or_error',
            episode_outcomes=dict(collections.Counter(env.episode_outcomes)),
            stage_outcomes={k:dict(v) for k,v in stage_outcomes.items()}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['original','no_gate','road_only'],default='road_only')
    p.add_argument('--stage',default='high_real_ratio',choices=['none','low','high','real_ratio','high_real_ratio'])
    p.add_argument('--resume',default='results/mcw_runs/stage3_high_v5_b/best_model.pt')
    p.add_argument('--out',required=True)
    p.add_argument('--train',action='store_true')
    p.add_argument('--steps',type=int,default=65536)
    p.add_argument('--seed',type=int,default=23)
    p.add_argument('--eval-seed',type=int,default=50001)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--entropy',type=float,default=.01)
    p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--value-coef',type=float,default=.5)
    p.add_argument('--critic-lr-mult',type=float,default=1.)
    p.add_argument('--side-risk-penalty',type=float,default=.3)
    p.add_argument('--acceleration-risk-penalty',type=float,default=0.)
    p.add_argument('--future-goal-coef',type=float,default=0.)
    p.add_argument('--invalid-command-penalty',type=float,default=.01)
    p.add_argument('--empty-fraction',type=float,default=0.,help='Training-only fraction of empty-road episodes; evaluation always uses --stage')
    p.add_argument('--detach-value',action='store_true',help='Fit the value head without sending critic gradients through the actor features')
    p.add_argument('--focus-routes',type=int,nargs='+',choices=range(1,17),default=[])
    p.add_argument('--adapt-risk',action='store_true',help='Train only traffic input columns and the value head; preserve the road policy when traffic inputs are zero')
    p.add_argument('--reset-std',action='store_true')
    p.add_argument('--stochastic',action='store_true',help='Evaluate the learned sampled PPO policy, with a recorded per-episode RNG seed')
    p.add_argument('--routes',type=int,nargs='+',choices=range(1,17),default=list(range(1,17)))
    p.add_argument('--trace',action='store_true')
    p.add_argument('--split',choices=['development','heldout'],default='development')
    args=p.parse_args()
    if args.episodes < 1 or args.steps < 1:
        p.error('episodes and steps must be positive')
    if args.train and args.mode != 'road_only':
        p.error('This runner trains road_only; original/no_gate are evaluation controls')
    if args.train and args.split == 'heldout':
        p.error('A heldout run must be an evaluation, never training')
    if not 0<=args.empty_fraction<1 or (args.empty_fraction and (not args.train or args.stage=='none')):
        p.error('empty-fraction must be in [0,1) and is only for traffic training')
    torch.set_num_threads(1)
    out=Path(args.out)
    if (out/'manifest.json').exists():
        p.error('Output already contains a run; use a new --out to preserve its evidence')
    out.mkdir(parents=True,exist_ok=True)
    env,agent,protocol=make(args)
    if not args.train:
        agent.save(str(out/'evaluated_model.pt'))
        protocol['evaluated_model_sha256']=hashlib.sha256((out/'evaluated_model.pt').read_bytes()).hexdigest()
    write(out/'manifest.json',protocol)
    for name in protocol['code_hashes']:
        snapshot=out/'source'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    try:
        if args.train: train(env,agent,args,out)
        else: evaluate(env,agent,args,out)
    finally: env.close()


if __name__=='__main__': main()
