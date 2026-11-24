"""Fresh PPO only: jys3 observations/rewards, no teacher, no safety overrides.

python -m utils.jys3_scratch audit
python -m utils.jys3_scratch train --name explore201 --seed 201 --steps 262144
python -m utils.jys3_scratch evaluate --run explore201 --step 262144
"""
import argparse
from collections import Counter, deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime'))
os.environ['IC_LIBSUMO'] = '1'
from utils.jys3 import environment, HP, OBS_DIM, SCHEMA, atomic_torch, m, torch
from env.sumo_env import traci, SumoHighwayEnv
from env.navigation_observation import RouteNavigation
import numpy as np

OUT = ROOT / 'results' / 'jys3_scratch'
BASE_COMMIT = 'b5a7241925cc88757cac8592c98fdd5532830437'
DEV_BASE = 710000000
HOLD_BASE = 810000000


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def append(path, obj):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(policy):
    policy = getattr(policy, 'policy', policy)
    digest = hashlib.sha256()
    for key, tensor in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden pretrained load, lane oracle or safety gate invoked')


@contextmanager
def no_weights():
    with patch.object(torch, 'load', forbidden), patch.object(torch.jit, 'load', forbidden):
        yield


def make_agent(seed, preset):
    hp = dict(HP)
    if preset == 'explore':
        hp.update(lr=.0002, gamma=.999, gae_lambda=.98, n_epochs=4,
                  minibatch_size=256, clip_eps=.15, value_coef=.1,
                  lane_entropy_coef=.03, lane_keep_bias=0.)
    with no_weights():
        agent = m.PPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=seed, **hp)
    assert not agent.optimizer.state
    return agent, hp


def make_env(folder, seed, stage='high_real_ratio'):
    env = environment(stage, seed, folder)
    assert not env.lc_safety_enabled and not env.variants
    assert env.observation_mode == 'N1' and env.clarity == 'baseline'
    assert env.visibility == 120 and env.obs_noise and env.occlusion
    # The physical link map and road-direction signs remain; target-lane queries cannot run.
    env.navigation.query = forbidden
    env.navigation_context = forbidden
    env._route_lane_ready = forbidden
    env._lane_change_is_safe = forbidden
    original_reset = env.reset
    def reset(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        traci.vehicle.setSpeedMode(env.ego_id, 96)
        traci.vehicle.setLaneChangeMode(env.ego_id, 0)
        assert traci.vehicle.getSpeedMode(env.ego_id) == 96
        assert traci.vehicle.getLaneChangeMode(env.ego_id) == 0
        assert obs.shape == (50,) and np.isfinite(obs).all()
        return obs, info
    env.reset = reset
    return env


def outcome(info, cut):
    # Collision takes precedence even if SUMO reports arrival in the same step.
    return ('collision' if info['collided'] else 'arrival' if info['arrived']
            else 'stall' if info.get('stalled') else 'timeout' if cut else 'other')


class Episodes:
    def __init__(self, env, seed, path):
        self.env, self.seed, self.path = env, seed, path
        self.n = 0
        self.counts = Counter()
        self.recent = deque(maxlen=100)

    def reset(self):
        block, index = divmod(self.n, 16)
        routes = list(m.ROUTES)
        random.Random(self.seed * 100000 + block).shuffle(routes)
        route = routes[index]
        # Independent of route compatibility: alternate first two existing starting lanes.
        lane = min(block % 2, self.env.counts[self.env.routes[route][0]] - 1)
        self.env.ego_depart_lane = str(lane)
        seed = self.seed * 1000000 + self.n
        self.record = dict(episode=self.n, route=route, seed=seed, lane=lane)
        self.n += 1
        self.ret, self.length = 0., 0
        return self.env.reset(seed=seed, options={'route': route})

    def step(self, action):
        obs, reward, done, cut, info = self.env.step(action)
        self.ret += reward
        self.length += 1
        assert np.isclose(info['accel_raw'], np.clip(action[0], -1, 1))
        assert np.isclose(info['lane_change_raw'], np.clip(action[1], -1, 1))
        if done or cut:
            result = outcome(info, cut)
            self.counts[result] += 1
            self.recent.append(result)
            append(self.path, dict(self.record, outcome=result, steps=self.length, reward=self.ret))
        return obs, reward, done, cut, info


def checkpoint(agent, folder, step, manifest):
    path = folder / f'policy_{step}.pt'
    if path.exists():
        raise FileExistsError(path)
    atomic_torch(dict(schema=SCHEMA, policy=agent.policy.state_dict(),
                     optimizer=agent.optimizer.state_dict(), torch_rng=torch.get_rng_state()), path)
    save(path.with_suffix('.json'), dict(schema='jys3-scratch-v1', step=step,
         checkpoint_sha256=sha(path), manifest_sha256=sha(folder / 'manifest.json'),
         initial_fingerprint=manifest['initial_fingerprint'], fingerprint=fingerprint(agent)))


def train(args):
    torch.set_num_threads(1)
    folder = OUT / args.name
    folder.mkdir(parents=True, exist_ok=False)
    agent, hp = make_agent(args.seed, args.preset)
    sources = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=ROOT, text=True).splitlines()
    manifest = dict(schema='jys3-scratch-v1', base_commit=BASE_COMMIT,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        seed=args.seed, preset=args.preset, hp=hp, initial='random', pretrained=False,
        imitation=False, teacher_actions=False, safety_gate=False, lane_oracle=False,
        speed_mode=96, lane_change_mode=0, observation_dim=50, steps=args.steps,
        initial_fingerprint=fingerprint(agent), dev_base=DEV_BASE, hold_base=HOLD_BASE,
        source_hashes={p:sha(ROOT / p) for p in sources},
        stage='high_real_ratio', time_limit_steps=1000, eval='deterministic',
        python=sys.version, torch=torch.__version__, sumo=traci.getVersion())
    save(folder / 'manifest.json', manifest)
    checkpoint(agent, folder, 0, manifest)
    start = time.monotonic()
    with no_weights():
        env = make_env(folder / 'map', args.seed)
        save(folder / 'scenario_hashes.json', {p.name:sha(p) for p in (folder/'map'/'high_real_ratio').iterdir() if p.is_file()})
        schedule = Episodes(env, args.seed, folder / 'episodes.jsonl')
        try:
            obs, _ = schedule.reset()
            stats = [0., 0]
            while agent.num_timesteps < args.steps:
                count = min(2048, args.steps - agent.num_timesteps)
                obs, last, _ = agent.collect_rollout(schedule, obs, ep_stats=stats, steps=count, timeout_is_failure=True)
                adv, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
                metrics = agent.update(adv, ret)
                record = dict(step=agent.num_timesteps, episodes=schedule.n, outcomes=dict(schedule.counts),
                              recent=dict(Counter(schedule.recent)), seconds=time.monotonic()-start, **metrics)
                save(folder / 'progress.json', record)
                append(folder / 'updates.jsonl', record)
                if agent.num_timesteps % 8192 == 0:
                    print(json.dumps(record), flush=True)
                if agent.num_timesteps % 65536 == 0 or agent.num_timesteps == args.steps or (folder/'STOP').exists():
                    checkpoint(agent, folder, agent.num_timesteps, manifest)
                if (folder/'STOP').exists():
                    break
            save(folder / 'training_summary.json', dict(record, completed=agent.num_timesteps == args.steps))
        finally:
            env.close()


def evaluate(args):
    torch.set_num_threads(1)
    folder = OUT / args.run
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    path = folder / f'policy_{args.step}.pt'
    proof = json.loads(path.with_suffix('.json').read_text())
    assert manifest['schema'] == proof['schema'] == 'jys3-scratch-v1'
    assert manifest['initial'] == 'random' and not manifest['pretrained'] and not manifest['imitation']
    assert sha(path) == proof['checkpoint_sha256'] and sha(folder/'manifest.json') == proof['manifest_sha256']
    agent, _ = make_agent(manifest['seed'], manifest['preset'])
    assert fingerprint(agent) == manifest['initial_fingerprint']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    agent.policy.load_state_dict(payload['policy'], strict=True)
    assert fingerprint(agent) == proof['fingerprint']
    agent.policy.eval()
    base = HOLD_BASE if args.part == 'hold' else DEV_BASE
    repeats = 10 if args.part == 'hold' else args.repeats
    target = folder / f'{args.part}_{args.step}.json'
    if target.exists():
        raise FileExistsError(target)
    # Lock final selection before opening held-out seeds; do not tune against them.
    if args.part == 'hold':
        dev = json.loads((folder / f'dev_{args.step}.json').read_text(encoding='utf-8'))
        if dev['n'] < 32 or not dev['target_met']:
            raise RuntimeError('Final evaluation requires a passing development evaluation of at least 32 episodes')
        lock = OUT / 'heldout_selection.json'
        if lock.exists():
            raise RuntimeError('Held-out selection already used; register new independent seeds before another final test')
        save(lock, dict(run=args.run, step=args.step, checkpoint_sha256=sha(path), base=base, repeats=repeats))
    env = make_env(folder / f'eval_map_{args.part}_{args.step}', manifest['seed'])
    rows = []
    try:
        for rep in range(repeats):
            for route in range(1, 17):
                name = f'route_{route:02d}'
                lane = min(rep % 2, env.counts[env.routes[name][0]]-1)
                env.ego_depart_lane = str(lane)
                seed = base + rep * 100 + route
                obs, _ = env.reset(seed=seed, options={'route':name})
                history = deque(maxlen=15)
                actions, changes = Counter(), 0
                for step in range(1, 1001):
                    action = agent.predict(obs, deterministic=True)
                    snap = env.snapshot
                    history.append(dict(edge=snap[0], lane=int(snap[1]), speed=float(snap[7]), remaining=float(snap[-1]),
                                        action=action.tolist(), obs=obs.tolist()))
                    obs, _, done, cut, info = env.step(action)
                    actions[int(info['lane_change'])] += 1
                    changes += int(info['lane_after'] is not None and info['lane_before'] != info['lane_after'])
                    if done or cut:
                        break
                result = outcome(info, cut)
                row = dict(route=name, rep=rep, seed=seed, lane=lane, outcome=result, steps=step,
                           actions=dict(actions), lane_index_changes=changes)
                rows.append(row)
                if result != 'arrival':
                    append(target.with_suffix('.failures.jsonl'), dict(row, trace=list(history)))
                save(target.with_suffix('.progress.json'), dict(completed=len(rows), counts=dict(Counter(r['outcome'] for r in rows))))
                print(json.dumps(row), flush=True)
        counts = Counter(r['outcome'] for r in rows)
        save(target, dict(run=args.run, step=args.step, part=args.part, checkpoint_sha256=sha(path),
             n=len(rows), counts=dict(counts), completion_rate=counts['arrival']/len(rows),
             collision_rate=counts['collision']/len(rows),
             target_met=counts['arrival']/len(rows) >= .95 and counts['collision']/len(rows) < .05, rows=rows))
    finally:
        env.close()


def audit():
    torch.set_num_threads(1)
    folder = OUT/'audit'
    folder.mkdir(parents=True, exist_ok=True)
    with no_weights():
        a, _ = make_agent(201, 'explore')
        b, _ = make_agent(201, 'explore')
        c, _ = make_agent(202, 'explore')
        assert fingerprint(a) == fingerprint(b) != fingerprint(c)
        env = make_env(folder/'map', 201)
        checks = []
        try:
            for number in range(1,17):
                name = f'route_{number:02d}'
                obs, _ = env.reset(seed=610000000+number, options={'route':name})
                # Fixed dangerous acceleration must be able to produce a real collision.
                for step in range(1,1001):
                    action = np.array([1., 0.], dtype=np.float32)
                    obs, reward, done, cut, info = env.step(action)
                    assert info['accel_raw'] == 1. and info['lane_change_raw'] == 0.
                    assert obs.shape == (50,) and np.isfinite(obs).all()
                    expected = -15. if info['collided'] else 5. if info['arrived'] else -5. if cut else .25 if info['fork_passed'] else 0.
                    assert reward == expected
                    if done or cut:
                        break
                checks.append(dict(route=name, outcome=outcome(info,cut), steps=step))
            assert any(r['outcome']=='collision' for r in checks)
            # Physically available lateral command reaches SUMO even when sensor says unsafe.
            env.reset(seed=610000004, options={'route':'route_04'})
            with patch.object(traci.vehicle, 'changeLane', wraps=traci.vehicle.changeLane) as change:
                if traci.edge.getLaneNumber(traci.vehicle.getRoadID(env.ego_id)) > 1:
                    assert env._apply_lane_change(1)
                    assert change.call_count == 1
            save(folder/'report.json', dict(passed=True, random_init_reproduced=True,
                 old_weight_loading_blocked=True, oracle_and_gate_calls_blocked=True,
                 action_forwarding=True, observation_dim=50, speed_mode=96, lane_change_mode=0,
                 initial_fingerprint=fingerprint(a), dangerous_driving=checks, sumo=traci.getVersion()))
            print((folder/'report.json').read_text(), flush=True)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--name', required=True)
    train_parser.add_argument('--seed', type=int, default=201)
    train_parser.add_argument('--steps', type=int, default=262144)
    train_parser.add_argument('--preset', choices=['baseline','explore'], default='explore')
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('--run', required=True)
    eval_parser.add_argument('--step', type=int, required=True)
    eval_parser.add_argument('--part', choices=['dev','hold'], default='dev')
    eval_parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'audit':
        audit()
    elif args.command == 'train':
        if not 1 <= args.seed <= 2000 or args.steps < 2048 or args.steps % 2048:
            parser.error('seed must be 1..2000, steps must be a positive multiple of 2048')
        report = OUT/'audit'/'report.json'
        if not report.exists() or not json.loads(report.read_text())['passed']:
            parser.error('Run audit successfully before training')
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
