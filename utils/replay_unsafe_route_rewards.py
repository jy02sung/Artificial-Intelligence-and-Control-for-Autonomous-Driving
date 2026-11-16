"""Measure actual route bonuses at unsafe lane-change attempts in fixed replays."""
from utils import run_lane_followup as trial
s = trial.s
t = s.t


def main():
    s.a.b.torch.set_num_threads(1)
    directory = s.OUT / 'runs/future'
    manifest = t.read(directory / 'manifest.json')
    checkpoint = directory / 'policy-100352.pt'
    agent = s.a.b.PPO(obs_dim=63, act_dim=2, **manifest['hp'])
    agent.load(str(checkpoint))
    output = s.ROOT / 'results/unsafe_route_reward_replay_20261116'
    for route, seed in [('route_13', 79102), ('route_11', 79103)]:
        env = s.make_env('future', evaluation=True, none=False)
        events = []
        v = s.a.b.traci.vehicle
        try:
            obs, _ = env.reset(seed=seed, options={'route': route})
            while True:
                action = agent.predict(obs, deterministic=True)
                risk = env.safety_context(action)['lane']
                pre = dict(t=env.step_count * .2, edge=v.getRoadID(env.ego_id), lane=v.getLaneIndex(env.ego_id),
                           speed=v.getSpeed(env.ego_id), risk=risk, action=action.tolist())
                obs, reward, done, cut, info = env.step(action)
                parts = info['route_reward_parts']
                selection = parts.get('selection', 0) * env.adaptive_config['route_scale']
                maintenance = parts.get('maintenance', 0) * env.adaptive_config['route_scale']
                if selection > 1e-8 or (risk >= .2 and maintenance > 1e-8):
                    events.append(dict(pre, selection=selection, maintenance=maintenance,
                                       lane_penalty=parts.get('lane', 0) * env.adaptive_config['lane_scale'],
                                       total_reward=reward, lane_applied=info.get('lane_change_applied')))
                if done or cut:
                    result = dict(route=route, seed=seed, checkpoint_sha=s.a.b.digest(checkpoint),
                                  outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit',
                                  seconds=env.step_count * .2, route_overrides=env.route_override_count, events=events)
                    t.save(output / f'{route}-{seed}.json', result)
                    assert env.route_override_count == 0
                    print(result, flush=True)
                    break
        finally:
            env.close()


if __name__ == '__main__':
    main()
