"""Read-only replay of a screened candidate, with actual lane change and collision evidence."""
from collections import deque
from utils import run_safe_route_reward_trials as trial
s = trial.s
t = s.t


def main():
    s.a.b.torch.set_num_threads(1)
    checkpoint = s.OUT / 'runs/combined/policy-51200.pt'
    expected = t.read(s.OUT / 'early-evaluation/51200/combined/policy-51200-partial.json')
    expected = next(r for r in expected if r['route'] == 'route_02' and r['seed'] == 79102 and r['traffic'] == 'reference')
    manifest = t.read(checkpoint.parent / 'manifest.json')
    for file, digest in t.read(s.OUT / 'manifest.json')['source_hashes'].items():
        assert s.a.b.digest(s.ROOT / file) == digest, file
    digest = s.a.b.digest(checkpoint)
    agent = s.a.b.PPO(obs_dim=63, act_dim=2, **manifest['hp'])
    agent.load(str(checkpoint))
    env = s.make_env('combined', evaluation=True, none=False)
    tail = deque(maxlen=50)
    v = s.a.b.traci.vehicle
    try:
        obs, _ = env.reset(seed=79102, options={'route': 'route_02'})
        while True:
            action = agent.predict(obs, deterministic=True)
            row = dict(t=env.step_count * env.step_length, edge=v.getRoadID(env.ego_id),
                       lane=v.getLaneIndex(env.ego_id), speed=v.getSpeed(env.ego_id),
                       action=action.tolist(), risk=env.safety_context(action),
                       neighbors={str(offset): {str(front): env._neighbor(offset, front) for front in (True, False)} for offset in (-1, 0, 1)})
            obs, reward, done, cut, info = env.step(action)
            row.update(info={k:info.get(k) for k in ('lane_before','lane_after','lane_change','lane_change_applied','collided')}, reward=reward, parts=info['route_reward_parts'])
            tail.append(row)
            if done or cut:
                collisions = []
                for c in s.a.b.traci.simulation.getCollisions():
                    collisions.append({k:getattr(c,k,None) for k in ('collider','victim','colliderSpeed','victimSpeed','type','lane','pos')})
                outcome = 'collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
                seconds = env.step_count * env.step_length
                assert outcome == expected['outcome'] and abs(seconds - expected['seconds']) < 1e-6
                assert env.route_override_count == 0 and s.a.b.digest(checkpoint) == digest
                result = dict(route='route_02',seed=79102,outcome=outcome,seconds=seconds,checkpoint_sha=digest,
                              exact_outcome_reproduced=True,route_overrides=0,collisions=collisions,tail=list(tail))
                t.save(s.OUT / 'route02-79102-collision-replay.json', result)
                print({k:v for k,v in result.items() if k!='tail'}, flush=True)
                break
    finally:
        env.close()


if __name__ == '__main__':
    main()
