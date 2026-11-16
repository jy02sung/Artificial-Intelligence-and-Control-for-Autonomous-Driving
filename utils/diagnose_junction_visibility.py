"""Read-only visibility/braking-distance replay; no changed policy or vehicle action."""
import math
from utils import run_traffic_recovery_validation as v
from env.junction_conflict_trials import conflict_risk
s = v.s
t = s.t


def main():
    s.a.b.torch.set_num_threads(1)
    manifest = t.read(v.MODEL.parent / 'manifest.json')
    agent = s.a.b.PPO(obs_dim=63, act_dim=2, **manifest['hp'])
    agent.load(str(v.MODEL))
    output = s.ROOT / 'results/junction_visibility_diagnosis_20261116'
    for seed in (69101, 69103):
        env = s.make_env('route', evaluation=True, none=False)
        vehicle = s.a.b.traci.vehicle
        first_seen = {}
        first_risk = {}
        try:
            obs, _ = env.reset(seed=seed, options={'route': 'route_01'})
            while True:
                ego = env.ego_id
                speed = vehicle.getSpeed(ego)
                pos = vehicle.getPosition(ego)
                for foe in vehicle.getJunctionFoes(ego, 150):
                    vid, ed, fd, ex, fx = foe[:5]
                    if vid != 'flow_route_07.224' or ex < -vehicle.getLength(ego) or fx < -vehicle.getLength(vid):
                        continue
                    q = vehicle.getPosition(vid)
                    physical = math.hypot(q[0] - pos[0], q[1] - pos[1])
                    risk = conflict_risk(ed, fd, speed, vehicle.getSpeed(vid))[0]
                    state = dict(t=env.step_count * env.step_length, speed=speed, max_decel=env.max_decel,
                                 ego_distance=ed, foe_distance=fd, physical_distance=physical, risk=risk,
                                 ideal_stopping_distance=speed * speed / (2 * env.max_decel),
                                 one_step_reaction_distance=speed * env.step_length)
                    for horizon in (50, 100, 150):
                        if ed <= horizon and physical <= env.visibility:
                            first_seen.setdefault(horizon, state)
                            if risk > 0:
                                first_risk.setdefault(horizon, state)
                action = agent.predict(obs, deterministic=True)
                obs, _, done, cut, info = env.step(action)
                if done or cut:
                    result = dict(seed=seed, route='route_01', seconds=env.step_count * env.step_length,
                                  outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout',
                                  physical_visibility=env.visibility, first_seen=first_seen, first_nonzero_risk=first_risk, route_overrides=env.route_override_count)
                    assert env.route_override_count == 0
                    t.save(output / f'{seed}.json', result)
                    print(result, flush=True)
                    break
        finally:
            env.close()


if __name__ == '__main__':
    main()
