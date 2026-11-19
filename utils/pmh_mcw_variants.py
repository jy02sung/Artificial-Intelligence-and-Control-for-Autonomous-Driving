"""mcw2 model with the 7th-meeting changes (pmh), on top of utils/mcw_train.py.

utils/mcw_train.py (the team's final mcw2 trainer) is used unchanged; this
wrapper only swaps make_env() and adds options. Everything not listed here
(47-dim observation layout, hybrid PPO, lr 1e-4 constant, rewards incl. the
lateral TTC/gap penalty, lane-change penalty 0.01, stall/timeout penalties,
no safety gate) stays exactly as in mcw2.

Meeting decisions (docs: 7th meeting answers) and how they map here:

  1. detection range 150 m        --visibility 150   (OBSERVATION["visibility"],
                                   the normaliser of neighbour gaps, lane
                                   connectivity and density; mcw2 uses 50 m)
  2. Gaussian sensor noise         --noise fixed      sigma_d = 1.0 m, sigma_v = 0.5 m/s
                                   --noise distance   sigma = sigma0 + k * gap:
                                                      sigma_d = 0.5 + 0.02 d  (3.5 m at 150 m)
                                                      sigma_v = 0.25 + 0.005 d (1.0 m/s at 150 m)
     Applied to the observed neighbour gap and speed of every detected vehicle
     (obs[1:21]) in training and evaluation. Rewards use the true values.
  3. SUMO navigation               --sumo-nav: the route-guidance slots come
                                   from SUMO's own getBestLanes() instead of
                                   the team RouteNavigation:
                                     obs[31] = bestLaneOffset / 3
                                     obs[32] = distance the current lane can
                                               still be followed / horizon
                                     obs[33] = obs[31]
                                   and the team-navigation lookahead slots
                                   obs[37:47] are zero.
  4. no safety gate                unchanged (--no-safety); collision reduction
                                   comes from rewards/observation only:
                                   ① TTC + target-lane gap penalty and ③ lane
                                   change penalty already exist in mcw2,
                                   ② adjacent-lane front/rear gap & relative
                                   speed are obs[1:21] (mcw2 fixed their
                                   disappearance), ④ --weave-weight N repeats
                                   the routes through the 51 m weave
                                   170059994 (route_03/04/05/13/14/15) N times
                                   in the training route schedule.

Usage: python -m utils.pmh_mcw_variants [--visibility 150] [--noise fixed|distance]
       [--sumo-nav] [--weave-weight 2] <utils.mcw_train arguments>
"""
import argparse
import sys

import numpy as np

from utils import mcw_train as m

traci = m.traci
WEAVE_ROUTES = ('route_03', 'route_04', 'route_05', 'route_13', 'route_14', 'route_15')
NOISE = {
    'none': None,
    'fixed': lambda d: (1.0, 0.5),
    'distance': lambda d: (0.5 + 0.02 * d, 0.25 + 0.005 * d),
}


class VariantEnv(m.McwEnv):
    def __init__(self, *args, noise='none', sumo_nav=False, weave_weight=1, **kwargs):
        self.noise = noise
        self.sumo_nav = sumo_nav
        self.weave_weight = int(weave_weight)
        self.noise_rng = np.random.default_rng(0)
        self.weighted_schedule = []
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if 'route' not in options and self.weave_weight > 1:
            if not self.weighted_schedule:
                pool = [r for r in self.routes if r.startswith('route_')]
                pool += [r for r in pool if r in WEAVE_ROUTES] * (self.weave_weight - 1)
                self.weighted_schedule = list(self.route_rng.permutation(pool))
            options['route'] = str(self.weighted_schedule.pop())
        self.noise_rng = np.random.default_rng(None if seed is None else seed + 7919)
        return super().reset(seed=seed, options=options)

    def _get_obs(self):
        obs = super()._get_obs()
        if self.ego_id not in traci.vehicle.getIDList():
            return obs
        if NOISE[self.noise] is not None:
            obs = obs.copy()
            W = self.visibility
            for k in range(5):
                i = 1 + 4 * k
                for d_i, sign, empty in ((i, 1.0, 1.0), (i + 2, -1.0, -1.0)):
                    if obs[d_i] == empty and obs[d_i + 1] == empty:
                        continue                       # no vehicle in that slot
                    gap = abs(float(obs[d_i])) * W
                    s_d, s_v = NOISE[self.noise](gap)
                    gap = max(gap + self.noise_rng.normal(0.0, s_d), 0.0)
                    obs[d_i] = sign * min(gap / W, 1.0)
                    obs[d_i + 1] = float(np.clip(obs[d_i + 1] + self.noise_rng.normal(0.0, s_v) / self.vmax,
                                                 -1.0, 1.0))
        if self.sumo_nav:
            obs = obs.copy()
            v = traci.vehicle
            lane = v.getLaneID(self.ego_id)
            if not lane.startswith(':'):
                pos = v.getLanePosition(self.ego_id)
                for data in v.getBestLanes(self.ego_id):
                    if data[0] == lane:
                        offset, length = data[3], data[1]
                        obs[31] = float(np.clip(offset / 3.0, -1.0, 1.0))
                        obs[32] = float(np.clip((length - pos) / self.prep_horizon, 0.0, 1.0))
                        obs[33] = obs[31]
                        break
            obs[37:47] = 0.0
        return obs.astype(np.float32)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--visibility', type=float, default=50.0)
    p.add_argument('--noise', choices=sorted(NOISE), default='none')
    p.add_argument('--sumo-nav', action='store_true')
    p.add_argument('--weave-weight', type=int, default=1)
    extra, rest = p.parse_known_args()
    observation = dict(m.OBSERVATION, visibility=extra.visibility)

    def make_env(cfg_path, road, nav, args, action_cfg, reward_cfg, sim_cfg, ego_cfg, traffic_cfg):
        return VariantEnv(
            str(cfg_path), road, ego_cfg, sim_cfg, observation, action_cfg, reward_cfg,
            traffic_cfg, (), args.seed,
            navigation=nav, clarity='lookahead',
            route_conn=args.route_conn, ttc_reward=args.ttc_reward,
            prep_reward=args.prep_reward, prep_horizon=args.nav_horizon,
            timeout_penalty=args.timeout_penalty, stall_penalty=args.stall_penalty,
            noise=extra.noise, sumo_nav=extra.sumo_nav, weave_weight=extra.weave_weight)

    m.make_env = make_env
    # mcw_train writes its scenario into one shared cache folder; parallel runs
    # would rewrite the route file while others read it. Use one folder per run.
    name = rest[rest.index('--name') + 1] if '--name' in rest else 'default'
    original_prepare = m.prepare_scenario
    m.prepare_scenario = lambda stage, cache_dir: original_prepare(
        stage, m.ROOT / 'results' / 'mcw_scenarios_pmh' / name)
    sys.argv = [sys.argv[0]] + rest
    print(f'[pmh_mcw_variants] visibility={extra.visibility} noise={extra.noise} '
          f'sumo_nav={extra.sumo_nav} weave_weight={extra.weave_weight}', flush=True)
    m.main()
    if '--name' in rest:
        name = rest[rest.index('--name') + 1]
        import json
        path = m.ROOT / 'results' / 'mcw_runs' / name / 'manifest.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            data['pmh_variant'] = dict(vars(extra), weave_routes=WEAVE_ROUTES,
                                       noise_sigma='fixed: d 1.0 m, v 0.5 m/s; distance: d 0.5+0.02d, v 0.25+0.005d')
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
