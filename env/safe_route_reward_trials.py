"""Reward eligibility and lateral-risk observation ablations; actions are untouched."""
import numpy as np
from utils.run_traffic_recovery_auto import AdaptiveEnv


def unsafe_route_withholding(parts, route_scale, risk):
    if risk < .2:
        return 0.
    return route_scale * sum(max(parts.get(k, 0), 0) for k in ('selection', 'maintenance'))


class SafeRouteRewardEnv(AdaptiveEnv):
    def __init__(self, *args, condition=None, **kwargs):
        if condition is None:
            args, condition = args[:-1], args[-1]
        assert condition in ('safe_reward', 'lateral_zero', 'combined')
        self.safe_route_condition = condition
        super().__init__(*args, condition='route', **kwargs)

    def _get_obs(self):
        obs = super()._get_obs()
        if self.safe_route_condition in ('lateral_zero', 'combined'):
            contexts = [self.safety_context(np.array([0., d], dtype=np.float32)) for d in (1, 0, -1)]
            obs[57:60] = [contexts[0]['lane'], contexts[1]['gap'], contexts[2]['lane']]
        return obs

    def step(self, action):
        risk = self.safety_context(action)['lane']
        obs, reward, done, cut, info = super().step(action)
        withheld = unsafe_route_withholding(info['route_reward_parts'], self.adaptive_config['route_scale'], risk) if not done and self.safe_route_condition in ('safe_reward', 'combined') else 0.
        self.reward_counts['unsafe_route_withheld_total'] -= withheld
        self.reward_counts['unsafe_route_withheld_events'] += int(withheld > 1e-8)
        self.reward_counts['bonus_total'] -= withheld
        info['route_selection_reward'] -= withheld
        info['route_reward_parts']['unsafe_route_withheld'] = -withheld
        return obs, reward - withheld, done, cut, info
