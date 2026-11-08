"""Observation-only ablation. Parent actions, rewards and road stay unchanged."""
import numpy as np
from gymnasium import spaces
from env.navigation_observation import NavigationEnv
from env.sumo_env import traci


def validity_features(nav):
    return [1. if nav['current_targets'] else -1.,
            1. if nav['prepare_targets'] else -1.]


class ClarityEnv(NavigationEnv):
    def __init__(self, *args, clarity='baseline', **kwargs):
        assert clarity in ('baseline', 'validity', 'lookahead')
        self.clarity = clarity
        super().__init__(*args, observation_mode='N1', **kwargs)
        # Keep parent obs_dim=37: its terminal and IC append logic uses this value.
        # Only the exposed policy space is extended.
        self.observation_space = spaces.Box(-1, 1, (47,), np.float32)

    def _get_obs(self):
        original = super()._get_obs()
        extra = np.zeros(10, dtype=np.float32)
        if self.clarity != 'baseline' and self.ego_id in traci.vehicle.getIDList():
            extra[:2] = validity_features(self.navigation_context())
        result = np.concatenate((original, extra)).astype(np.float32)
        assert result.shape == (47,) and np.isfinite(result).all()
        return result
