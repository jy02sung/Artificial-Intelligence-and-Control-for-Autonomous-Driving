import unittest
import numpy as np
from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer
from utils.timeout_rollout import TimeoutBoundaryMixin

class FixedPPO(TimeoutBoundaryMixin, PPO):
    pass

class ToyEnv:
    def __init__(self, timeout=True):
        self.n = 0
        self.timeout = timeout
    def step(self, action):
        self.n += 1
        if self.n == 1:
            return np.array([7.]), 1., not self.timeout, self.timeout, {}
        return np.array([0.]), 2., True, False, {}
    def reset(self):
        return np.array([100.]), {}

def run(cls, steps=2, timeout=True):
    a = cls.__new__(cls)
    a.n_steps=steps; a.num_timesteps=0; a.gamma=.99; a.buffer=RolloutBuffer()
    a.act=lambda obs:(np.zeros(2),0.,float(obs[0]))
    a.get_value=lambda obs:float(obs[0])
    _, last, _ = a.collect_rollout(ToyEnv(timeout),np.array([3.]),ep_stats=[0.,0.])
    return a.buffer.compute_gae(last,.99,.95)[1]

class TimeoutBoundaryTests(unittest.TestCase):
    def test_timeout_does_not_include_next_episode(self):
        np.testing.assert_allclose(run(FixedPPO), [7.93,2.],rtol=1e-6)
        self.assertGreater(abs(float(run(PPO)[0])-7.93),1.)
    def test_timeout_at_rollout_end_bootstraps_once(self):
        np.testing.assert_allclose(run(FixedPPO,1), [7.93],rtol=1e-6)
    def test_true_terminal_unchanged(self):
        np.testing.assert_allclose(run(FixedPPO,timeout=False),run(PPO,timeout=False))

if __name__=='__main__':unittest.main()
