"""Integration checks for the experiment's substantive no-oracle/no-shield contract."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from utils.mcw_policy_only import make, traci
from utils.mcw_train import McwEnv


class PolicyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = SimpleNamespace(stage='high_real_ratio', mode='road_only', seed=23, train=False,
            lr=1e-4, entropy=.01, reset_std=False,
            side_risk_penalty=.3, acceleration_risk_penalty=.1,
            future_goal_coef=.005, invalid_command_penalty=.2,
            resume='results/mcw_runs/stage3_high_v5_b/best_model.pt')
        cls.env, cls.agent, cls.protocol = make(args)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_lane_answers_absent_and_navigation_not_called(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('Lane-answer oracle was accessed')
        self.env.navigation.query = forbidden
        self.env.context = forbidden
        self.env._route_lane_ready = forbidden
        obs, _ = self.env.reset(seed=40005, options={'route':'route_05'})
        for _ in range(15):
            np.testing.assert_array_equal(obs[31:34], 0.)
            np.testing.assert_array_equal(obs[35:47], 0.)
            obs, _, done, cut, info = self.env.step(np.array([-.1, 0.], np.float32))
            if done or cut:
                break

    def test_cached_scan_matches_uncached_at_each_step_and_reset(self):
        for _ in range(2):
            self.env.reset(seed=40005, options={'route':'route_05'})
            for step in range(20):
                for side in (-1,1):
                    self.assertEqual(self.env._scan_lane_neighbors(side), McwEnv._scan_lane_neighbors(self.env,side))
                    for front in (False,True):
                        self.assertEqual(self.env._neighbor(side,front),self.env._neighbor_uncached(side,front))
                _,_,done,cut,_=self.env.step(np.array([-.1,0.],np.float32))
                if done or cut: break

    def test_dangerous_command_is_not_protected(self):
        # This is a deliberately unsafe fixed command, not a learned controller.
        self.env.reset(seed=40004, options={'route':'route_04'})
        result = None
        for _ in range(300):
            _, _, done, cut, info = self.env.step(np.array([1., 1.], np.float32))
            self.assertEqual(info['accel_raw'], 1.)
            self.assertEqual(info['lane_change_raw'], 1.)
            self.assertFalse(self.env.lc_safety_enabled)
            if done or cut:
                result = info
                break
        self.assertIsNotNone(result, 'Probe must reach an observable ending')
        self.assertTrue(result['collided'], 'Environment must allow this known unsafe collision')

    def test_risk_adaptation_preserves_actions_without_traffic(self):
        args=SimpleNamespace(stage='high_real_ratio',mode='road_only',seed=23,train=False,
            lr=.001,entropy=.002,reset_std=False,adapt_risk=True,
            resume='results/mcw_runs/stage3_high_v5_b/best_model.pt')
        env,agent,protocol=make(args)
        columns=protocol['risk_adaptation_columns']
        obs=torch.randn(4,env.policy_dim)
        obs[:,columns]=0.
        with torch.no_grad():
            before=agent.policy.det_action(obs).clone()
            weights=agent.policy.trunk[0].weight.clone()
        crowded=obs.clone();crowded[:,columns]=1.
        mu,_,logits,value=agent.policy(crowded)
        agent.optimizer.zero_grad()
        (mu.sum()+logits[:,0].sum()+value.sum()).backward()
        agent.optimizer.step()
        with torch.no_grad():
            torch.testing.assert_close(before,agent.policy.det_action(obs),atol=0.,rtol=0.)
            other=[i for i in range(env.policy_dim) if i not in columns]
            torch.testing.assert_close(weights[:,other],agent.policy.trunk[0].weight[:,other],atol=0.,rtol=0.)
        env.close()


if __name__ == '__main__':
    unittest.main()
