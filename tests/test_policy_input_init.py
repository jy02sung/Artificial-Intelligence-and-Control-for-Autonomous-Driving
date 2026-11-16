import copy
import unittest
import torch
from utils.networks import HybridActorCritic
from utils.policy_input_init import preserve_existing_inputs


class InputInitializationTest(unittest.TestCase):
    def test_preserves_old_outputs_and_new_inputs_can_learn(self):
        torch.manual_seed(72)
        baseline = HybridActorCritic(63)
        policy = copy.deepcopy(baseline)
        before = copy.deepcopy(policy.state_dict())
        preserve_existing_inputs(policy, range(47, 50))
        obs = torch.randn(64, 63)
        old = obs.clone()
        old[:, 47:50] = 0
        for expected, actual in zip(baseline(old), policy(obs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in policy.state_dict().items():
            if key == 'trunk.0.weight':
                expected = before[key].clone()
                expected[:, 47:50] = 0
            else:
                expected = before[key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        sum(t.sum() for t in policy(obs)).backward()
        self.assertGreater(float(policy.trunk[0].weight.grad[:, 47:50].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
