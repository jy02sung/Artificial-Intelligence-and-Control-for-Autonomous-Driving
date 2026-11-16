import unittest
import numpy as np
import torch
from utils.merge_followup_ppo import MergeFollowupPPO

class FollowupTests(unittest.TestCase):
    def test_unfreezes_existing_actor_without_initial_change(self):
        a=MergeFollowupPPO(63,2,hidden_sizes=(16,16),device='cpu')
        x=torch.randn(8,63);x[:,60:]=0
        before=[v.detach().clone() for v in a.policy(x)[:3]]
        a.configure_adapter()
        for p in a.policy.parameters():self.assertTrue(p.requires_grad)
        for old,new in zip(before,a.policy(x)[:3]):self.assertTrue(torch.equal(old,new))
        a.optimizer.zero_grad();sum(v.sum() for v in a.policy(x)).backward();a.optimizer.step()
        self.assertFalse(torch.equal(before[0],a.policy(x)[0]))

if __name__=='__main__':unittest.main()
