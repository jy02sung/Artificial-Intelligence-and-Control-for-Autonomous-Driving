import unittest
import numpy as np
import torch
from algorithms.ppo import PPO
from utils.curriculum_ppo import KeepPPO, distribution_kl

class CurriculumTests(unittest.TestCase):
    def test_sample_log_probability_matches_update(self):
        a=PPO(63,2,hidden_sizes=(16,16),device='cpu')
        obs=torch.randn(128,63)
        action,lp,_=a.policy.sample(obs)
        replay,_,_=a.policy.evaluate_actions(obs,action)
        torch.testing.assert_close(lp,replay)
        self.assertTrue(set(action[:,1].tolist()) <= {-1.,0.,1.})
    def test_keep_kl_zero_for_identical_policy(self):
        a=PPO(63,2,hidden_sizes=(16,16),device='cpu')
        x=torch.randn(12,63)
        self.assertAlmostEqual(distribution_kl(a.policy(x),a.policy(x)).item(),0.,places=6)
    def test_kl_guard_and_separate_supervision(self):
        torch.set_num_threads(1)
        a=KeepPPO(63,2,hidden_sizes=(16,16),device='cpu',n_epochs=10,minibatch_size=8)
        data=dict(obs=np.zeros((16,63),dtype=np.float32),actions=np.zeros((16,2),dtype=np.float32),category=np.array(['cruise']*16))
        a.set_supervision(a.policy,data)
        a.target_kl=1e-12
        for i in range(16):
            obs=np.full(63,i/16,dtype=np.float32);action,lp,val=a.act(obs)
            a.buffer.add(obs,action,lp,1.,val,i==15)
        n=len(a.buffer)
        result=a.update(np.linspace(-1,1,16,dtype=np.float32),np.linspace(0,2,16,dtype=np.float32))
        self.assertTrue(result['early_stop'])
        self.assertLess(result['optimizer_steps'],20)
        self.assertEqual(len(a.buffer),n)

    def test_separate_learning_rates_preserve_loaded_outputs(self):
        a=KeepPPO(63,2,hidden_sizes=(16,16),device='cpu')
        self.assertEqual([g['lr'] for g in a.optimizer.param_groups],[3e-5,3e-4])
        self.assertFalse(set(map(id,a.optimizer.param_groups[0]['params'])) & set(map(id,a.optimizer.param_groups[1]['params'])))

if __name__=='__main__':unittest.main()
