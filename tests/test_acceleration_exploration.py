import unittest
import torch
from utils.networks import HybridActorCritic
from utils.acceleration_exploration import reset_acceleration_std

class AccelerationExplorationTest(unittest.TestCase):
    def test_changes_only_std_not_deterministic_outputs(self):
        torch.manual_seed(11)
        policy=HybridActorCritic(63,hidden_sizes=(8,8),activation='relu',init_log_std=-1.3)
        obs=torch.randn(12,63)
        before=[x.detach().clone() for x in policy(obs)]
        weights={k:v.clone() for k,v in policy.state_dict().items()}
        reset_acceleration_std(policy,.6)
        after=policy(obs)
        for i in (0,2,3):self.assertTrue(torch.equal(before[i],after[i]))
        self.assertTrue(torch.allclose(after[1],torch.tensor([.6])))
        for key,value in weights.items():
            if key!='log_std':self.assertTrue(torch.equal(value,policy.state_dict()[key]))

    def test_rejects_invalid_std(self):
        p=HybridActorCritic(63,hidden_sizes=(8,8))
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):reset_acceleration_std(p,value)

if __name__=='__main__':unittest.main()
