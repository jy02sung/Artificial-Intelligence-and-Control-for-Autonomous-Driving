import unittest
import numpy as np
import torch
from utils.lane_followup_ppo import LaneFollowupPPO

class LaneFollowupTests(unittest.TestCase):
    def test_lane_supervision_is_finite_and_updates_policy(self):
        torch.set_num_threads(1)
        a=LaneFollowupPPO(63,2,hidden_sizes=(16,16),device='cpu',n_epochs=1,minibatch_size=16)
        a.configure_adapter()
        data=dict(obs=np.full((32,63),.2,dtype=np.float32),actions=np.tile([-.8,1.],(32,1)).astype(np.float32),category=np.array(['safe_change']*32))
        a.set_supervision(a.policy,data,data);a.safe_weight=.1;a.lane_data=data
        old=a.policy.trunk[0].weight.detach().clone()
        for i,o in enumerate(data['obs']):
            act,lp,val=a.act(o);a.buffer.add(o,act,lp,1.,val,i==31)
        m=a.update(np.linspace(-1,1,32,dtype=np.float32),np.linspace(0,2,32,dtype=np.float32))
        self.assertGreater(m['lane_supervision_loss'],0)
        self.assertTrue(np.isfinite(m['lane_supervision_loss']))
        self.assertFalse(torch.equal(old,a.policy.trunk[0].weight))

if __name__=='__main__':unittest.main()
