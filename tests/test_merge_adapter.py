import unittest
import numpy as np
import torch
from utils.merge_adapter_ppo import MergeAdapterPPO

class MergeAdapterTests(unittest.TestCase):
    def test_only_new_input_connections_change_actor(self):
        torch.set_num_threads(1)
        a=MergeAdapterPPO(63,2,hidden_sizes=(16,16),device='cpu',n_epochs=2,minibatch_size=16)
        a.configure_adapter()
        obs=torch.randn(32,63);obs[:,60:63]=0
        before=[x.detach().clone() for x in a.policy(obs)[:3]]
        saved={k:v.clone() for k,v in a.policy.state_dict().items()}
        data=dict(obs=np.ones((32,63),dtype=np.float32)*.3,actions=np.tile([-.8,0.],(32,1)).astype(np.float32),category=np.array(['yield']*32))
        a.set_supervision(a.policy,data,data);a.safe_weight=.1
        for i in range(32):
            o=data['obs'][i];act,lp,val=a.act(o);a.buffer.add(o,act,lp,1.,val,i==31)
        a.update(np.linspace(-1,1,32,dtype=np.float32),np.linspace(0,2,32,dtype=np.float32))
        for old,new in zip(before,a.policy(obs)[:3]):self.assertTrue(torch.equal(old,new))
        for key,v in a.policy.state_dict().items():
            if key.startswith('v_head.'):continue
            if key=='trunk.0.weight':self.assertTrue(torch.equal(v[:,:60],saved[key][:,:60]))
            else:self.assertTrue(torch.equal(v,saved[key]),key)
        self.assertGreater(a.policy.trunk[0].weight[:,60:63].abs().sum().item(),0)
    def test_initial_new_features_do_not_change_outputs(self):
        a=MergeAdapterPPO(63,2,hidden_sizes=(16,16),device='cpu');a.configure_adapter()
        x=torch.randn(10,63);y=x.clone();y[:,60:63]=0
        for p,q in zip(a.policy(x)[:3],a.policy(y)[:3]):self.assertTrue(torch.equal(p,q))

if __name__=='__main__':unittest.main()
