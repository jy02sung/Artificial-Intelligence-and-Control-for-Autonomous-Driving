"""Regression using the live PPO collector and GAE buffer."""
import numpy as np
from types import SimpleNamespace
from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer

def run():
    results={}
    for steps in (1,2):
        for fixed in (False,True):
            values=[]
            for reset_v in (0.,10.):
                class Env:
                    count=0
                    def step(self, action):
                        self.count+=1
                        return 2., -15. if self.count==1 else 0., False,self.count==1,{}
                    def reset(self):
                        return 1.,{}
                v=lambda x:{0.:0.,1.:reset_v,2.:2.}[x]
                agent=SimpleNamespace(n_steps=steps,num_timesteps=0,buffer=RolloutBuffer(),gamma=.99,
                    act=lambda x:(np.zeros(2),0.,v(x)),get_value=v)
                _,last,_=PPO.collect_rollout(agent,Env(),0.,ep_stats=[0.,0],timeout_is_failure=fixed)
                adv,_=agent.buffer.compute_gae(last,.99,.95)
                values.append(float(adv[0]))
                if steps==2:assert np.isclose(adv[1],.99*2-reset_v)
            assert (values[0]==values[1])==fixed
            if fixed:assert values==[-15.,-15.]
            results[f"{steps}_{fixed}"]=values
    return results
if __name__=="__main__":print(run())
