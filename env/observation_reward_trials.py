"""One-factor observation/reward trials. No action overrides or route teachers."""
import numpy as np
from gymnasium import spaces
from env.navigation_clarity import ClarityEnv
from env.sumo_env import traci

CONDITIONS=('baseline','obs_presence','obs_risk','reward_gap','reward_block','reward_lane')

def approach_risk(neighbor,ego_speed,front):
    if neighbor is None:return 0.
    gap,speed=neighbor
    if gap<=0:return 1.
    closing=ego_speed-speed if front else speed-ego_speed
    return float(np.clip(3.*max(closing,0.)/gap,0,1))

def lane_deficit(front,rear,ego_speed):
    deficits=[]
    for neighbor,is_front in ((front,True),(rear,False)):
        if neighbor is None:continue
        gap,speed=neighbor
        required=2.5+1.5*(ego_speed if is_front else speed)
        deficits.append(float(np.clip(1.-gap/required,0,1)))
    return max(deficits,default=0.)

class TrialEnv(ClarityEnv):
    def __init__(self,cfg,road,ego,sim,obs,action,reward,traffic,seed,navigation,condition):
        if condition not in CONDITIONS:raise ValueError(condition)
        self.condition=condition
        rw=dict(reward)
        if condition=='reward_gap':rw['time_gap_s']=1.5
        if condition=='reward_block':rw['blocked_penalty']=0.
        super().__init__(cfg,road,ego,sim,obs,action,rw,traffic,(),seed,navigation=navigation,clarity='baseline')
        self.observation_space=spaces.Box(-1,1,(63,),np.float32)

    def _get_obs(self):
        original=super()._get_obs();extra=np.zeros(16,dtype=np.float32)
        if self.ego_id in traci.vehicle.getIDList():
            if self.condition=='obs_presence':
                extra[:10]=[float(self._neighbor(offset,front) is not None) for offset in self.obs_offsets for front in (True,False)]
            elif self.condition=='obs_risk':
                speed=traci.vehicle.getSpeed(self.ego_id)
                extra[10:]=[approach_risk(self._neighbor(offset,front),speed,front) for offset in (1,0,-1) for front in (True,False)]
        result=np.concatenate((original,extra)).astype(np.float32)
        assert result.shape==(63,) and np.isfinite(result).all()
        return result

    def step(self,action):
        deficit=0.
        if self.condition=='reward_lane' and self.ego_id in traci.vehicle.getIDList():
            direction=self._quantize_lane_change(action[self.lane_change_index])
            if direction:
                deficit=lane_deficit(self._neighbor(direction,True),self._neighbor(direction,False),traci.vehicle.getSpeed(self.ego_id))
        obs,reward,done,cut,info=super().step(action)
        penalty=-.1*deficit*self.step_length if not done else 0.
        info['trial_reward_components']=dict(base=reward,lane_risk=penalty)
        return obs,reward+penalty,done,cut,info
