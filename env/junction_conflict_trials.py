"""Local junction-conflict observation/reward ablations; never overrides actions."""
import math
import numpy as np
from utils.run_traffic_recovery_auto import AdaptiveEnv
from env.route_following_trials import traci

def conflict_risk(ego_dist,foe_dist,ego_speed,foe_speed):
    te=max(ego_dist,0)/max(ego_speed,.5);tf=max(foe_dist,0)/max(foe_speed,.5)
    risk=max(0,1-abs(te-tf)/2)*max(0,1-max(te,tf)/5)
    return risk,min(te/5,1),min(tf/5,1)

class ConflictEnv(AdaptiveEnv):
    def __init__(self,*args,condition=None,**kwargs):
        if condition is None:args,condition=args[:-1],args[-1]
        assert condition in ('control','observation','reward','combined')
        self.conflict_condition=condition
        super().__init__(*args,condition='route',**kwargs)

    def conflict_features(self):
        v=traci.vehicle
        if self.ego_id not in v.getIDList():return (0.,0.,0.)
        pos=v.getPosition(self.ego_id);speed=v.getSpeed(self.ego_id);best=(0.,0.,0.)
        # SUMO maps conflicting junction lanes; only physically visible vehicles qualify.
        # Priority response flags and future trajectories are not supplied to the policy.
        for item in v.getJunctionFoes(self.ego_id,self.visibility):
            vid,ed,fd,ex,fx=item[:5]
            if ex < -v.getLength(self.ego_id) or fx < -v.getLength(vid):continue
            q=v.getPosition(vid)
            if math.hypot(q[0]-pos[0],q[1]-pos[1])>self.visibility:continue
            value=conflict_risk(ed,fd,speed,v.getSpeed(vid))
            if value[0]>best[0]:best=value
        return best

    def _get_obs(self):
        obs=super()._get_obs()
        if self.conflict_condition in ('observation','combined'):obs[60:63]=self.conflict_features()
        assert np.isfinite(obs).all()
        return obs

    def step(self,action):
        risk=self.conflict_features()[0]
        speed=traci.vehicle.getSpeed(self.ego_id) if self.ego_id in traci.vehicle.getIDList() else 0
        obs,reward,done,cut,info=super().step(action)
        penalty=-.1*risk*min(speed/self.vmax,1) if not done and self.conflict_condition in ('reward','combined') else 0.
        self.reward_counts['junction_risk_steps']+=int(risk>0)
        self.reward_counts['junction_brake_steps']+=int(risk>0 and action[0]<0)
        self.reward_counts['junction_penalty_total']+=penalty
        self.reward_counts['bonus_total']+=penalty
        info['route_selection_reward']+=penalty
        info['route_reward_parts']['junction']=penalty
        return obs,reward+penalty,done,cut,info
