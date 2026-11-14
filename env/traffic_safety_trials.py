"""Reward-only safety ablations and explicit multi-factor candidate. No action overrides."""
import numpy as np
from env.route_learning_trials import RouteLearningEnv
from env.route_following_trials import RouteTrialEnv,traci

def gap_deficit(neighbor,speed,min_gap):
    if neighbor is None:return 0.
    return float(np.clip(1-(neighbor[0]+min_gap)/(min_gap+1.5*max(speed,0)),0,1))

def closing_risk(neighbor,speed,front,min_gap):
    if neighbor is None:return 0.
    gap,other=neighbor
    if gap<=0:return 1.
    closing=speed-other if front else other-speed
    return float(np.clip(3*max(closing,0)/(gap+min_gap),0,1))

class TrafficSafetyEnv(RouteLearningEnv):
    def __init__(self,*args,condition=None,**kwargs):
        if condition is None:args,condition=args[:-1],args[-1]
        assert condition in ('baseline','gap','lane','combo')
        self.safety_condition=condition
        super().__init__(*args,condition='route_reward' if condition=='combo' else 'baseline',**kwargs)
        if condition=='combo':self.rw=dict(self.rw,blocked_penalty=0.)

    def _get_obs(self):
        if self.safety_condition!='combo':return super()._get_obs()
        previous=self.route_condition;self.route_condition='guidance'
        try:obs=RouteTrialEnv._get_obs(self)
        finally:self.route_condition=previous
        if self.ego_id in traci.vehicle.getIDList():
            v=traci.vehicle.getSpeed(self.ego_id);mg=traci.vehicle.getMinGap(self.ego_id)
            obs[60:63]=[max(closing_risk(self._neighbor(o,True),v,True,mg),closing_risk(self._neighbor(o,False),v,False,mg)) for o in (1,0,-1)]
        return obs

    def safety_context(self,action):
        v=traci.vehicle
        if self.ego_id not in v.getIDList():return dict(gap=0.,lane=0.,ttc=None)
        speed=v.getSpeed(self.ego_id);mg=v.getMinGap(self.ego_id);front=self._neighbor(0,True)
        ttc=(front[0]+mg)/(speed-front[1]) if front and speed>front[1] else None
        direction=self._quantize_lane_change(action[1]);edge=v.getRoadID(self.ego_id)
        risk=0.
        if direction and not edge.startswith(':') and 0<=v.getLaneIndex(self.ego_id)+direction<traci.edge.getLaneNumber(edge):
            f,b=self._neighbor(direction,True),self._neighbor(direction,False)
            risk=max(gap_deficit(f,speed,mg),gap_deficit(b,b[1] if b else 0,mg),closing_risk(f,speed,True,mg),closing_risk(b,speed,False,mg))
        return dict(gap=gap_deficit(front,speed,mg),lane=risk,ttc=ttc)

    def step(self,action):
        risk=self.safety_context(action)
        obs,reward,done,cut,info=super().step(action)
        gap=-.1*risk['gap'] if not done and self.safety_condition in ('gap','combo') else 0.
        lane=-.1*risk['lane'] if not done and self.safety_condition in ('lane','combo') else 0.
        for name,value in (('gap',gap),('lane',lane)):
            self.reward_counts[name+'_total']+=value;self.reward_counts[name+'_events']+=int(abs(value)>1e-8)
        self.reward_counts['bonus_total']+=gap+lane
        self.reward_counts['risk_steps']+=int(risk['ttc'] is not None and risk['ttc']<3)
        self.reward_counts['risk_brake_requests']+=int(risk['ttc'] is not None and risk['ttc']<3 and action[0]<0)
        info['route_selection_reward']+=gap+lane
        info['route_reward_parts'].update(gap=gap,lane=lane);info['safety_context']=risk
        return obs,reward+gap+lane,done,cut,info
