"""Preserve the traffic safety candidate; add route rewards without changing observation/action."""
from env.traffic_safety_trials import TrafficSafetyEnv,traci

class RouteRecoveryEnv(TrafficSafetyEnv):
    def __init__(self,*args,condition=None,**kwargs):
        if condition is None:args,condition=args[:-1],args[-1]
        assert condition in ('control','route','opportunity')
        self.recovery_condition=condition
        super().__init__(*args,condition='lane',**kwargs)
        if condition!='control':self.route_condition='route_reward'

    def step(self,action):
        penalty=0.
        v=traci.vehicle
        if self.recovery_condition=='opportunity' and self.ego_id in v.getIDList():
            edge=v.getRoadID(self.ego_id);lane=v.getLaneIndex(self.ego_id)
            targets=self.navigation_context()['current_targets']
            remaining=traci.lane.getLength(v.getLaneID(self.ego_id))-v.getLanePosition(self.ego_id)
            if not edge.startswith(':') and targets and lane not in targets and remaining<=300 and v.getSpeed(self.ego_id)>.5:
                direction=1 if self.navigation.offset(lane,targets)>0 else -1
                hypothetical=action.copy();hypothetical[1]=direction
                risk=self.safety_context(hypothetical)['lane']
                if risk<.2 and self._quantize_lane_change(action[1])!=direction:penalty=-.02*self.step_length/.5
        obs,reward,done,cut,info=super().step(action)
        if done:penalty=0.
        self.reward_counts['opportunity_total']+=penalty
        self.reward_counts['opportunity_events']+=int(penalty<0)
        self.reward_counts['bonus_total']+=penalty
        info['route_selection_reward']+=penalty
        info['route_reward_parts']['opportunity']=penalty
        return obs,reward+penalty,done,cut,info
