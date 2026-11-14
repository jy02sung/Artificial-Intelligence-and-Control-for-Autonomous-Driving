"""No-traffic route-learning ablations with training reward accounting."""
import collections
import numpy as np
from env.route_following_trials import RouteTrialEnv, traci

def compact_observation(full):
    out=full.copy();out[51:57]=0;out[59:]=0
    return out

def reward_parts(bonus,maintenance,scale):
    residual=bonus-maintenance
    return dict(selection=max(residual,0)*scale,maintenance=maintenance*scale,departure=min(residual,0)*scale)

class RouteLearningEnv(RouteTrialEnv):
    def __init__(self,*args,condition=None,**kwargs):
        # Existing factory supplies condition positionally.
        if condition is None:args,condition=args[:-1],args[-1]
        self.learning_condition=condition;self.reward_counts=collections.Counter()
        parent='guidance' if condition in ('compact','guidance') else 'route_reward' if condition in ('route_reward','scale5','dwell') else 'baseline'
        super().__init__(*args,condition=parent,**kwargs)

    def _get_obs(self):
        obs=super()._get_obs()
        return compact_observation(obs) if self.learning_condition=='compact' else obs

    def step(self,action):
        paid=sum(x['paid'] for x in self.route_rewards.values());wrong=False
        v=traci.vehicle
        if self.ego_id in v.getIDList():
            edge=v.getRoadID(self.ego_id);targets=self.navigation_context()['current_targets']
            remaining=traci.lane.getLength(v.getLaneID(self.ego_id))-v.getLanePosition(self.ego_id)
            wrong=not edge.startswith(':') and bool(targets) and v.getLaneIndex(self.ego_id) not in targets and remaining<=300
        obs,reward,done,cut,info=super().step(action)
        old=info['route_selection_reward'];maintenance=sum(x['paid'] for x in self.route_rewards.values())-paid
        parts=reward_parts(old,maintenance,5 if self.learning_condition=='scale5' else 1)
        parts['dwell']=-.02*self.step_length/.5 if self.learning_condition=='dwell' and wrong and not done else 0.
        bonus=sum(parts.values());base=reward-old
        self.reward_counts.update(dict(steps=1,base_total=base,bonus_total=bonus))
        for key,value in parts.items():
            self.reward_counts[key+'_total']+=value
            self.reward_counts[key+'_events']+=int(abs(value)>1e-8)
        info['route_selection_reward']=bonus;info['route_reward_parts']=parts
        return obs,base+bonus,done,cut,info
