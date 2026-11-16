"""Safety reward magnitude ablations and anticipatory observation combination."""
import numpy as np
from env.junction_lookahead_trials import JunctionLookaheadEnv
from utils.run_traffic_recovery_auto import AdaptiveEnv

CONDITIONS = ('collision20', 'lane4', 'anticipatory')


def safety_extra(condition, collided, terminal, lane_risk, merge_risk, speed_ratio):
    assert condition in CONDITIONS
    return dict(collision=-15. if collided and condition in ('collision20','anticipatory') else 0.,
                lane=-.3*lane_risk if not terminal and condition in ('lane4','anticipatory') else 0.,
                merge=-.1*merge_risk*speed_ratio if not terminal and condition=='anticipatory' else 0.)


class SafetyStrengthEnv(JunctionLookaheadEnv):
    def __init__(self,*args,condition=None,**kwargs):
        if condition is None:
            args,condition=args[:-1],args[-1]
        assert condition in CONDITIONS
        self.strength_condition=condition
        super().__init__(*args,condition='ahead150',**kwargs)

    def _get_obs(self):
        if self.strength_condition!='anticipatory':
            return AdaptiveEnv._get_obs(self)
        obs=super()._get_obs()
        contexts=[self.safety_context(np.array([0.,d],dtype=np.float32)) for d in (1,0,-1)]
        obs[57:60]=[contexts[0]['lane'],contexts[1]['gap'],contexts[2]['lane']]
        return obs

    def step(self,action):
        risk=self.safety_context(action)['lane']
        merge=self.conflict_features()[0] if self.strength_condition=='anticipatory' else 0.
        from env.route_following_trials import traci
        speed=traci.vehicle.getSpeed(self.ego_id)/self.vmax if self.ego_id in traci.vehicle.getIDList() else 0.
        obs,reward,done,cut,info=super().step(action)
        extra=safety_extra(self.strength_condition,info['collided'],done,risk,merge,speed)
        total=sum(extra.values())
        for key,value in extra.items():
            name='strength_'+key
            self.reward_counts[name+'_total']+=value
            self.reward_counts[name+'_events']+=int(abs(value)>1e-8)
            info['route_reward_parts'][name]=value
        self.reward_counts['bonus_total']+=total
        info['route_selection_reward']+=total
        return obs,reward+total,done,cut,info
