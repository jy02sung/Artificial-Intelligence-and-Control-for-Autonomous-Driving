"""Two independent observation additions for late lane changes; actions/rewards fixed."""
import numpy as np
from utils.run_traffic_recovery_auto import AdaptiveEnv
from env.route_following_trials import traci

class LaneFollowupEnv(AdaptiveEnv):
    def __init__(self,*args,condition=None,**kwargs):
        if condition is None:args,condition=args[:-1],args[-1]
        assert condition in ('future','lateral')
        self.followup_condition=condition
        super().__init__(*args,condition='route',**kwargs)
    def _get_obs(self):
        obs=super()._get_obs()
        v=traci.vehicle
        if self.ego_id not in v.getIDList():return obs
        if self.followup_condition=='future':
            n=self.navigation.prepared(self.routes[self.episode_route],v.getRouteIndex(self.ego_id),v.getLaneIndex(self.ego_id),traci.lane.getLength(v.getLaneID(self.ego_id))-v.getLanePosition(self.ego_id),v.getRoadID(self.ego_id).startswith(':'))
            obs[47:50]=[1 if n['future_direction_valid'] else -1,n['distance']/self.navigation.horizon if n['distance'] is not None else -1,np.clip(n['future_direction']/3,-1,1)]
        else:
            risks=[]
            for direction in (1,0,-1):
                context=self.safety_context(np.array([0.,direction],dtype=np.float32))
                risks.append(context['lane'] if direction else context['gap'])
            obs[57:60]=risks
        assert obs.shape==(63,) and np.isfinite(obs).all()
        return obs
