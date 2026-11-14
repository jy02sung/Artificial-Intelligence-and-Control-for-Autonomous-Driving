"""Navigation-only and route-reward-only ablations; no action override."""
import numpy as np
from env.navigation_observation import RouteNavigation
from env.observation_reward_trials import TrialEnv
from env.sumo_env import traci

class PreparedNavigation(RouteNavigation):
    def prepared(self,path,index,lane,remaining,internal=False):
        result=dict(valid=False,prepare_targets=[],distance=None,future_targets=[],future_direction=0,
                    future_direction_valid=False,cost=0,decision_edge=None)
        if internal or index<0 or index+1>=len(path):return result
        distance=max(remaining,0.)
        for j in range(index,len(path)-1):
            if distance>self.horizon:break
            edge,nxt=path[j:j+2];targets={a for a,b in self.links[edge,nxt]}
            if targets and len(targets)<self.counts[edge]:
                solid=getattr(self,'solid',set())
                costs={i:min((abs(i-g) if edge not in solid or i==g else float('inf')) for g in targets) for i in range(self.counts[edge])}
                chosen=targets
                for k in range(j-1,index-1,-1):
                    src,dst=path[k:k+2]
                    outgoing={i:min((costs[b] for a,b in self.links[src,dst] if a==i),default=float('inf')) for i in range(self.counts[src])}
                    best=min(outgoing.values())
                    chosen={i for i,c in outgoing.items() if c==best and np.isfinite(c)}
                    costs={i:min((abs(i-a)+c for a,c in outgoing.items() if src not in solid or i==a),default=float('inf')) for i in range(self.counts[src])}
                reach={lane}
                for k in range(index,j):reach={b for a,b in self.links[path[k],path[k+1]] if a in reach}
                landing=min(reach,key=lambda i:abs(self.offset(i,targets))) if reach else None
                result.update(valid=bool(chosen),prepare_targets=sorted(chosen),distance=distance,future_targets=sorted(targets),
                    future_direction=self.offset(landing,targets) if reach else 0,future_direction_valid=bool(reach),
                    cost=costs.get(lane,float('inf')),decision_edge=edge)
                return result
            distance+=self.link_lengths.get((edge,nxt),0.)+self.lengths[nxt]
        return result

def selection_reward(state,key,before,after,progress,terminal):
    if terminal:return 0.
    item=state.setdefault(key,dict(best=before,paid=0.))
    reward=.1*max(min(item['best'],before)-after,0)
    item['best']=min(item['best'],after)
    if before==0 and after>0:reward-=.1
    if after==0:
        earned=min(max(progress,0)*.005,max(.5-item['paid'],0))
        item['paid']+=earned;reward+=earned
    return reward

class RouteTrialEnv(TrialEnv):
    def __init__(self,cfg,road,ego,sim,obs,action,reward,traffic,seed,navigation,condition):
        assert condition in ('baseline','guidance','route_reward')
        self.route_condition=condition;self.route_rewards={}
        navigation.solid=set(road.get('solid_lane_edges',()))
        super().__init__(cfg,road,ego,sim,obs,action,reward,traffic,seed,navigation,'baseline')

    def reset(self,*args,**kwargs):
        self.route_rewards={}
        return super().reset(*args,**kwargs)

    def _get_obs(self):
        obs=super()._get_obs()
        if self.route_condition!='guidance' or self.ego_id not in traci.vehicle.getIDList():return obs
        v=traci.vehicle;edge=v.getRoadID(self.ego_id);lane=v.getLaneIndex(self.ego_id)
        current=self.navigation_context()
        n=self.navigation.prepared(self.routes[self.episode_route],v.getRouteIndex(self.ego_id),lane,
            traci.lane.getLength(v.getLaneID(self.ego_id))-v.getLanePosition(self.ego_id),edge.startswith(':'))
        obs[33]=np.clip(self.navigation.offset(lane,n['prepare_targets'])/3,-1,1)
        extra=np.zeros(16,dtype=np.float32)
        extra[:4]=[1 if current['current_targets'] else -1,1 if n['valid'] else -1,1 if n['decision_edge'] else -1,n['distance']/300 if n['distance'] is not None else -1]
        if n['decision_edge']:
            count=self.navigation.counts[n['decision_edge']];assert count<=6
            extra[4:10]=[1 if i in n['future_targets'] else -1 if i<count else 0 for i in range(6)]
        extra[10:13]=[np.clip(n['future_direction']/3,-1,1),1 if n['future_direction_valid'] else -1,min(n['cost']/5,1)]
        obs[47:]=extra
        assert obs.shape==(63,) and np.isfinite(obs).all()
        return obs

    def step(self,action):
        v=traci.vehicle;before=None
        if self.route_condition=='route_reward' and self.ego_id in v.getIDList():
            edge=v.getRoadID(self.ego_id);lane=v.getLaneIndex(self.ego_id);nav=self.navigation_context()
            remaining=traci.lane.getLength(v.getLaneID(self.ego_id))-v.getLanePosition(self.ego_id)
            targets=nav['current_targets']
            if targets and len(targets)<self.navigation.counts.get(edge,0) and remaining<=300:
                before=(edge,v.getRouteIndex(self.ego_id),v.getLanePosition(self.ego_id),abs(self.navigation.offset(lane,targets)),targets)
        obs,reward,done,cut,info=super().step(action)
        bonus=0.
        if before and not done and self.ego_id in v.getIDList() and v.getRoadID(self.ego_id)==before[0]:
            edge,idx,pos,distance,targets=before
            bonus=selection_reward(self.route_rewards,(edge,idx),distance,abs(self.navigation.offset(v.getLaneIndex(self.ego_id),targets)),v.getLanePosition(self.ego_id)-pos,False)
        info['route_selection_reward']=bonus
        return obs,reward+bonus,done,cut,info
