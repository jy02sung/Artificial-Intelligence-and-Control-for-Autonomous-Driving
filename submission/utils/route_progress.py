import json
import numpy as np
from utils import driving_env as j


class FactorEnv:
    progress_cap = 2.0
    def __init__(self, arm, folder, seed):
        self.sim = j.environment("high_real_ratio",seed,folder)
        self.progress_enabled = arm in ("F1","F3")
        self.nav_enabled = arm in ("F2","F3")
        self.lane_lengths = {}
        self.route_coords = {}
        self.training_16 = None
        self.training_log = None
        self.training_steps = 0
        self.case16 = None

    def __getattr__(self,key):
        return getattr(self.sim,key)

    @property
    def ego_depart_lane(self):
        return self.sim.ego_depart_lane

    @ego_depart_lane.setter
    def ego_depart_lane(self,v):
        self.sim.ego_depart_lane=v

    def second(self,path,idx,remaining):
        count=0
        for k in range(idx,len(path)-1):
            if remaining>400: break
            edge,nxt=path[k:k+2]
            if (edge,nxt) in self.signs:
                count+=1
                if count==2: return [self.signs[edge,nxt],remaining/400]
            remaining+=self.navigation.link_lengths.get((edge,nxt),0)+self.navigation.lengths[nxt]
        return [0.,-1.]

    def observe(self,obs):
        cue=[0.,0.]
        if self.nav_enabled:
            s=self.snapshot
            cue=self.second(s[2],s[3],s[-1]) if s and not s[0].startswith(":") else [0.,-1.]
        return np.asarray([*obs,*cue],dtype=np.float32)

    def coordinate(self):
        edge,lane,path,idx,*_=self.snapshot
        if edge.startswith(":"):return self.high
        lane_id=f"{edge}_{lane}"
        if lane_id not in self.lane_lengths:
            self.lane_lengths[lane_id]=j.traci.lane.getLength(lane_id)
        fraction=1-np.clip(self.snapshot[-1]/self.lane_lengths[lane_id],0,1)
        return self.starts[idx]+fraction*self.navigation.lengths[edge]

    def reset(self,**kw):
        obs,info=self.sim.reset(**kw)
        path=tuple(self.snapshot[2])
        if path not in self.route_coords:
            starts=[0.]
            for edge,nxt in zip(path,path[1:]):
                starts.append(starts[-1]+self.navigation.lengths[edge]+self.navigation.link_lengths.get((edge,nxt),0))
            self.route_coords[path]=(starts,starts[-1]+self.navigation.lengths[path[-1]])
        self.starts,self.total=self.route_coords[path]
        self.high=0.
        self.start=self.high=self.coordinate()
        if self.training_log is not None:
            self.distance_meter=TimeoutDistances()
            self.episode_start=self.training_steps
            self.traffic_seed=kw.get('seed')
        if self.training_16 is not None:
            self.case16=None
            if self.episode_route=="route_16" and self.snapshot[1]==0:
                self.case16=dict(seed=kw.get("seed"),left=0,changed=False,main=False,outcome="partial")
                self.training_16.append(self.case16)
        return self.observe(obs),info

    def step(self,action):
        before=self.snapshot
        obs,reward,done,cut,info=self.sim.step(action)
        if self.case16 is not None:
            if before[0]=="86198728" and before[1]==0:
                self.case16["left"]+=int(action[1]>0)
                if self.snapshot and self.snapshot[0]==before[0] and self.snapshot[1]>0:
                    self.case16["changed"]=True
            if info["fork_passed"] and self.last_external[1]=="86198725":self.case16["main"]=True
            if done or cut:
                self.case16["outcome"]="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout"

        coord=self.high
        if info["arrived"]:coord=self.total
        elif self.snapshot and info["route_intact"] and not info["collided"]:coord=self.coordinate()
        gain=max(0.,coord-self.high)
        self.high=max(self.high,coord)
        bonus=self.progress_cap*gain/(self.total-self.start) if self.progress_enabled else 0.
        info["progress_bonus"]=bonus
        if self.training_log is not None:
            self.training_steps+=1
            length=self.training_steps-self.episode_start
            metrics=self.distance_meter.observe(self,length,info)
            if done or cut:
                outcome="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout" if cut else "other"
                record=dict(step=self.training_steps,route=self.episode_route,traffic_seed=self.traffic_seed,
                            episode_steps=length,outcome=outcome,timeout_distances=metrics if outcome=="timeout" else None)
                with self.training_log.open("a",encoding="utf8") as fp:
                    fp.write(json.dumps(record)+"\n")
        return self.observe(obs),reward+bonus,done,cut,info


class TimeoutDistances:
    """Measure longitudinal trip progress at a 1000-step timeout."""
    def __init__(self):
        self.at_850 = None
        self.link_anchor = None

    def observe(self, env, step, info):
        snapshot = env.snapshot
        if snapshot is None:
            return None
        edge = snapshot[0]
        # Anchor each internal link to the last known external-road progress.
        # SUMO odometry includes distance inside junctions.
        if edge.startswith(":") and self.link_anchor is None:
            self.link_anchor = (env.high, j.traci.vehicle.getDistance(env.ego_id),
                                j.traci.vehicle.getLanePosition(env.ego_id))
        elif not edge.startswith(":"):
            self.link_anchor = None
        if step not in (850, 1000):
            return None
        odo = j.traci.vehicle.getDistance(env.ego_id)
        if step == 850:
            self.at_850 = odo
            return None
        if info["stalled"] or info["arrived"] or info["collided"]:
            return None
        if self.link_anchor:
            # First internal lane starts at the end of the last external edge.
            idx = env.last_external[0]
            origin = env.starts[idx] + env.navigation.lengths[env.last_external[1]]
            _, entry_odo, entry_pos = self.link_anchor
            coordinate = origin + entry_pos + odo - entry_odo
        else:
            coordinate = env.coordinate()
        assert self.at_850 is not None and odo >= self.at_850 - 1e-5
        return dict(remaining_m=round(float(max(0, env.total-coordinate)),3),
                    last30_m=round(max(0., odo-self.at_850),3),
                    route_length_m=round(env.total,3), end_edge=edge,
                    end_lane=snapshot[1], end_speed_mps=round(snapshot[7],3),
                    intact=bool(info["route_intact"]))
