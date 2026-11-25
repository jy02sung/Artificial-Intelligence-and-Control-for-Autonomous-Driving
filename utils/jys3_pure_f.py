"""Factorial jys3 PPO experiment: route progress reward x second road-level fork."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import numpy as np
from utils import jys3 as j
from utils.jys3_ppo_compare import OUT, STEPS, save

ROOT = OUT / "pure_f"
ARMS = ("F0","F1","F2","F3")

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

def agent(seed):
    j.torch.set_num_threads(1)
    return j.m.PPO(obs_dim=52,act_dim=2,buffer=j.m.RolloutBuffer(),seed=seed,**j.HP)

def train(job, initial_weights=None):
    arm,seed,marks=job
    folder=ROOT/f"{arm}_{seed}"
    folder.mkdir(exist_ok=False)
    model=agent(seed)
    if initial_weights is not None:
        model.policy.load_state_dict(j.torch.load(initial_weights, weights_only=True)['policy'])
    env=FactorEnv(arm,folder/"map",seed)
    try:
        env.training_16=[]
        env.training_log=folder/'training_outcomes.jsonl'
        sched=j.ScheduledEpisodes(env,seed,folder/"episodes.jsonl")
        obs,_=sched.reset()
        cues={}
        for route in (13,14):
            path=env.routes[f"route_{route:02d}"]
            idx=path.index("170059994")
            cues[route]=env.second(path,idx,40.)
        assert cues[13][0]==-cues[14][0] and cues[13][0]!=0 and abs(cues[13][1]-cues[14][1])<1e-6
        save(folder/"preflight.json",dict(second_cues=cues,initial_hash=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()))
        stats=[0.,0]
        for n in marks:
            while model.num_timesteps<n:
                obs,last,_=model.collect_rollout(sched,obs,ep_stats=stats,steps=min(2048,n-model.num_timesteps),timeout_is_failure=True)
                adv,ret=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
                model.update(adv,ret)
            j.atomic_torch(dict(policy=model.policy.state_dict()),folder/f"policy_{n}.pt")
            save(folder/f"training16_{n}.json",env.training_16)
    finally:env.close()

def evaluate(job):
    arm,seed,n,rep,base=job
    folder=ROOT/f"{arm}_{seed}"
    model=agent(seed)
    model.policy.load_state_dict(j.torch.load(folder/f"policy_{n}.pt",weights_only=True)["policy"])
    model.policy.eval()
    rows=[]
    with TemporaryDirectory() as tmp:
        env=FactorEnv(arm,Path(tmp),seed)
        try:
            for route in range(1,17):
                name=f"route_{route:02d}"
                env.ego_depart_lane=str(min(rep%2,env.counts[env.routes[name][0]]-1))
                traffic_seed=base+(rep//2)*100+route
                obs,_=env.reset(seed=traffic_seed,options={"route":name})
                start=[env.snapshot[i] for i in (0,1,7,-1)]
                slow=moves=lefts=slow_side_near=0
                actual_left=None
                first=None
                prep=None
                forks=[]
                bonus=0.
                track=[]
                speed_totals=Counter()
                distance_meter=TimeoutDistances()
                timeout_distances=None
                for t in range(1,1001):
                    edge,lane,_,idx,_,_,_,v,remaining=env.snapshot
                    if route in (13,14) and edge=="170059994":prep=lane
                    act=model.predict(obs,deterministic=True)
                    if route==16 and idx==0 and edge=="86198728" and lane==0 and act[1]>0:
                        lefts+=1
                        if first is None:first=round(remaining,1)
                    if t%25==1 and route in (13,14,16):
                        track.append([t,edge,lane,round(remaining,1),round(v,2),round(float(act[0]),2),int(act[1])])
                    obs,_,done,cut,info=env.step(act)
                    post=env.snapshot
                    timeout_distances=distance_meter.observe(env,t,info)
                    speed_totals.update(info.get("speed_metrics", {}))
                    if post and not(done or edge.startswith(":")) and edge==post[0] and lane!=post[1]:moves+=1
                    if route==16 and idx==0 and edge=="86198728" and lane==0 and post and post[0]==edge and post[1]>lane and actual_left is None:
                        actual_left=round(remaining,1)
                    if info["fork_passed"]:forks.append(env.last_external[1])
                    if post and not done and info["speed"] is not None and info["speed"]<=.5 and (info["gap"] is None or info["gap"]>25):
                        slow+=1
                        if not post[0].startswith(":") and post[3]+1<len(post[2]):
                            targets=env.links.get(post[0],{}).get(post[2][post[3]+1],())
                            if targets and post[1] not in targets:
                                direction=1 if min(targets,key=lambda i:abs(i-post[1]))>post[1] else -1
                                neighbors=env._scan_lane_neighbors(direction)
                                slow_side_near+=int(any(x and x[0]<=10. for x in neighbors))
                    bonus+=info["progress_bonus"]
                    if done or cut:break
                outcome="arrival" if info["arrived"] else "collision" if info["collided"] else "stall" if info["stalled"] else "timeout" if cut else "other"
                assert bonus<=env.progress_cap+1e-5 and (outcome!="arrival" or abs(bonus-(env.progress_cap if env.progress_enabled else 0))<1e-5)
                rows.append(dict(speed_metrics=dict(speed_totals),timeout_distances=timeout_distances,route=route,case=rep,traffic_seed=traffic_seed,start=start,outcome=outcome,steps=t,slow=slow,slow_side_near=slow_side_near,moves=moves,actual_left=actual_left,first_left=first,left_cmds=lefts,prep=prep,forks=forks,bonus=bonus,track=track))
        finally:env.close()
    save(folder/f"eval_{base}_{n}_{rep}.json",rows)

def run(arms,seeds,marks,base,eval_marks):
    with ProcessPoolExecutor(min(4,len(arms)*len(seeds))) as trains, ProcessPoolExecutor(8) as ev:
        ts={trains.submit(train,(a,s,marks)):(a,s) for s in seeds for a in arms}
        jobs=[]
        for f in as_completed(ts):
            f.result()
            a,s=ts[f]
            jobs.extend(ev.submit(evaluate,(a,s,n,r,base)) for n in eval_marks for r in range(4))
        for f in as_completed(jobs):f.result()

if __name__=="__main__":
    ROOT.mkdir(exist_ok=False)
    save(ROOT/"protocol.json",dict(arms=ARMS,steps=STEPS,seed=11,hp=j.HP,eval_base=34110000,progress_budget=2,nav_horizon_m=400))
    run(ARMS,[11],STEPS,34110000,STEPS)

