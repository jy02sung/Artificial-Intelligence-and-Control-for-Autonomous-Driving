"""Step-balanced full-route/curriculum scheduler with full deterministic replay."""
import collections,copy,hashlib,json
import numpy as np
from utils.curriculum_data import make_env,context,s,OUT

class CurriculumEnv:
    groups={'merge':['route_01','route_02'],'safe_change':['route_11'],'consecutive':['route_13','route_14','route_15']}
    def __init__(self,seed):
        self.env=make_env();self.rng=np.random.default_rng(seed)
        self.counts=collections.Counter();self.episodes=[];self.spec=None;self.history=[];self.full_schedule=[];self.group_cursor=0
        self.last_obs=None;self.replaying=False;self.exposure=collections.Counter();self.episode_risks=0;self.episode_brakes=0
        self.catalog=json.loads((OUT/'data-manifest.json').read_text())['episodes']
    def __getattr__(self,name):return getattr(self.env,name)
    def reset(self,seed=None,options=None):
        mode='full' if self.counts['full']<=self.counts['risk'] else 'risk'
        if mode=='full':
            if not self.full_schedule:self.full_schedule=list(self.rng.permutation([f'route_{i:02d}' for i in range(1,17)]))
            route=str(self.full_schedule.pop());group='full'
        else:
            group=list(self.groups)[self.group_cursor%3];self.group_cursor+=1
            route=str(self.rng.choice(self.groups[group]))
        self.spec=dict(route=route,seed=int(self.rng.integers(920001,980000)),warmup_steps=int(self.rng.choice([4000,4500,5000])),mode=mode,group=group)
        if mode=='risk' and self.group_cursor%2:
            choices=[x for x in self.catalog if x['route'] in self.groups[group]]
            chosen=choices[int(self.rng.integers(len(choices)))]
            self.spec.update(route=chosen['route'],seed=chosen['seed'],warmup_steps=chosen['warmup_steps'],catalog_outcome=chosen['outcome'])
            route=chosen['route']
        self.episode_risks=0;self.episode_brakes=0
        self.env.warmup_steps=self.spec['warmup_steps'];self.history=[]
        self.last_obs,info=self.env.reset(seed=self.spec['seed'],options={'route':route})
        return self.last_obs,info
    def step(self,action):
        ctx=context(self.env,action) if self.env.step_count%5==0 else None
        obs,r,done,cut,info=self.env.step(action)
        self.history.append(np.asarray(action).tolist());self.last_obs=obs
        self.counts[self.spec['mode']]+=1;self.counts['route/'+self.spec['route']]+=1
        self.counts['group/'+self.spec['group']]+=1
        if ctx is not None:
            risk=ctx['merge_risk']>0 or ctx['ttc'] is not None and ctx['ttc']<5 or ctx['lane_risk']>.2
            self.exposure['sampled_states']+=1;self.exposure['risk_states']+=int(risk)
            self.exposure['risk_brake_attempts']+=int(risk and action[0]<0)
            self.episode_risks+=int(risk);self.episode_brakes+=int(risk and action[0]<0)
            v=s.a.b.traci.vehicle
            self.exposure['risk_actual_decelerations']+=int(risk and self.env.ego_id in v.getIDList() and v.getSpeed(self.env.ego_id)<ctx['speed']-1e-5)
            self.exposure['lane_preparation_states']+=int(ctx['constrained'])
        if done or cut:
            outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit'
            self.episodes.append(dict(**self.spec,steps=len(self.history),outcome=outcome,risk_samples=self.episode_risks,risk_brake_attempts=self.episode_brakes,risk_safe_pass=bool(outcome=='arrival' and self.episode_risks)))
        return obs,r,done,cut,info
    def fingerprint(self):
        v=s.a.b.traci.vehicle
        rows=[(x,v.getRoadID(x),v.getLaneIndex(x),v.getLanePosition(x),v.getSpeed(x)) for x in sorted(v.getIDList())]
        return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state,counts=dict(self.counts),episodes=self.episodes,spec=self.spec,history=self.history,full_schedule=self.full_schedule,group_cursor=self.group_cursor,obs=self.last_obs,exposure=dict(self.exposure),fingerprint=self.fingerprint(),episode_risks=self.episode_risks,episode_brakes=self.episode_brakes)
    def load_state_dict(self,state):
        spec=state['spec'];self.env.warmup_steps=spec['warmup_steps']
        obs,_=self.env.reset(seed=spec['seed'],options={'route':spec['route']})
        for action in state['history']:
            obs,_,done,cut,_=self.env.step(np.asarray(action,dtype=np.float32))
            assert not(done or cut),'Resume checkpoint must follow reset at terminal boundary'
        np.testing.assert_array_equal(obs,state['obs']);assert self.fingerprint()==state['fingerprint'],'SUMO replay differs'
        self.rng.bit_generator.state=state['rng'];self.counts=collections.Counter(state['counts']);self.episodes=state['episodes'];self.spec=spec;self.history=state['history'];self.full_schedule=state['full_schedule'];self.group_cursor=state['group_cursor'];self.last_obs=obs;self.exposure=collections.Counter(state['exposure']);self.episode_risks=state['episode_risks'];self.episode_brakes=state['episode_brakes']
        return obs
    def close(self):self.env.close()
