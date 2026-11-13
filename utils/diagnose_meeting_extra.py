"""Additional bounded diagnostics. Action interventions are NOT policy scores."""
import concurrent.futures as cf
import json
from pathlib import Path
import shutil
import numpy as np
from utils import diagnose_meeting_av as d

a=d.a
OUT=a.ROOT/'results/meeting_diagnosis_20261114/extra'

class TimingProbe(d.RecordedEnv):
    def reset(self,*args,**kwargs):
        self.last_change=-100.;self.interventions=0
        return super().reset(*args,**kwargs)

    def step(self,action):
        requested=action.copy();applied=action.copy();t=self.step_count*self.step_length
        if (self.probe=='startup1' and t<1.-1e-6) or (self.probe=='interval1' and t-self.last_change<1.-1e-6):
            applied[1]=0.
        self.interventions+=int(applied[1]!=requested[1])
        obs,reward,done,cut,info=super().step(applied)
        if info['lane_after'] is not None and info['lane_before']!=info['lane_after']:
            self.last_change=t
        info['probe_requested_action']=requested.tolist()
        info['probe_applied_action']=applied.tolist()
        info['probe_interventions']=self.interventions
        self.history[-1]['probe_requested_action']=requested.tolist()
        if self.step_count*self.step_length>=60 and not done:
            cut=True;info['diagnostic_window_end']=True
        if done or cut:
            a.r.save(self.record_path/'detailed.json',dict(metrics=dict(self.metrics),interventions=self.interventions,last_30_seconds=list(self.history)))
        return obs,reward,done,cut,info

def run(spec):
    a.OUT=OUT;TimingProbe.probe=spec[0];TimingProbe.record_path=OUT/'runs'/f'{spec[0]}-{spec[1]}'
    a.r.c.ClarityEnv=TimingProbe
    row=a.one(spec)
    row.update(forced_correction=True,diagnostic_action_intervention=spec[0],not_policy_performance=True,window_limit_seconds=60)
    if row['outcome']=='timeout':row['outcome']='diagnostic_window_end'
    a.r.save(TimingProbe.record_path/'complete.json',row)
    return row

def sensitivity():
    modeldir=a.r.sources('baseline',22);manifest=json.loads((modeldir/'manifest.json').read_text())
    a.b.torch.set_num_threads(1);agent=a.b.PPO(obs_dim=47,act_dim=2,**manifest['hp']);agent.load(str(modeldir/'policy-600064.pt'))
    rows=[]
    for case in ('reference-route_09','reference-route_10','low-route_03'):
        trace=json.loads((d.OUT/'runs'/case/'detailed.json').read_text())['last_30_seconds']
        base=np.array(trace[-1]['observation'],dtype=np.float32)
        for gap in (0.,1.,3.,6.,15.,30.,50.):
            for relative in (-1.,-.5,0.,.5):
                obs=base.copy();obs[9]=gap/50.;obs[10]=relative
                action=agent.predict(obs,deterministic=True)
                rows.append(dict(case=case,gap_m=gap,relative_speed_normalized=relative,action=action.tolist()))
    limits=[]
    for case in ('reference-route_01','reference-route_02','reference-route_10'):
        trace=json.loads((d.OUT/'runs'/case/'detailed.json').read_text())['last_30_seconds']
        base=np.array(trace[-1]['observation'],dtype=np.float32)
        for limit in (11.11,16.67,22.22,27.78):
            obs=base.copy();obs[34]=limit/27.78
            limits.append(dict(case=case,current_limit_mps=limit,action=agent.predict(obs,deterministic=True).tolist()))
    a.r.save(OUT/'sensitivity.json',dict(note='Synthetic single-state inference; modified only named channels. Not a rollout or performance estimate.',front_gap_relative_speed=rows,current_limit=limits))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for mode in ('startup1','interval1'):
        dst=OUT/'inputs'/mode;dst.mkdir(parents=True,exist_ok=True)
        for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):shutil.copy2(d.OUT/'inputs/reference'/name,dst/name)
    cases=[(m,f'route_{r:02d}') for m in ('startup1','interval1') for r in (3,7,13,14)]
    a.r.save(OUT/'manifest.json',dict(cases=cases,simulation_seed=9101,policy='baseline-22/policy-600064.pt',traffic_vph=15000,no_training=True,not_policy_performance=True,interventions=dict(startup1='Suppress lateral action during first 1s only',interval1='Suppress lateral action until 1s after last observed lane index change'),diagnostic_window=60,source_sha=a.b.digest(a.ROOT/'utils/diagnose_meeting_extra.py')))
    sensitivity()
    with cf.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases))
    a.r.save(OUT/'summary.json',results)
    print(json.dumps([{k:r[k] for k in ('mode','route','outcome','seconds')} for r in results],indent=2))

if __name__=='__main__':main()
