"""Frozen-policy M1/M2/M7 replay. Instrumentation only; no training or override."""
import collections
import concurrent.futures as cf
import json
import os
import shutil
import xml.etree.ElementTree as E
from utils import av_hv_release_check as a
from utils.hv_measured_traffic import allocate

OUT=a.ROOT/'results/meeting_diagnosis_20261114/av'
Base=a.r.c.ClarityEnv

class RecordedEnv(Base):
    def reset(self,*args,**kwargs):
        self.history=collections.deque(maxlen=151)
        self.metrics=collections.Counter()
        result=super().reset(*args,**kwargs)
        self.last_obs=result[0].copy()
        return result

    def step(self,action):
        tr=a.b.traci;v=tr.vehicle;ego=self.ego_id
        lane=v.getLaneID(ego);edge=v.getRoadID(ego);pos=v.getLanePosition(ego)
        speed=v.getSpeed(ego);xy=v.getPosition(ego)
        neighbors=[]
        for vid in v.getIDList():
            if vid==ego:continue
            x,y=v.getPosition(vid)
            if (x-xy[0])**2+(y-xy[1])**2<=60**2:
                neighbors.append(dict(id=vid,lane=v.getLaneID(vid),position=v.getLanePosition(vid),speed=v.getSpeed(vid),acceleration=v.getAcceleration(vid),length=v.getLength(vid),xy=[x,y]))
        raw={str(m):list(v.getNeighbors(ego,m)) for m in (0,1,2,3)}
        filtered={str(d):dict(front=self._neighbor(d,True),rear=self._neighbor(d,False),scan=self._scan_lane_neighbors(d)) for d in (-1,1)}
        lead=v.getLeader(ego,200)
        record=dict(t=self.step_count*.2,lane=lane,edge=edge,position=pos,speed=speed,acceleration=v.getAcceleration(ego),limit=tr.lane.getMaxSpeed(lane),remaining=tr.lane.getLength(lane)-pos,leader=lead,neighbors=neighbors,raw_neighbors=raw,filtered=filtered,observation=self.last_obs.tolist(),action=action.tolist(),navigation=self.navigation_context())
        self.metrics['steps']+=1
        self.metrics['negative_gap_steps']+=int(any(g<0 for pairs in raw.values() for _,g in pairs))
        if lead and lead[1]<6:self.metrics['gap_under6_steps']+=1;self.metrics['positive_action_gap_under6_steps']+=int(action[0]>0)
        result=super().step(action)
        obs,reward,done,cut,info=result
        record.update(reward=reward,info=info,collision_ids=list(tr.simulation.getCollidingVehiclesIDList()))
        if ego in v.getIDList():record['actual_acceleration']=v.getAcceleration(ego)
        self.history.append(record);self.last_obs=obs.copy()
        if done or cut:
            a.r.save(self.record_path/'detailed.json',dict(metrics=dict(self.metrics),last_30_seconds=list(self.history)))
        return result

def run(spec):
    a.OUT=OUT
    dest=OUT/'runs'/f'{spec[0]}-{spec[1]}'
    RecordedEnv.record_path=dest
    a.r.c.ClarityEnv=RecordedEnv
    return a.one(spec)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=a.SOURCES['reference'];routes=E.parse(src/'highway.rou.xml')
    original=[int(f.get('number')) for f in routes.findall('flow')]
    for mode,fraction in [('reference',1),('low',.1),('none',0)]:
        d=OUT/'inputs'/mode;d.mkdir(parents=True,exist_ok=True)
        for name in ('network.net.xml','run.sumocfg'):shutil.copy2(src/name,d/name)
        tree=E.parse(src/'highway.rou.xml')
        if fraction==0:
            for f in tree.findall('flow'):tree.getroot().remove(f)
        elif fraction!=1:
            for f,n in zip(tree.findall('flow'),allocate(round(sum(original)*fraction),original)):f.set('number',str(n))
        tree.write(d/'highway.rou.xml',encoding='utf8',xml_declaration=True)
    cases=[(m,f'route_{i:02d}') for m in ('reference','none','low') for i in (1,2,3,7,9,10,13,14)]
    a.r.save(OUT/'manifest.json',dict(cases=cases,simulation_seed=9101,policy='baseline-22/policy-600064.pt',density_vph=dict(reference=15000,low=1500,none=0),workers=4,training=False,forced_correction=False,commit=a.b.COMMIT,code_sha=a.b.digest(a.ROOT/"utils/diagnose_meeting_av.py"),inputs={m:{n:a.b.digest(OUT/'inputs'/m/n) for n in ('network.net.xml','highway.rou.xml','run.sumocfg')} for m in ('reference','none','low')}))
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run,cases))
    a.r.save(OUT/'summary.json',results)
    print(json.dumps([{k:r.get(k) for k in ('mode','route','outcome','seconds')} for r in results],indent=2))

if __name__=='__main__':main()
