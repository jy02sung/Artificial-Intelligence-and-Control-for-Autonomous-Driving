"""Summarize fixed-policy diagnostics without changing simulator or policy."""
import collections
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from env import sumo_env as se

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/meeting_diagnosis_20261114'

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def main():
    rows=[];evidence=[]
    for p in sorted((OUT/'av/runs').glob('*/complete.json')):
        r=json.loads(p.read_text());d=json.loads((p.parent/'detailed.json').read_text());xs=d['last_30_seconds'];last=xs[-1]
        ids=set(r.get('collision_ids',[]))-{'ego'}
        partners=[v for v in last['neighbors'] if v['id'] in ids]
        same_leader=bool(last['leader'] and last['leader'][0] in ids)
        hidden=[]
        for x in xs:
            for mode,pairs in x['raw_neighbors'].items():
                for vid,gap in pairs:
                    if vid in ids and gap<0:
                        direction='1' if int(mode)%2==0 else '-1'
                        kind='front' if int(mode)>=2 else 'rear'
                        hidden.append(dict(t=x['t'],id=vid,gap=gap,direction=direction,kind=kind,filtered=x['filtered'][direction][kind]))
        row=dict(mode=r['mode'],route=r['route'],outcome=r['outcome'],seconds=r['seconds'],insertion_delay=r['insertion_delay'],collision_edge=last['edge'] if ids else '',collision_with_current_leader=same_leader,partner_negative_gap_discard_count=sum(x['filtered'] is None for x in hidden),last_speed=last['speed'],last_limit=last['limit'],last_accel_action=last['action'][0],last_lane_action=last['action'][1],last_leader_gap=last['leader'][1] if last['leader'] else '',route_overrides=r['route_overrides'])
        rows.append(row)
        evidence.append(dict(**row,metrics=d['metrics'],partners=partners,discarded_partner_samples=hidden,last_observation=last['observation'],last_info=last['info']))
    with (OUT/'av-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('av-evidence.json',evidence)
    aggregates={mode:dict(collections.Counter(r['outcome'] for r in rows if r['mode']==mode)) for mode in ('none','low','reference')}
    save('av-aggregate.json',aggregates)
    # Demonstrate existing function semantics, not a replacement detector.
    obj=SimpleNamespace(ego_id='ego',visibility=50,_NEIGHBOR_MODES=se.SumoHighwayEnv._NEIGHBOR_MODES)
    probes=[]
    for name,gap,neighbor_speed in [('overlap',-3.,25.),('near',1.,10.),('approaching',20.,10.),('receding',20.,35.),('beyond_visibility',51.,20.)]:
        with patch.object(se.traci.vehicle,'getNeighbors',return_value=(('other',gap),)),patch.object(se.traci.vehicle,'getSpeed',return_value=neighbor_speed):
            result=se.SumoHighwayEnv._neighbor(obj,1,True)
        assert (result is None)==(gap<0 or gap>50)
        probes.append(dict(case=name,raw_gap=gap,returned=result))
    with patch.object(se.traci.vehicle,'getNeighbors',return_value=()):
        assert se.SumoHighwayEnv._neighbor(obj,1,True) is None
    probes.append(dict(case='absent',returned=None))
    save('neighbor-probes.json',probes)
    # Shadow inference is not a rollout and cannot establish a crash reduction.
    from utils import av_hv_release_check as a
    import numpy as np
    modeldir=a.r.sources('baseline',22)
    manifest=json.loads((modeldir/'manifest.json').read_text())
    a.b.torch.set_num_threads(1)
    agent=a.b.PPO(obs_dim=47,act_dim=2,**manifest['hp'])
    agent.load(str(modeldir/'policy-600064.pt'))
    shadow=[]
    for p in sorted((OUT/'av/runs').glob('*/detailed.json')):
        x=json.loads(p.read_text())['last_30_seconds'][-1]
        obs=np.array(x['observation'],dtype=np.float32);new=obs.copy();changed=[]
        for mode,pairs in x['raw_neighbors'].items():
            if not pairs:continue
            vid,gap=min(pairs,key=lambda pair:pair[1])
            partner=next((v for v in x['neighbors'] if v['id']==vid),None)
            if gap>=0 or partner is None:continue
            direction=1 if int(mode)%2==0 else -1
            idx=1+4*[2,1,0,-1,-2].index(direction)+(0 if int(mode)>=2 else 2)
            new[idx]=0.;new[idx+1]=np.clip((partner['speed']-x['speed'])/27.78,-1,1)
            changed.append(vid)
        if changed:
            shadow.append(dict(case=p.parent.name,original_action=agent.predict(obs,deterministic=True).tolist(),clamped_gap_action=agent.predict(new,deterministic=True).tolist(),changed_vehicle_ids=changed))
    save('shadow-observation.json',dict(note='Offline single-state inference only, no rollout or retraining. Clamp negative gap to zero and restore relative speed.',rows=shadow))
    print(json.dumps(aggregates,indent=2))
    print('Collision partner negative-gap discarded:',sum(r['outcome']=='collision' and r['partner_negative_gap_discard_count']>0 for r in rows))
    print('Current leader collisions:',sum(r['collision_with_current_leader'] for r in rows))

if __name__=='__main__':main()
