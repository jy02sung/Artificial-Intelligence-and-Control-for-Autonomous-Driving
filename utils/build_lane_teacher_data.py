"""Audit lane demonstrations from completed training-only teacher episodes."""
import json
import numpy as np
from utils.curriculum_data import OUT,save,digest
from utils.probe_lane_teacher import CASES

def build():
    folder=OUT/'lane-teacher-candidates';records=[];summaries=[];sources={}
    for route,seed,_ in CASES:
        for mode in ('baseline','candidate'):
            p=folder/f'{route}-{seed}-{mode}.json'
            if not p.exists():raise RuntimeError(f'Incomplete teacher collection: {p.name}')
            x=json.loads(p.read_text(encoding='utf8'));sources[str(p.relative_to(OUT))]=digest(p)
            summaries.append(dict(route=route,seed=seed,mode=mode,outcome=x['outcome']))
            if mode!='candidate' or x['outcome']!='arrival':continue
            rows=x['rows']
            accepted_changes=[]
            for i,r in enumerate(rows):
                c=r['context'];a=r['action'];actual=r['after_edge']==c['edge'] and r['after_lane'] is not None and r['after_lane']!=c['lane']
                window=[z for z in rows[i:] if z['t']<=r['t']+5]
                margin=all((z['context']['ttc'] is None or z['context']['ttc']>1.5) and (z['context']['rear_ttc'] is None or z['context']['rear_ttc']>2) and z['context']['lane_risk']<.2 for z in window)
                if actual and c['needed']*a[1]>0 and c['lane_risk']<.1 and margin:
                    accepted_changes.append(i)
                    records.append(dict(obs=r['obs'],actions=a,category='safe_change',route=route,seed=seed,t=r['t'],source_file=p.name))
            # A wait label must precede a subsequently verified safe executed change.
            for i,r in enumerate(rows):
                if not any(i<j and rows[j]['t']-r['t']<=10 for j in accepted_changes):continue
                needed=int(np.sign(r['context']['needed']));risk=r['target_risks'].get(str(needed),0)
                window=[z for z in rows[i:] if z['t']<=r['t']+5]
                margin=all((z['context']['ttc'] is None or z['context']['ttc']>1.5) and (z['context']['rear_ttc'] is None or z['context']['rear_ttc']>2) and z['context']['lane_risk']<.2 for z in window)
                if needed and risk>.1 and r['action'][1]==0 and i%5==0 and margin:
                    records.append(dict(obs=r['obs'],actions=r['action'],category='safe_wait',route=route,seed=seed,t=r['t'],source_file=p.name))
    counts={cat:sum(r['category']==cat for r in records) for cat in ('safe_change','safe_wait')}
    routes=sorted({r['route'] for r in records if r['category']=='safe_change'})
    qualified=counts['safe_change']>=10 and counts['safe_wait']>=10 and len(routes)>=2
    report=dict(qualified=qualified,counts=counts,routes=routes,episodes=summaries,sources=sources,records=records,criteria='arrival, executed route-directed change, next5s TTC/rear margins; waiting linked to safe change within10s; minimum10each and2routes')
    save(OUT/'lane-teacher-audit.json',report)
    if qualified:
        target=OUT/'F2-data';target.mkdir(exist_ok=True)
        np.savez_compressed(target/'lane.npz',obs=np.asarray([r['obs'] for r in records],dtype=np.float32),actions=np.asarray([r['actions'] for r in records],dtype=np.float32),category=np.asarray([r['category'] for r in records]))
        save(target/'manifest.json',dict(audit_sha=digest(OUT/'lane-teacher-audit.json'),lane_sha=digest(target/'lane.npz'),counts=counts,routes=routes))
    print({k:v for k,v in report.items() if k not in ('sources','records','episodes')})

if __name__=='__main__':build()
