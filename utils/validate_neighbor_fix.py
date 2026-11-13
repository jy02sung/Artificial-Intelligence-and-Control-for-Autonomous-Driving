"""Paired frozen-policy replay after the common negative-gap detector fix."""
import concurrent.futures as cf
import json
import shutil
from utils import diagnose_meeting_av as d

OUT=d.a.ROOT/'results/neighbor_fix_20261114'
BEFORE=d.OUT

def run(spec):
    d.OUT=OUT
    return d.run(spec)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for mode in ('none','low','reference'):
        folder=OUT/'inputs'/mode;folder.mkdir(parents=True,exist_ok=True)
        for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):
            src=BEFORE/'inputs'/mode/name;dst=folder/name
            shutil.copy2(src,dst)
            assert d.a.b.digest(src)==d.a.b.digest(dst)
    cases=[(m,f'route_{i:02d}') for m in ('reference','none','low') for i in (1,2,3,7,9,10,13,14)]
    cases += [('none',f'route_{i:02d}') for i in (4,5,6,8,11,12,15,16)]
    d.a.r.save(OUT/'manifest.json',dict(cases=cases,policy='baseline-22/policy-600064.pt',seed=9101,forced_correction=False,training=False,detector_sha=d.a.b.digest(d.a.ROOT/'env/sumo_env.py'),code_sha=d.a.b.digest(d.a.ROOT/'utils/validate_neighbor_fix.py'),baseline=str(BEFORE),identical_inputs=True))
    with cf.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases))
    d.a.r.save(OUT/'summary.json',results)
    checked=0
    for mode,route in cases:
        detail=json.loads((OUT/'runs'/f'{mode}-{route}'/'detailed.json').read_text())
        for x in detail['last_30_seconds']:
            for m,pairs in x['raw_neighbors'].items():
                if not pairs:continue
                _,gap=min(pairs,key=lambda v:v[1])
                if gap>=0:continue
                side='1' if int(m)%2==0 else '-1';kind='front' if int(m)>=2 else 'rear'
                actual=x['filtered'][side][kind]
                assert actual is not None and actual[0]==0,(mode,route,x['t'],m,actual)
                idx=1+4*[2,1,0,-1,-2].index(int(side))+(0 if kind=='front' else 2)
                assert x['observation'][idx]==0,(mode,route,x['t'],m,x['observation'][idx])
                checked+=1
    d.a.r.save(OUT/'verification.json',dict(completed=len(results),negative_gap_channels_verified=checked,route_overrides=sum(r['route_overrides'] for r in results),teleports=sum(r['teleports'] for r in results)))
    print(json.dumps([{k:r[k] for k in ('mode','route','outcome','seconds')} for r in results],indent=2))
    print('Negative-gap policy channels verified:',checked)

if __name__=='__main__':main()
