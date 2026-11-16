"""Freeze C-only supervision; preserve the original A/B archives."""
import json
import numpy as np
from utils.curriculum_data import OUT,save,digest

def build():
    folder=OUT/'C-data';folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        for f,h in m['sources'].items():assert digest(OUT/f)==h
        for n in ['keep','safe']:assert digest(folder/f'{n}.npz')==m[n+'_sha']
        print(m);return
    sources={};lookup={}
    for p in (OUT/'augmented-observations').glob('route_*.json'):
        x=json.loads(p.read_text(encoding='utf8'));assert x['baseline_observation_exact']
        sources[str(p.relative_to(OUT))]=digest(p)
        lookup[x['route'],x['seed']]={round(r['t'],3):r['merge'] for r in x['features']}
    datasets={}
    for name in ['keep','safe']:
        path=OUT/f'{name}-provenance.json';sources[path.name]=digest(path)
        records=json.loads(path.read_text(encoding='utf8'))['records'];data=[]
        for r in records:
            row=dict(r);row['obs']=list(r['obs']);row['obs'][60:63]=lookup[r['route'],r['seed']][round(r['t'],3)]
            if name=='safe':row['category']='baseline_yield'
            data.append(row)
        datasets[name]=data
    path=OUT/'teacher-candidate-audit.json';sources[path.name]=digest(path)
    audit=json.loads(path.read_text(encoding='utf8'))
    for r in audit['records']:
        row=dict(r);row['obs']=list(r['obs']);row['obs'][60:63]=r['merge'];row['category']='validated_candidate'
        datasets['safe'].append(row)
        p=OUT/'teacher-candidates'/r['source_file'];sources[str(p.relative_to(OUT))]=digest(p)
    seeds=sorted({r['seed'] for data in datasets.values() for r in data})
    assert not set(seeds)&set([79101,79102,79103,*range(991001,991021)])
    for name,data in datasets.items():
        np.savez_compressed(folder/f'{name}.npz',obs=np.asarray([x['obs'] for x in data],dtype=np.float32),actions=np.asarray([x['action'] for x in data],dtype=np.float32),category=np.asarray([x['category'] for x in data]))
        save(folder/f'{name}-provenance.json',data)
    m=dict(sources=sources,keep=len(datasets['keep']),safe=len(datasets['safe']),candidate_labels=len(audit['records']),candidate_early_labels=audit['early_count'],training_seeds=seeds,supervision='acceleration-only safe loss; reference-distribution keep loss',columns=[60,61,62],visibility=50,road_search=150,keep_sha=digest(folder/'keep.npz'),safe_sha=digest(folder/'safe.npz'))
    save(folder/'manifest.json',m);print({k:v for k,v in m.items() if k!='sources'})

if __name__=='__main__':build()
