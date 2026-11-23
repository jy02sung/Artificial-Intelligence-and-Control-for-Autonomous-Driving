"""Select without accessing any holdout scores. Reuse the conventional PPO holdout if it wins."""
import time
from . import local_driver_ppo as d
D=d.D
if __name__=='__main__':
    for family in ('saved','fresh','lane'):d.wait(D/(family+'_dev_best.json'))
    cands=[d.read(D/(family+'_dev_best.json')) for family in ('saved','fresh','lane')]
    winner=max(cands,key=lambda z:(z['counts'].get('arrival',0)>=380 and z['counts'].get('collision',0)<20,z['counts'].get('arrival',0),-z['counts'].get('collision',0)))
    nm=winner['name']
    d.x.write_json(D/'recommended_freeze.json',winner|dict(sha256=d.sha(d.model(nm)),eligible=cands))
    if nm.startswith('lane_'):
        d.evaluate(nm,'final')
    else:
        d.wait(D/(nm+'_final.json'))
    result=d.read(D/(nm+'_final.json'))
    assert result['sha256']==d.sha(d.model(nm))
    print(nm,result['counts'],flush=True)
