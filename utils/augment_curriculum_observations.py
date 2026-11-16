"""Replay preserved demonstrations exactly; append diagnostic merge inputs separately."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from utils.curriculum_data import OUT,ROOT,make_env,save,digest
from env.junction_lookahead_trials import JunctionLookaheadEnv

def run(shard=0,shards=1,limit=None):
    assert 0<=shard<shards
    torch.set_num_threads(1)
    out=OUT/'augmented-observations';out.mkdir(exist_ok=True)
    records=json.loads((OUT/'data-manifest.json').read_text(encoding='utf8'))['episodes']
    chosen=[x for x in records if x['outcome']=='arrival'][shard::shards]
    if limit is not None:chosen=chosen[:limit]
    env=make_env();env.planning_horizon=150
    source_hashes={f:digest(ROOT/f) for f in ['env/junction_lookahead_trials.py','env/junction_conflict_trials.py','utils/augment_curriculum_observations.py']}
    try:
        for item in chosen:
            name=f"{item['route']}-{item['seed']}"
            source=OUT/'data-episodes'/f'{name}.json';dest=out/f'{name}.json'
            signature=dict(original_sha=digest(source),feature_sources=source_hashes,planning_horizon=150,visibility=50)
            if dest.exists():
                existing=json.loads(dest.read_text(encoding='utf8'))
                assert existing['signature']==signature
                continue
            original=json.loads(source.read_text(encoding='utf8'))
            env.warmup_steps=item['warmup_steps'];obs,_=env.reset(seed=item['seed'],options={'route':item['route']});features=[]
            for index,row in enumerate(original['rows']):
                np.testing.assert_array_equal(obs,np.asarray(row['obs'],dtype=np.float32))
                assert env.visibility==50
                feature=JunctionLookaheadEnv.conflict_features(env)
                features.append(dict(t=row['t'],merge=list(feature)))
                obs,_,done,cut,info=env.step(np.asarray(row['action'],dtype=np.float32))
                if index<len(original['rows'])-1:assert not(done or cut)
            assert done and info['arrived'] and not info['collided'] and env.route_override_count==0
            save(dest,dict(signature=signature,route=item['route'],seed=item['seed'],warmup_steps=item['warmup_steps'],baseline_observation_exact=True,outcome='arrival',features=features))
            print(dict(episode=name,states=len(features),risk_states=sum(x['merge'][0]>0 for x in features),exact=True),flush=True)
    finally:env.close()
    save(out/f'shard-{shard}-complete.json',dict(shard=shard,shards=shards,episodes=len(chosen),limited=limit is not None,source_hashes=source_hashes))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int);a=p.parse_args();run(a.shard,a.shards,a.limit)
