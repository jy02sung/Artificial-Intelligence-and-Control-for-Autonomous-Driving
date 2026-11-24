"""Cold-start DAgger with the same student-driven training and driving evaluation cases."""
import json
from collections import Counter
import torch
from utils.jys3 import HP, SCHEMA, OBS_DIM, m
from utils.jys3_dagger_one import batch, load_rollouts, supervised_fit, OUT as PRIOR

if __name__=='__main__':
    folder=PRIOR.parent/'jys3_dagger_cold'
    folder.mkdir(exist_ok=True)
    if any(p.name != "random_init.pt" for p in folder.iterdir()):
        raise FileExistsError(folder)
    initial=folder/'random_init.pt'
    agent=m.PPO(obs_dim=OBS_DIM,act_dim=2,buffer=m.RolloutBuffer(),seed=11,**HP)
    torch.save(dict(schema=SCHEMA,policy=agent.policy.state_dict()),initial)
    rows=batch(True,folder=folder,initial=initial)
    valid=batch(True,folder=folder,initial=initial,reps=2,base=28110000,name='validation.json')
    x,y=load_rollouts(rows,folder)
    vx,vy=load_rollouts(valid,folder)
    supervised_fit(x,y,vx,vy,initial,folder,len(x))
    result=batch(False,folder=folder,initial=initial)
    ref=json.loads((PRIOR/'evaluation.json').read_bytes())
    def keys(records):
        return [(r['seed'],r['lane'],r['route']) for r in records]
    assert keys(ref)==keys(result) and not set(keys(result)).intersection(keys(rows)+keys(valid))
    print(Counter(r['outcome'] for r in result))
