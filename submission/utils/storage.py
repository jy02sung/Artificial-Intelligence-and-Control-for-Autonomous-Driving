"""Training samples, run metadata and atomic checkpoints."""
import json
import numpy as np
import torch
from utils.driving_env import OBS_DIM


def read(path):
    return json.loads(path.read_bytes())


def save(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


def atomic_torch(obj,path):
    temp=path.with_suffix(".tmp")
    torch.save(obj,temp)
    temp.replace(path)


def save_samples(folder,row,obs,actions):
    name=f"route_{row['route']:02d}_{row['seed']}.npz"
    with (folder/(name+'.tmp')).open('wb') as f:
        np.savez_compressed(f,obs=obs,actions=actions)
    (folder/(name+'.tmp')).replace(folder/name)
    row['file']=name


def load_rollouts(rows,folder):
    xs,ys=[],[]
    for r in rows:
        with np.load(folder/r['file']) as data:
            assert data['obs'].shape==(r['steps'],OBS_DIM) and data['actions'].shape==(r['steps'],2)
            assert np.isfinite(data['obs']).all() and np.isfinite(data['actions']).all()
            xs.append(torch.from_numpy(data['obs']));ys.append(torch.from_numpy(data['actions']))
    return torch.cat(xs),torch.cat(ys)
