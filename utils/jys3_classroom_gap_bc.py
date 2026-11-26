"""Run the unmodified classroom BC entry point on the two physical-fork datasets."""
import argparse
import csv
import hashlib
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from algorithms.bc import BCPolicy
from utils import jys3_gap_bc_eval as evaluation
from utils.jys3_gap_bc import ROOT as PREVIOUS, DENSE, SETS, EVAL_BASE, prior

ROOT = PREVIOUS.parent / 'jys3_classroom_gap_bc'


def train(name):
    source = SETS[name]
    rows = sorted(prior.v3.read(source/'train.json'), key=lambda r:(r['route'], r['rep']%2, r['rep']))
    assert len({r['seed'] for r in rows}) == len(rows) == 160
    assert Counter(r['route'] for r in rows) == {i:10 for i in range(1,17)}
    panel = {EVAL_BASE + rep*100 + route for rep in range(8) for route in range(1,17)}
    assert panel.isdisjoint(r['seed'] for r in rows)
    x, y, episodes, manifest = [], [], [], []
    for i,row in enumerate(rows):
        file = source/'samples'/row['file']
        assert row['outcome']=='arrival' and hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
        with np.load(file, allow_pickle=False) as sample:
            assert sample['obs'].shape==(row['steps'],50) and sample['actions'].shape==(row['steps'],2)
            x.append(sample['obs'])
            y.append(sample['actions'])
        episodes.append(np.full(row['steps'], i, dtype=np.int32))
        manifest.append(dict(episode=i, route=row['route'], lane=row['lane'], seed=row['seed'], steps=row['steps']))
    inputs = ROOT/'inputs'
    inputs.mkdir(parents=True,exist_ok=True)
    file = inputs/f'{name}.npz'
    np.savez_compressed(file, observation=np.concatenate(x), action_raw=np.concatenate(y), episode=np.concatenate(episodes))
    prior.v3.save(inputs/f'{name}.json',manifest)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    with (ROOT/f'{name}_train.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'train_bc.py','--data',str(file),'--out-dir',str(ROOT/name)],
                       cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    config = prior.v3.read(ROOT/name/'config.json')
    assert (config['epochs'],config['batch_size'],config['lr'],config['val_fraction'],config['seed'])==(50,256,.0003,.2,0)
    assert len(config['train_episodes'])==128 and len(config['val_episodes'])==32


def load_model(folder,name):
    return BCPolicy.load(folder/name/'model.pt')


def audit():
    """Check the saved model against its own best epoch and common expert labels."""
    torch.set_num_threads(2)
    common = prior.v3.read(DENSE/'valid.json')
    inputs, targets = prior.dagger.load_rollouts(common, DENSE/'valid_samples')
    assert len(common)==32 and len(inputs)==11555
    summary = {}
    for name in SETS:
        cfg = prior.v3.read(ROOT/name/'config.json')
        manifest = prior.v3.read(ROOT/'inputs'/f'{name}.json')
        assert {r['seed'] for r in common}.isdisjoint(r['seed'] for r in manifest)
        with (ROOT/name/'training_log.csv').open(encoding='utf8', newline='') as stream:
            history = list(csv.DictReader(stream))
        assert len(history)==50
        best = min(history,key=lambda r:float(r['val_loss']))
        model = load_model(ROOT,name)
        with np.load(ROOT/'inputs'/f'{name}.npz') as data, torch.no_grad():
            val = np.isin(data['episode'],cfg['val_episodes'])
            train_changes=int(np.count_nonzero(data['action_raw'][~val,1]))
            pred = model(torch.from_numpy(data['observation'][val]))
            mse = torch.nn.functional.mse_loss(pred,torch.from_numpy(data['action_raw'][val])).item()
            assert abs(mse - float(best['val_loss'])) < 1e-6
            common_pred = model(inputs)
        raw = common_pred[:,1]
        cmd = (raw>=.5).int() - (raw<=-.5).int()
        changed = targets[:,1]!=0
        summary[name] = dict(best_epoch=int(best['epoch']), validation_mse=mse,
            train_steps=cfg['train_samples'],validation_steps=cfg['val_samples'],
            train_change_labels=train_changes,
            common_change_labels=int(changed.sum()), common_correct_changes=int((cmd[changed]==targets[changed,1]).sum()),
            common_spurious_changes=int((cmd[~changed]!=0).sum()),
            common_accel_mae=float((common_pred[:,0]-targets[:,0]).abs().mean()),
            checkpoint_sha256=hashlib.sha256((ROOT/name/'model.pt').read_bytes()).hexdigest())
    prior.v3.save(ROOT/'audit.json',summary)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['train','evaluate','audit'])
    phase=parser.parse_args().phase
    if phase=='train':
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(train,SETS))
        pairs = [[(r['route'],r['lane']) for r in prior.v3.read(ROOT/'inputs'/f'{n}.json')] for n in SETS]
        assert pairs[0]==pairs[1]
    elif phase=='audit':
        audit()
    else:
        evaluation.main(ROOT,load_model)
