"""Compare 50 and 100 classroom epochs using identical prebuilt data and split."""
import csv
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from utils import jys3_classroom_gap_bc as course

ROOT = course.ROOT.parent / 'jys3_classroom_gap_bc_100'
read = course.prior.v3.read
save = course.prior.v3.save


def fit(name):
    ROOT.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    data = course.ROOT/'inputs'/f'{name}.npz'
    with (ROOT/f'{name}_train.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'train_bc.py','--data',str(data),'--out-dir',str(ROOT/name),
                        '--epochs','100'],cwd=Path(__file__).resolve().parents[1],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    original = course.ROOT/name
    current = ROOT/name
    assert read(current/'config.json') == dict(read(original/'config.json'), epochs=100)
    logs = [p.joinpath('training_log.csv').read_text().splitlines() for p in (original,current)]
    assert len(logs[1])==101 and logs[1][:51] == logs[0], 'first 50 epochs differ'


def audit():
    torch.set_num_threads(2)
    rows=read(course.DENSE/'valid.json')
    x,y=course.prior.dagger.load_rollouts(rows,course.DENSE/'valid_samples')
    changed=y[:,1]!=0
    summary={}
    for name in course.SETS:
        summary[name]={}
        starts=[]
        for budget,folder in [(50,course.ROOT),(100,ROOT)]:
            cfg=read(folder/name/'config.json')
            with (folder/name/'training_log.csv').open(newline='') as stream:
                best=min(csv.DictReader(stream),key=lambda row:float(row['val_loss']))
            model=course.load_model(folder,name)
            with torch.no_grad(), np.load(course.ROOT/'inputs'/f'{name}.npz') as inp:
                valid=np.isin(inp['episode'],cfg['val_episodes'])
                pred=model(torch.from_numpy(inp['observation'][valid]))
                target=torch.from_numpy(inp['action_raw'][valid])
                mse=float((pred-target).square().mean())
                assert abs(mse-float(best['val_loss']))<1e-6
                raw=model(x)[:,1]
            thresholds={}
            for threshold in (.5,.3):
                cmd=(raw>=threshold).int()-(raw<=-threshold).int()
                thresholds[str(threshold)]=dict(correct=int((cmd[changed]==y[changed,1]).sum()),
                    spurious=int((cmd[~changed]!=0).sum()))
            scores=read(folder/'evaluation.json')[name]
            episodes=read(folder/f'eval_{name}.json')
            starts.append([(r['seed'],r['start'],r['obs_sha256']) for r in episodes])
            summary[name][budget]=dict(epoch=int(best['epoch']),mse=mse,common_changed=int(changed.sum()),
                common_keep=int((~changed).sum()),thresholds=thresholds, correct_sign=int((raw[changed]*y[changed,1]>0).sum()),
                outcomes=scores['outcomes'],first_gap=scores['common_first_gap']['after_wait'],
                delay=scores['after_wait'],missed=scores['all']['missed_to_end'])
        assert len(starts[0])==128 and starts[0]==starts[1]
    save(ROOT/'comparison.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(fit,course.SETS))
    course.evaluation.main(ROOT,course.load_model)
    audit()
