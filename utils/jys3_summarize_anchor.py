"""Summarize paired frozen-policy evaluations and behavior on fixed expert observations."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch

from utils import jys3_compare_imitation_collision as paired
from utils import jys3_physical_imitation_ppo_anchor as trial


def main():
    torch.set_num_threads(1)
    root=trial.ROOT
    results=[]
    original=paired.rows(root,0)
    older=paired.rows(trial.base.ROOT,0)
    old_to_new=paired.transitions(older,original)
    files=sorted((trial.base.imitation.ROOT/'expert'/'validation').glob('*.npz'))
    xs,ys=[],[]
    for file in files:
        with np.load(file) as sample:
            xs.append(sample['obs'])
            ys.append(sample['actions'])
    x=torch.from_numpy(np.concatenate(xs))
    y=torch.from_numpy(np.concatenate(ys))
    assert (len(x),int((y[:,0]>.1).sum()),int((y[:,1]==0).sum()))==(11548,2076,11519)
    for file in sorted(root.glob('eval_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
        step=int(file.stem.split('_')[-1])
        rows=paired.rows(root,step)
        change=paired.transitions(original,rows)
        policy=trial.model_at(step).policy
        with torch.no_grad():
            a=policy.det_action(x)
        results.append(dict(step=step,outcomes=Counter(r['outcome'] for r in rows),
                            baseline_transitions=change,
                            expert_accel_to_brake=int(((y[:,0]>.1)&(a[:,0]<-.1)).sum()),
                            expert_keep_to_move=int(((y[:,1]==0)&(a[:,1]!=0)).sum()),
                            expert_move_correct=int(((y[:,1]!=0)&(a[:,1]==y[:,1])).sum())))
    selection=paired.read(root/'selection.json')
    holdbase=paired.read(root/'holdout_0.json')
    used={r['seed'] for r in original}
    for f in ('ppo_episodes.jsonl','warm_episodes.jsonl'):
        used.update(json.loads(line)[2] for line in (root/f).read_text().splitlines())
    used.update(r['seed'] for r in paired.read(trial.base.imitation.data.ROOT/'train.json'))
    for folder in ('BC_DAgger/collection','expert/validation'):
        for f in (trial.base.imitation.ROOT/folder).glob('*.json'):
            used.update(r['seed'] for r in paired.read(f))
    hold_seeds={r['seed'] for r in holdbase}
    assert len(hold_seeds)==64 and not used & hold_seeds
    hold={}
    for step in sorted({0,*[selection[k] for k in ('best_ppo','last')]}):
        rows=paired.read(root/f'holdout_{step}.json')
        hold[step]=dict(outcomes=Counter(r['outcome'] for r in rows),
                        transitions=paired.transitions(holdbase,rows))
    data=dict(screen=results,holdout=hold,original_rule_change=old_to_new,selection=selection)
    trial.base.v3.save(root/'comparison.json',data)
    for row in results: print(row['step'],row['outcomes'],row['expert_accel_to_brake'])
    print('holdout',hold)


if __name__=='__main__':
    main()

