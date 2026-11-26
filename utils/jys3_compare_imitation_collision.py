"""Compare paired driving outcomes and acceleration on fixed expert observations."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch

from utils import jys3_physical_imitation_ppo as run


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(root, step):
    panels = read(root/f'eval_{step}.json')
    assert set(panels) == set(run.imitation.PANELS)
    assert all(len(panel) == 64 for panel in panels.values())
    return [row for name in run.imitation.PANELS for row in panels[name]]


def transitions(before, after):
    changes = Counter()
    for a, b in zip(before, after, strict=True):
        assert all(a[key] == b[key] for key in ('start','route','seed','case'))
        changes[f"{a['outcome']} -> {b['outcome']}"] += 1
    return changes


def main():
    torch.set_num_threads(1)
    old = run.ROOT
    new = old.with_name(old.name+'_col5')
    baseline = rows(old,0)
    assert baseline == rows(new,0)
    files = sorted((run.imitation.ROOT/'expert'/'validation').glob('*.npz'))
    obs, actions = [], []
    for file in files:
        with np.load(file) as samples:
            obs.append(samples['obs'])
            actions.append(samples['actions'])
    x = torch.from_numpy(np.concatenate(obs))
    y = torch.from_numpy(np.concatenate(actions))
    accel, keep = y[:,0]>.1, y[:,1]==0
    assert (len(x), int(accel.sum()), int(keep.sum())) == (11548,2076,11519)
    result = []
    for step in sorted(int(p.stem.split('_')[-1]) for p in new.glob('eval_*.json')):
        for penalty,root in [(-7,old),(-5,new)]:
            if not (root/f'eval_{step}.json').exists():
                continue
            current = rows(root, step)
            counts = Counter(r['outcome'] for r in current)
            changes = transitions(baseline,current)
            model = run.imitation.dagger.policy(run.checkpoint(step,root)).policy
            with torch.no_grad():
                prediction = model.det_action(x)
            record = dict(step=step, collision_penalty=penalty, outcomes=counts,
                          baseline_transitions=changes,
                          expert_accel_to_brake=int((accel & (prediction[:,0]<-.1)).sum()),
                          expert_keep_to_move=int((keep & (prediction[:,1]!=0)).sum()),
                          route_arrivals={r:sum(x['outcome']=='arrival' and x['route']==r for x in current) for r in range(1,17)})
            if penalty==-5 and (old/f'eval_{step}.json').exists():
                record['col7_to_col5']=transitions(rows(old,step),current)
            result.append(record)
            print(step,penalty,dict(counts),changes['arrival -> arrival'],record['expert_accel_to_brake'],record['expert_keep_to_move'])
    run.v3.save(new/'comparison.json',result)


if __name__=='__main__':
    main()
