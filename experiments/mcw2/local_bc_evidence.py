"""Describe slow, open-lane recovery examples in successful expert trajectories."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
D = Path('results/local_native_bc_ppo')
def coverage(kind):
    counts = Counter()
    for file in sorted((D / kind).glob('*.npz')):
        with np.load(file) as demo:
            if str(demo['outcome']) != 'arrival':
                continue
            obs = demo['obs']
            act = demo['actions'][:, 0]
            slow_open = (obs[:, 0] * 27.78 < 3.0) & (obs[:, 9] * 120 > 60)
            counts.update(episodes=1, rows=len(obs), slow_open=int(slow_open.sum()),
                          positive=int((act[slow_open] > 0).sum()),
                          episodes_with_slow_open=int(slow_open.any()))
    return counts
if __name__ == '__main__':
    assert (D / 'recovery_summary.json').exists()
    result = {kind:coverage(kind) for kind in ('demos','recovery_demos')}
    assert result['demos']['episodes'] == 768
    print(result)
    (D / 'coverage.json').write_text(json.dumps(result,indent=2),encoding='utf8')
