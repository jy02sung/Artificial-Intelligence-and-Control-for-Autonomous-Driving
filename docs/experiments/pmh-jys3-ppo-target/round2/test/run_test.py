import json, sys
from pathlib import Path
from utils import pmh_jys3_dagger_multi as dm
from utils.pmh_jys3_ppo_target import TEST_BASE, START, OUT
if __name__ == '__main__':
    R = OUT / 'round2'
    C = {'start': START}
    for s in sys.argv[1:]:
        run, step = s.split(':'); C[f'{run}@{step}'] = R / run / f'policy_{step}.pt'
    out = R / 'test'; res = {}
    for name, ck in C.items():
        for base in (TEST_BASE, 30010000):
            rows = dm.run_batch(ck, False, 16, base, out, 8)
            res.setdefault(name, {})[str(base)] = dict(counts=dm.counts(rows), rows=rows)
            (out / 'test.json').write_text(json.dumps(res, indent=1))
            print(name, base, dm.counts(rows), flush=True)
