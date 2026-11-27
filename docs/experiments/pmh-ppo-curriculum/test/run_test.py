import json, sys
from utils import pmh_jys3_dagger_multi as dm
from utils.pmh_ppo_curriculum import OUT
if __name__ == '__main__':
    out = OUT / 'test'; res = {}
    for s in sys.argv[1:]:
        seed, step = s.split(':'); ck = OUT / f'seed{seed}' / f'policy_{step}.pt'
        for base in (48010000, 30010000):
            rows = dm.run_batch(ck, False, 16, base, out, 8)
            res.setdefault(f'seed{seed}@{step}', {})[str(base)] = dict(counts=dm.counts(rows), rows=rows)
            (out / 'test.json').write_text(json.dumps(res, indent=1))
            print(s, base, dm.counts(rows), flush=True)
