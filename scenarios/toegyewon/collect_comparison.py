"""Run a matched four-controller study without modifying existing datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from comparison_setup import MODELS
from compare_controllers import report

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1000);args=p.parse_args()
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    root=HERE/'results'/f'controller_comparison_{stamp}';root.mkdir()
    old={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'datasets').rglob('dataset.npz')}
    protocol=dict(seed=args.seed,controllers=list(MODELS),episodes_per_controller=30,old_dataset_sha256=old,
                  source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')},complete=False)
    (root/'study.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(root,flush=True)
    def run(model):
        with (root/f'{model}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(HERE/'collect.py'),'--rules','--paired','--repeats','5',
                '--seed',str(args.seed),'--ego-controller',model,'--out',str(root/model)],stdout=log,stderr=subprocess.STDOUT)
        print(f'{model}: exit {result.returncode}',flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,MODELS))
    if any(codes):raise RuntimeError(f'Collection failed; inspect logs in {root}')
    report(root)
    assert all(hashlib.sha256((HERE/f).read_bytes()).hexdigest()==digest for f,digest in old.items())
    protocol['complete']=True
    (root/'study.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(f'COMPLETE {root}',flush=True)

if __name__=='__main__':main()
