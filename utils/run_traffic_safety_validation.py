"""Frozen-policy validation on unseen traffic seeds; three CPU workers, no learning."""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
from utils import run_traffic_safety_trials as s

t=s.t
ROOT=t.ROOT
OUT=ROOT/'results/traffic_safety_validation_20261115'
OLD=s.OUT
CANDIDATES={
 'original':('baseline',ROOT/'results/route_following_trials_20261114/runs/baseline/policy-200000.pt'),
 'combo':('combo',OLD/'runs/combo/policy-200000.pt'),
 'lane':('lane',OLD/'runs/lane/policy-151552.pt'),
}
SEEDS=tuple(range(49101,49121))

def check_frozen():
    m=t.read(OUT/'manifest.json')
    for p,h in m['hashes'].items():
        assert s.a.b.digest(ROOT/p)==h,('Frozen input changed',p)

def worker(name,seed,smoke=False):
    check_frozen()
    s.a.b.torch.set_num_threads(1)
    condition,checkpoint=CANDIDATES[name]
    directory=OUT/('smoke' if smoke else 'cases')/name/str(seed)
    result=directory/(checkpoint.stem+'-evaluation.json')
    if result.exists():return str(result)
    directory.mkdir(parents=True,exist_ok=True)
    t.save(directory/'manifest.json',t.read(checkpoint.parent/'manifest.json'))
    t.OUT=OUT;t.EVAL_SEEDS=(seed,)
    def environment(c,evaluation=False,none=False):
        env=s.make_env(c,evaluation=evaluation,none=none)
        # Reuse the exact previous evaluator but omit duplicate no-traffic cases.
        if none:env.routes={}
        elif smoke:env.routes={k:v for k,v in env.routes.items() if k=='route_01'}
        assert not env.lc_safety_enabled
        return env
    t.make_env=environment
    rows=s.evaluate(condition,checkpoint,directory,False)
    assert len(rows)==(1 if smoke else 16)
    assert all(r['traffic']=='reference' and r['seed']==seed for r in rows)
    assert all(r.get('route_overrides',0)==0 for r in rows)
    t.save(directory/'progress.json',dict(stage='complete',completed=len(rows),target=len(rows)))
    check_frozen()
    return str(result)

def publish():
    rows=[]
    lines=['# 고정 정책 신규 교통 seed 검증','', '재학습·강제 보정 없음. seed 49101~49120, 16개 경로, 정책별 320회. 미완료 결과는 중간 집계다.','', '|정책|평가 수|완주|충돌|기타|','|---|---:|---:|---:|---:|']
    for name in CANDIDATES:
        group=[]
        for p in (OUT/'cases'/name).glob('*/*-evaluation.json'):group.extend(t.read(p))
        rows.extend(dict(r,candidate=name) for r in group)
        a=sum(r['outcome']=='arrival' for r in group);c=sum(r['outcome']=='collision' for r in group)
        lines.append(f'|{name}|{len(group)}/320|{a}|{c}|{len(group)-a-c}|')
    t.save(OUT/'summary.json',rows)
    (ROOT/'docs/experiments/ic-traffic-safety-validation-results-20261115.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return len(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        assert shutil.disk_usage(OUT).free>20*1024**3
        if not (OUT/'manifest.json').exists():
            files=list(t.SOURCE_FILES)+['env/route_learning_trials.py','env/route_following_trials.py','utils/run_traffic_safety_validation.py']
            for mode in ('none','reference'):
                dest=OUT/'inputs'/mode;dest.mkdir(parents=True,exist_ok=True)
                for name in ('network.net.xml','highway.rou.xml','run.sumocfg'):
                    shutil.copy2(OLD/'inputs'/mode/name,dest/name);files.append(str((dest/name).relative_to(ROOT)))
            for _,p in CANDIDATES.values():files.extend([str(p.relative_to(ROOT)),str((p.parent/'manifest.json').relative_to(ROOT))])
            t.save(OUT/'manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes={p:s.a.b.digest(ROOT/p) for p in files},seeds=SEEDS,workers=3,candidates={n:str(p) for n,(_,p) in CANDIDATES.items()}))
        check_frozen()
        if not args.smoke:
            assert all(list((OUT/'smoke'/n).glob('*/*-evaluation.json')) for n in CANDIDATES),'Smoke gate required'
        jobs=[(name,seed) for seed in (SEEDS[:1] if args.smoke else SEEDS) for name in CANDIDATES]
        t.save(OUT/'status.json',dict(stage='smoke' if args.smoke else 'running',pid=os.getpid(),target=3 if args.smoke else 960))
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,n,seed,args.smoke) for n,seed in jobs]
            for future in cf.as_completed(futures):
                print(future.result(),flush=True)
                if not args.smoke:t.save(OUT/'status.json',dict(stage='running',pid=os.getpid(),completed=publish(),target=960))
        t.save(OUT/'status.json',dict(stage='smoke_complete' if args.smoke else 'complete',completed=3 if args.smoke else publish()))
    except Exception as exc:
        t.save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
