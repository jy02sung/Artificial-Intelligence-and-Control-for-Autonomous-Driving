"""Frozen successful candidate: exact replay gate then unseen-seed validation."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['IC_RECOVERY_CONFIG']=str(ROOT/'results/traffic_recovery_auto_20261116/round-02/config.json')
from utils import run_traffic_recovery_auto as a
import concurrent.futures as cf
import collections
s=a.s;t=s.t
OUT=ROOT/'results/traffic_recovery_validation_20261116'
MODEL=ROOT/'results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000.pt'

def worker(seed):
    s.a.b.torch.set_num_threads(1)
    manifest=t.read(OUT/'manifest.json')
    for p,h in manifest['hashes'].items():assert s.a.b.digest(ROOT/p)==h,('changed',p)
    directory=OUT/'cases'/str(seed);directory.mkdir(parents=True,exist_ok=True)
    result=directory/'policy-200000-evaluation.json'
    if result.exists():return t.read(result)
    t.save(directory/'manifest.json',t.read(MODEL.parent/'manifest.json'))
    t.EVAL_SEEDS=(seed,)
    def make(c,evaluation=False,none=False):
        env=s.make_env(c,evaluation,none)
        if none:env.routes={}
        assert not env.lc_safety_enabled
        return env
    t.make_env=make
    rows=s.evaluate('route',MODEL,directory)
    assert len(rows)==16 and all(r['route_overrides']==0 for r in rows)
    return rows

def publish(rows,stage):
    count=collections.Counter(r['outcome'] for r in rows)
    t.save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),completed=len(rows),target=320,outcomes=dict(count)))
    t.save(OUT/'summary.json',rows)
    lines=['# 2차 경로 보상 후보 신규 seed 검증','',f'상태: {stage}. {len(rows)}/320회. 고정 모델 round-02/route/policy-200000.pt. seed69101~69120. 재학습·강제 보정 없음.','', '|완주|충돌|기타|','|---:|---:|---:|',f"|{count['arrival']}|{count['collision']}|{len(rows)-count['arrival']-count['collision']}|",'', '|경로|완주|충돌|기타|','|---|---:|---:|---:|']
    for route in sorted({r['route'] for r in rows}):
        g=[r for r in rows if r['route']==route];c=collections.Counter(r['outcome'] for r in g)
        lines.append(f"|{route}|{c['arrival']}|{c['collision']}|{len(g)-c['arrival']-c['collision']}|")
    (ROOT/'docs/experiments/ic-traffic-recovery-validation-20261116.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        files=[MODEL,MODEL.parent/'manifest.json',a.CONFIG,Path(__file__),*[ROOT/p for p in s.SOURCE_FILES],*(s.OUT/'inputs/reference').iterdir()]
        hashes={str(p.relative_to(ROOT)):s.a.b.digest(p) for p in files if p.is_file()}
        if (OUT/'manifest.json').exists():assert t.read(OUT/'manifest.json')['hashes']==hashes
        else:t.save(OUT/'manifest.json',dict(hashes=hashes,seeds=list(range(69101,69121)),model=str(MODEL)))
        t.save(OUT/'status.json',dict(stage='replay_gate',pid=os.getpid()))
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            replay=pool.submit(worker,59101).result()
            old=[r for r in t.read(MODEL.parent/'policy-200000-evaluation.json') if r['traffic']=='reference' and r['seed']==59101]
            assert {(r['route'],r['outcome'],r['seconds']) for r in replay}=={(r['route'],r['outcome'],r['seconds']) for r in old},'Replay mismatch'
            t.save(OUT/'replay-gate.json',dict(passed=True,cases=16))
            rows=[];publish(rows,'running')
            for future in cf.as_completed([pool.submit(worker,seed) for seed in range(69101,69121)]):
                rows.extend(future.result());publish(rows,'running')
        publish(rows,'complete')
    except Exception as exc:
        t.save(OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
