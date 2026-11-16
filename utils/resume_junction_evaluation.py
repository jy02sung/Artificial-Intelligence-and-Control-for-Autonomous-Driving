"""Reuse completed early evaluations and prioritize the queued combined trial."""
import concurrent.futures as cf
import os
from utils import run_junction_conflict_trials as j
s=j.s;t=s.t

def work(condition):
    if condition=='combined':return s.train(condition)
    s.a.b.torch.set_num_threads(1)
    directory=s.OUT/'runs'/condition
    assert (directory/'training-complete.json').exists()
    assert s.a.b.digest(directory/'model.pt')==t.read(directory/'training-complete.json')['model_sha']
    manifest=t.read(s.OUT/'manifest.json')
    for f,h in manifest['source_hashes'].items():assert s.a.b.digest(s.ROOT/f)==h,('changed',f)
    for checkpoint in sorted(directory.glob('policy-*.pt'),key=lambda p:int(p.stem.split('-')[1]),reverse=True):
        dest=directory/(checkpoint.stem+'-evaluation.json')
        if dest.exists():continue
        matches=list((s.OUT/'early-evaluation').rglob(dest.name))
        matches=[p for p in matches if p.parent.name==condition]
        if matches:
            source=matches[0];rows=t.read(source)
            assert len(rows)==64 and len({(r['route'],r['seed'],r['traffic']) for r in rows})==64
            assert all(r['condition']==condition and r['checkpoint']==checkpoint.name and r['route_overrides']==0 for r in rows)
            assert t.read(source.parent/'manifest.json')==t.read(directory/'manifest.json')
            t.save(dest,rows);t.save(directory/(checkpoint.stem+'-evaluation-reuse.json'),dict(source=str(source),checkpoint_sha=s.a.b.digest(checkpoint)))
        else:s.evaluate(condition,checkpoint,directory)
    result=dict(condition=condition,steps=200000,evaluations=320,model_sha=s.a.b.digest(directory/'model.pt'))
    t.save(directory/'complete.json',result);t.save(directory/'progress.json',dict(stage='complete',**result));return result

def main():
    fd=os.open(s.OUT/'resume-evaluation.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        t.save(s.OUT/'status.json',dict(stage='running',pid=os.getpid(),note='training preserved; reusing completed immutable evaluations'))
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(work,c) for c in ('combined','control','observation','reward')]
            for f in cf.as_completed(futures):print(f.result(),flush=True);s.publish()
        t.save(s.OUT/'status.json',dict(stage='complete'))
    except Exception as exc:t.save(s.OUT/'status.json',dict(stage='error',error=repr(exc)));raise
    finally:(s.OUT/'resume-evaluation.lock').unlink(missing_ok=True)

if __name__=='__main__':main()
