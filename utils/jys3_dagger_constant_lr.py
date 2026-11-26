"""Keep the enhanced final DAgger student's Adam LR at 2e-4 through R5."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib

from utils import jys3_final_relu_dagger as ex
from utils.jys3_uniform_dagger import paired

ROOT = ex.ROOT.parent / "jys3_dagger_constant_lr"
PANELS = [("hold64_5",81260100,4),("final256",91000000,16),("novel256",104320000,16)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    rows = ex.p.read(folder/"episodes.json")
    return ex.dataset(rows, folder)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    ex.torch.set_num_threads(2)
    model = ex.checkpoint(2)
    protected = {k:digest(ex.checkpoint(k)) for k in range(6)}
    for k,h in protected.items():
        assert h==ex.p.read(ex.ROOT/f"fit{k}.json")["sha256"]
    expert = ex.p.read(ex.ROOT/"expert.json")["short_reserve"]
    x,y = load(ex.ROOT/"train")
    vx,vy = load(ex.ROOT/"valid")
    critical=[]
    screen=ex.p.read(ex.ROOT/"screen2/episodes.json")
    result=dict(start_sha256=protected[2],baseline_sha256=protected[5],rounds={},panels={})
    with ProcessPoolExecutor(4) as pool:
        for k in range(1,6):
            failures=Counter(r["route"] for r in screen if r["outcome"]!="arrival")
            weak=sorted(ex.ROUTES,key=lambda r:(-failures[r],r))[:4]
            for label,base,reps,routes in [("uniform",78000000,8,ex.ROUTES),("weak",79000000,4,weak)]:
                directory=(ex.ROOT if k<=3 else ROOT)/f"R{k}"/label
                if k<=3:
                    rows=ex.p.read(directory/"episodes.json")
                    if k==3:
                        assert [r["seed"] for r in rows]==ex.planned(base+100000*k,reps,routes)
                else:
                    rows=ex.schedule(pool,directory,base+100000*k,reps,expert,model,collect=True,routes=routes)
                assert len(rows)==reps*len(routes)
                offset=len(x)
                for r in rows:
                    end=offset+r["steps"]
                    if r["outcome"]!="arrival":
                        critical.extend(range(max(offset,end-15),end))
                    offset=end
                xx,yy=ex.dataset(rows,directory)
                x,y=ex.torch.cat((x,xx)),ex.torch.cat((y,yy))
                assert len(x)==offset
            if k<=2:
                continue
            if k==3:
                original=ex.p.read(ex.ROOT/"fit3.json")
                assert original["critical"]==len(critical) and len(x)==original["states"]
            ex.fit(k,x,y,vx,vy,critical,initial=model,output_dir=ROOT,learning_rate=.0002)
            model=ROOT/f"dagger{k}.pt"
            screen=ex.schedule(pool,ROOT/f"screen{k}",95380000,8,expert,model)
            ref=ex.p.read(ex.ROOT/f"screen{k}/episodes.json")
            result["rounds"][k]=dict(original=ex.tally(ref),fixed=ex.tally(screen),paired=paired(screen,ref),weak=weak)
            ex.p.save(ROOT/"rounds.json",result["rounds"])
        for name,base,reps in PANELS:
            rows=ex.schedule(pool,ROOT/name,base,reps,expert,model)
            ref=ex.p.read(ex.ROOT/name/"episodes.json")
            result["panels"][name]=dict(original=ex.tally(ref),fixed=ex.tally(rows),paired=paired(rows,ref))
    assert all(digest(ex.checkpoint(k))==h for k,h in protected.items())
    ex.p.save(ROOT/"summary.json",result)


if __name__=="__main__":
    main()
