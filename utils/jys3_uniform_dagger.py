"""R1-R5 without targeted routes or any state oversampling; match the 144-drive budget."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from utils import jys3_final_relu_dagger as ex

ROOT=ex.ROOT.parent/"jys3_uniform_dagger"
read,save=ex.p.read,ex.p.save


def paired(rows, reference):
    def ids(items):
        return [(r["seed"],r["route"],r["lane"],r["initial_sha256"]) for r in items]
    assert ids(rows)==ids(reference)
    return dict(Counter(a["outcome"]+" -> "+b["outcome"] for a,b in zip(reference,rows)))


def main():
    ROOT.mkdir(exist_ok=True)
    ex.torch.set_num_threads(2)
    previous=ex.checkpoint(0)
    start_hash=hashlib.sha256(previous.read_bytes()).hexdigest()
    assert read(ex.ROOT/"fit0.json")["sha256"]==start_hash
    selected=read(ex.ROOT/"expert.json")["short_reserve"]
    seeds=set()
    def reserve(values):
        assert len(values)==len(set(values)) and seeds.isdisjoint(values)
        seeds.update(values)
    def data(name):
        rows=read(ex.ROOT/name/"episodes.json")
        reserve([r["seed"] for r in rows])
        return ex.dataset(rows,name)
    x,y=data("train")
    vx,vy=data("valid")
    rounds={}
    panels=(("hold64",81260100,4,"hold64_5"),("final256",91000000,16,"final256"),
            ("novel256",104320000,16,"novel256"))
    for base,n in [(95380000,8)]+[(b,n) for _,b,n,_ in panels]:
        reserve(ex.planned(base,n))
    result=dict(start_sha256=start_hash,rounds=rounds,panels={})
    with ProcessPoolExecutor(10) as pool:
        for k in range(1,6):
            base=78000000+100000*k
            reserve(ex.planned(base,9))
            folder=ROOT/f"R{k}"
            rows=ex.schedule(pool,folder,base,9,selected,previous,collect=True)
            assert len(rows)==144 and [r["seed"] for r in rows]==ex.planned(base,9)
            ref=read(ex.ROOT/f"R{k}"/"uniform"/"episodes.json")
            paired(rows[:128],ref)
            if k==1:
                assert all((r["steps"],r["outcome"])==(s["steps"],s["outcome"]) for r,s in zip(rows,ref))
            xx,yy=ex.dataset(rows,folder)
            x,y=ex.torch.cat((x,xx)),ex.torch.cat((y,yy))
            ex.fit(k,x,y,vx,vy,[],uniform_dir=ROOT,initial=previous)
            previous=ROOT/f"dagger{k}.pt"
            screen=ex.schedule(pool,ROOT/f"screen{k}",95380000,8,selected,previous)
            ref=read(ex.ROOT/f"screen{k}"/"episodes.json")
            rounds[k]=dict(uniform=ex.tally(screen),enhanced=ex.tally(ref),
                           transitions=paired(screen,ref),states=len(x))
            save(ROOT/"rounds.json",rounds)
        for name,base,n,reference in panels:
            rows=ex.schedule(pool,ROOT/name,base,n,selected,previous)
            ref=read(ex.ROOT/reference/"episodes.json")
            result["panels"][name]=dict(uniform=ex.tally(rows),enhanced=ex.tally(ref),
                                        transitions=paired(rows,ref))
    save(ROOT/"summary.json",result)


if __name__=="__main__":
    main()
