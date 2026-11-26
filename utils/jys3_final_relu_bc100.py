"""Paired BC-only extension, with identical first 50 epochs and no holdout-based selection."""
from concurrent.futures import ProcessPoolExecutor
from utils import jys3_final_relu_dagger as experiment


def main():
    ex=experiment
    ex.torch.set_num_threads(2)
    read,save=ex.p.read,ex.p.save
    root=ex.ROOT
    train=ex.dataset(read(root/"train"/"episodes.json"),"train")
    valid=ex.dataset(read(root/"valid"/"episodes.json"),"valid")
    ex.fit(0,*train,*valid,[],extended_bc=True)
    original=read(root/"screen0"/"episodes.json")
    selected=read(root/"expert.json")["short_reserve"]
    rows={"bc50":original}
    with ProcessPoolExecutor(4) as pool:
        for key,file in (("best100","bc.pt"),("last100","last.pt")):
            rows[key]=ex.schedule(pool,"bc100/"+key,95380000,8,selected,root/"bc100"/file)
            assert [(r["seed"],r["initial_sha256"]) for r in rows[key]]==[(r["seed"],r["initial_sha256"]) for r in original]
    counts={key:ex.tally(group) for key,group in rows.items()}
    transitions={key:dict(ex.Counter(a["outcome"]+" -> "+b["outcome"] for a,b in zip(original,rows[key])))
                 for key in ("best100","last100")}
    save(root/"bc100"/"summary.json",dict(counts=counts,transitions=transitions))
    print(counts,flush=True)


if __name__=="__main__":
    main()
