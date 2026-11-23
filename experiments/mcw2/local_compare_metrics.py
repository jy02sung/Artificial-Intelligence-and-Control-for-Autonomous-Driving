"""Verify every split and aggregate equal-weight route statistics for local review."""
import csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
D=Path(__file__).resolve().parents[2]/"results/local_improvement"
X=D.parent/"experiment_X"
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def metrics(rows,ref=None):
    counts=Counter(x["outcome"] for x in rows)
    routes={}
    for r in sorted(set(x["route"] for x in rows)):
        routes[r]=dict(Counter(x["outcome"] for x in rows if x["route"]==r))
    n=len(rows);k=n//16
    assert len(routes)==16 and all(sum(c.values())==k for c in routes.values())
    ci={}
    for outcome in ["collision","arrival"]:
        risks=np.asarray([v.get(outcome,0)/k for v in routes.values()])
        sim=np.random.default_rng(66712).binomial(k,risks,(100000,16)).sum(axis=1)*100/n
        ci[outcome]=np.quantile(sim,[.025,.975]).round(3).tolist()
    out=dict(n=n,counts=counts,routes=routes,stratified_95pct=ci)
    if ref:
        assert [(v["route"],int(v["seed"])) for v in rows]==[(v["route"],int(v["seed"])) for v in ref]
        out["paired"]=Counter(a["outcome"]+"_to_"+b["outcome"] for a,b in zip(ref,rows))
    return out
def main():
    reference=read_csv(X/"baseline.csv");devseeds={int(v["seed"]) for v in reference}
    previous=json.loads((D.parent/"local_holdout_1600/plan.json").read_bytes())
    prior={s for _,s in previous["fresh"]}
    orig=json.loads((X/"protocol.json").read_bytes())
    train={s for _,s in json.loads((X/"training.json").read_bytes())["episodes_started"]}
    train.update(s for _,s in json.loads((D/"Y_training.json").read_bytes())["episodes"])
    train.update(s for _,s,_ in json.loads((D/"Z_seeds.json").read_bytes()))
    for fn in ("W_search","T_search","V_screen","V_search","U_search"):
        if (D/(fn+".json")).exists():train.update(v["seed"] for v in json.loads((D/(fn+".json")).read_bytes())["rows"])
    if (D/"R_train_seeds.json").exists():train.update(s for _,s,_ in json.loads((D/"R_train_seeds.json").read_bytes()))
    summary={"mcw2_dev":metrics(reference),"X_dev":metrics(read_csv(X/"candidate.csv"),reference)}
    assert not devseeds.intersection(train|prior)
    unseen=set()
    for path in sorted(list(D.glob("*_dev.json"))+list(D.glob("*_final.json"))):
        label=path.stem;model,stage=label.rsplit("_",1);data=json.loads(path.read_bytes());rows=data["rows"]
        assert data["sha256"]==digest(D/(model+".pt"))
        if stage=="dev":
            assert len(rows)==400
            summary[label]=metrics(rows,reference)
        else:
            ids={v["seed"] for v in rows}
            pairs=[(v["route"],v["seed"]) for v in rows]
            if model=="W":expected=[tuple(t) for t in orig["seeds"]["final"]]
            else:expected=[(r,s) for r,s,z in json.loads((D/(model+"_confirm_plan.json")).read_bytes())["tasks"]]
            assert len(rows)==len(ids)==1600 and pairs==expected
            assert not ids.intersection(prior|devseeds|train|unseen)
            unseen.update(ids)
            summary[label]=metrics(rows)
        summary[label]["sha256"]=data["sha256"]
    (D/"comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    for name,row in summary.items():print(name,row["n"],dict(row["counts"]),row["stratified_95pct"])
if __name__=="__main__":main()
