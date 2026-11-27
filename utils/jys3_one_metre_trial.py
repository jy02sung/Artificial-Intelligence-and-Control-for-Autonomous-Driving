"""Same parent and rollout seeds, one additional cost for observed gaps <= 1 metre."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

from utils import jys3_safe_prep_continuation as previous

T,L,C,mid = previous.T,previous.L,previous.C,previous.mid
SOURCE = mid.SOURCE/"lambda98"/"s33"
ROOT = mid.DEST.parent/"jys3_one_metre_trial"
ARMS = ("control","one_metre")
MARKS = (112500,125000)
FRESH = (1870320000,36)


def near(m):
    return any(v is not None and v <= 1. for v in (m["front_gap"],m["rear_gap"]))


class CountMoves(T.Episodes):
    extra = 0.

    def extra_move_cost(self, measurement):
        if near(measurement):
            assert C.dangerous(measurement)
            self.stats["one_metre_moves"] += 1
            self.stats["one_metre_extra_cost"] += self.extra
            return self.extra
        return 0.


class ChargeMoves(CountMoves):
    extra = 1.5


def train(arm):
    path = ROOT/arm
    previous.prepare(SOURCE,path)
    with patch.object(T,"Episodes",ChargeMoves if arm=="one_metre" else CountMoves):
        T.train(("A",33,"A"), directory=path,marks=MARKS,collision_reward=-14.,gae_lambda=.98)


def evaluate(job):
    arm,step,panel,rep = job
    if panel=="fresh":
        C.old.PANELS[panel]=FRESH
    cp = SOURCE/"policy_100000.pt" if arm=="parent" else ROOT/arm/f"policy_{step}.pt"
    C.evaluate("M5-FT",33,step,panel,rep,directory=ROOT/arm,checkpoint=cp,lateral=True,move_probe=near)


def batch(pool, models, panels):
    jobs=[(a,s,h,i) for h,(_,n) in panels.items() for i in range(n) for a,s in models]
    list(pool.map(evaluate,jobs))
    result = {}
    groups=T.P.groups()
    for arm,step in models:
        rows=[]
        for h,(base,n) in panels.items():
            for i in range(n):
                part=L.read(ROOT/arm/"parts"/f"{step}_{h}_{i}.json")
                assert [r["seed"] for r in part]==[base+100*i+k for k in range(1,17)]
                ref=L.read(SOURCE/"parts"/f"100000_{h}_{i}.json") if h!="fresh" else L.read(
                    ROOT/models[0][0]/"parts"/f"{models[0][1]}_{h}_{i}.json")
                assert all(x["initial_sha256"]==y["initial_sha256"] for x,y in zip(part,ref,strict=True))
                if arm=="parent" and h!="fresh":
                    assert all((x["steps"],x["outcome"],x["cutin"])==(y["steps"],y["outcome"],y["cutin"]) for x,y in zip(part,ref))
                rows.extend(part)
        needed=[r for r in rows if groups[r["route"],r["lane"]]=="needs_change"]
        result[f"{arm}/{step}"]=dict(outcomes=Counter(r["outcome"] for r in rows),
              required=len(needed),required_arrived=sum(r["outcome"]=="arrival" for r in needed),
              moves=sum(r["moves"] for r in rows),near=sum(r["flagged_moves"] for r in rows),
              cutin=sum(r["cutin"] for r in rows),route14=Counter(r["outcome"] for r in rows if r["route"]==14))
    return result


def validate():
    assert T.ft.sha(SOURCE/"policy_100000.pt")==mid.ARMS["lambda98"]
    assert near(dict(front_gap=1.,rear_gap=None)) and not near(dict(front_gap=None,rear_gap=1.001))
    assert not near(dict(front_gap=None,rear_gap=None))
    audit=ROOT.parent/"jys3_gae98_cutin_replay"/"move_audit"
    old=[r for p in audit.glob("100000_*.json") for r in L.read(p)]
    assert len(old)==1070 and sum(near(r["view"]) for r in old)==77
    assert all(r["danger"] for r in old if near(r["view"]))


def main():
    validate()
    parent=("parent",100000)
    models=[(a,n) for a in ARMS for n in MARKS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        training=[pool.submit(train,a) for a in ARMS]
        screen={"screen":C.old.PANELS["screen"]}
        result={"screen":batch(pool,[parent],screen)}
        for f in training:
            f.result()
        result["screen"].update(batch(pool,models,screen))
        L.save(ROOT/"summary.json",result)
        print(result,flush=True)
        def rank(model):
            c=result["screen"]["%s/%d"%model]["outcomes"]
            return c.get("arrival",0),-c.get("collision",0),-model[1]
        selected=[]
        for arm in ARMS:
            eligible=[(arm,n) for n in MARKS if rank((arm,n))[0]>=116 and rank((arm,n))[1]>=-6]
            if eligible:
                selected.append(max(eligible,key=rank))
        if selected:
            holds=batch(pool,[parent,*selected],{h:C.old.PANELS[h] for h in L.HOLDS})
            result["holdout"]=holds
            promoted=[m for m in selected if holds["%s/%d"%m]["outcomes"].get("arrival",0)>=519
                      and holds["%s/%d"%m]["outcomes"].get("collision",0)<=28]
            if promoted:
                result["fresh"]=batch(pool,[parent,*promoted],{"fresh":FRESH})
            L.save(ROOT/"summary.json",result)
    validate()


if __name__=="__main__":
    main()
