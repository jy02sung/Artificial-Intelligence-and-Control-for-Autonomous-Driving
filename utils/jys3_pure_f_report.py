"""Summarize matched on-road trajectories and render measured speed and lane changes."""
from collections import Counter
import json
from pathlib import Path
from utils.jys3_pure_f import ROOT, ARMS, STEPS, save

def read(arm,seed,n,base):
    return [r for rep in range(4) for r in json.loads((ROOT/f"{arm}_{seed}"/f"eval_{base}_{n}_{rep}.json").read_bytes())]

def aggregate(rows):
    c=Counter(r["outcome"] for r in rows)
    relevant=[r for r in rows if r["route"]==16 and r["start"][1]==0]
    def ready(route,lanes):
        cases=[r for r in rows if r["route"]==route and r["prep"] is not None]
        return [sum(r["prep"] in lanes for r in cases),len(cases)]
    return dict(outcomes=dict(c),slow_s=round(sum(r["slow"] for r in rows)*.2/len(rows),2),
                slow_pct=round(100*sum(r["slow"] for r in rows)/sum(r["steps"] for r in rows),1),
                side_s=round(sum(r.get("slow_side_near",0) for r in rows)*.2/len(rows),2),
                moves=round(sum(r["moves"] for r in rows)/len(rows),2),
                prep13=ready(13,[0,1]),prep14=ready(14,[2]),
                exit13=sum(r["route"]==13 and "169728835" in r["forks"] for r in rows),
                exit14=sum(r["route"]==14 and "797789364" in r["forks"] for r in rows),
                sixteen=dict(n=len(relevant),command=sum(r["first_left"] is not None for r in relevant),
                             change=sum(r.get("actual_left") is not None for r in relevant),
                             main=sum("86198725" in r["forks"] for r in relevant),
                             finish=sum(r["outcome"]=="arrival" for r in relevant),
                             first_m=[r["first_left"] for r in relevant]))

def plot(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams["font.family"]="Malgun Gothic"
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex="col",layout="constrained")
    colors=["#353c48","#c45e14","#197b66","#415fbe"]
    for col,(route,case) in enumerate([(13,1),(16,0)]):
        for (arm,rows),color in zip(groups.items(),colors):
            r=next(x for x in rows if (x["route"],x["case"])==(route,case))
            track=r["track"];sec=[(s[0]-1)*.2 for s in track]
            axes[0,col].plot(sec,[s[4]*3.6 for s in track],label=arm,color=color,linewidth=1.7)
            axes[1,col].step(sec,[s[2] for s in track],where="post",color=color,linewidth=1.7)
        axes[0,col].set_title(f"{route}번 경로 · 출발 {r['start'][1]}차로")
        axes[1,col].set_xlabel("주행 시간 (초)")
        axes[1,col].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0,0].set_ylabel("실제 속도 (km/h)");axes[1,0].set_ylabel("현재 도로의 차로 번호")
    axes[0,0].legend(ncol=4,fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle("32,768스텝 모델의 동일 조건 주행 · 5초 간격 실측")
    fig.savefig(ROOT/"drives.png",dpi=150)
    plt.close(fig)

def main():
    summaries={}; groups={}; ref=None
    for n in STEPS:
        for arm in ARMS:
            rows=read(arm,11,n,34110000)
            keys=[(r["case"],r["route"],r["traffic_seed"],r["start"]) for r in rows]
            if ref is None:ref=keys
            assert len(rows)==64 and keys==ref
            summaries[f"{arm}_{n}"]=aggregate(rows)
            if n==STEPS[-1]:groups[arm]=rows
    save(ROOT/"summary.json",summaries)
    plot(groups)
if __name__=="__main__":main()
