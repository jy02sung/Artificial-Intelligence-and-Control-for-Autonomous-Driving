"""Summarize verified E route-16 first-road training actions."""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
from utils.experiment_e_training_replay import AUDIT

def process(path):
    maximum=max(int(p.stem.split("_")[1]) for p in path.glob("policy_*.verified"))
    episodes={}
    for line in (path/"events.jsonl").read_text().splitlines():
        row=json.loads(line)
        kind,ep,tick,*v=row
        if tick>maximum:break
        if kind=="start":
            episodes[ep]=dict(start=tick,lane=v[0],commands=0,changed=0,from0=0,move01=0,active01=0,far01=0,paid01=0,first01=None,paid=0,paid_sum=0.,passed=False,arrived=False,ended=False,early_pass=0.)
            continue
        r=episodes[ep]
        if kind=="cmd":
            cmd,before,after,prep,dist=v
            if cmd==1 and before==0:r['from0']+=1
            if cmd==1 and before==0 and after==1:
                r['move01']+=1
                r['active01']+=int(dist<=400)
                r['far01']+=int(dist>400)
                r['paid01']+=int(prep>0)
                if r['first01'] is None:r['first01']=dist
            if cmd!=1:continue
            r['commands']+=1
            if after is not None and after>before:
                r['changed']+=1
                if prep>0:r['paid']+=1;r['paid_sum']+=prep
        elif kind=="pass":r.update(passed=True,early_pass=v[1])
        elif kind=="end":r.update(ended=True,arrived=v[1])
    return maximum,episodes
def run():
    rs=[];summary=defaultdict(Counter)
    for d in sorted(AUDIT.glob("E*_*")):
        limit,episodes=process(d)
        for ep,r in episodes.items():
            arm,seed=d.name.split("_")
            rs.append(dict(arm=arm,seed=seed,episode=ep,verified_to=limit,**r))
            if r['lane']!=0:continue
            for period in ('all',str(min(3,r['start']//50000)+1)):
                c=summary[arm,period]
                c.update({k:r[k] for k in ('commands','from0','move01','active01','far01','paid01','changed','paid','paid_sum','passed','arrived','ended')})
                c.update(dict(episodes=1,tried=int(r['commands']>0),moved=int(r['changed']>0),rewarded=int(r['paid']>0),paid_pass=int(r['paid']>0 and r['passed']),early=int(r['early_pass']>0),try0=int(r['from0']>0),move01ep=int(r['move01']>0),paid01pass=int(r['paid01']>0 and r['passed'])))
    for fname,rows in [('episodes.csv',rs),('summary.csv',[dict(arm=a,period=p,**v) for (a,p),v in summary.items()])]:
        if rows:
            with (AUDIT/fname).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for k,v in summary.items():
        if k[1]=='all':print(k,v)
if __name__=='__main__':run()
