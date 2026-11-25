"""Summarize paired held-out fork trials and verify their starting conditions."""
import json
from collections import Counter
from utils.route120_short import OUT,SEEDS,old,save

def main():
    scores={}
    for steps in (0,5000,20000):
        ref=[]
        for arm in ('original','route120'):
            rows=[]
            pair=0
            for seed in SEEDS:
                sample=json.loads((OUT/f'{arm}_{steps}_{seed}.json').read_bytes())
                for lane in (0,1):
                    forks=[x for x in sample if x['route'] in (13,14) and x['lane']==lane]
                    assert len(forks)==2 and forks[0]['start']==forks[1]['start']
                    pair+=all(x['passed'] for x in forks)
                rows+=sample
            starts=[x['start'] for x in rows]
            if ref: assert ref==starts
            ref=starts
            byroute={}
            for n in (13,14,16):
                group=[x for x in rows if x['route']==n]
                changes=[next((d for l,k,d in x['changes'] if l==0 and k>0),None) for x in group]
                byroute[n]=dict(total=len(group),reached=sum(x['reached'] for x in group),passed=sum(x['passed'] for x in group),outcomes=dict(Counter(x['outcome'] for x in group)))
                if n==16:
                    byroute[n].update(first_actual_left_m=changes,first_left_command_m=[x['first_left'] for x in group])
            scores[f'{arm}_{steps}']=dict(pair=pair,routes=byroute)
    save(OUT/'summary.json',scores)
    print(json.dumps(scores))
if __name__=='__main__': main()
