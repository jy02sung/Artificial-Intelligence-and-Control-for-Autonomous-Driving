"""Replay successful lane-sampling trials up to the first fork and log the last 0->1 move."""
import csv,itertools,json,tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from utils.experiment_e import OUT,agent,make

@torch.no_grad()
def run(job):
    arm,seed=job
    torch.set_num_threads(1)
    model=agent(seed)
    model.load(str(OUT/f"{arm}_{seed}/policy_200000.pt"))
    model.policy.eval()
    p=OUT/'training_replay'/f"{arm}_{seed}"/'matched_probe/results.json'
    rows=[]
    with tempfile.TemporaryDirectory() as folder:
        env=make('high_real_ratio',arm,seed,Path(folder))
        try:
            for case in json.loads(p.read_bytes()):
                if case['mode']!='sample_lane' or not case['passed']:continue
                sim,rep=case['sim'],case['rep']
                torch.manual_seed(40000000+sim+rep)
                env.ego_depart_lane='0'
                obs,_=env.reset(seed=sim,options={'route':'route_16'})
                stops=0
                moves=[]
                for tick in range(3000):
                    edge,lane,_,_,j,_,_,speed,rem=env.snapshot
                    if edge=='86198728' and lane==0 and rem<=1 and speed<=.1:stops+=1
                    mu,std,logits,_=model.policy(torch.as_tensor(obs)[None])
                    torch.normal(mu,std)
                    cmd=torch.distributions.Categorical(logits=logits).sample().item()
                    prev=obs
                    obs,_,done,cut,_=env.step([mu.item(),model.policy.LANE_CMDS[cmd]])
                    after=env.snapshot
                    if after and edge=='86198728' and lane==0 and after[0]==edge and after[1]==1:
                        probs=logits.softmax(-1)[0].tolist()
                        moves.append((rem,*probs))
                    if after and after[0]=='86198725':break
                    assert not done and not cut,(job,sim,rep)
                assert moves and after[0]=='86198725'
                dist,left,keep,right=moves[-1]
                rows.append(dict(arm=arm,seed=seed,sim=sim,rep=rep,first_distance=moves[0][0],last_distance=dist,stopped_steps=stops,left=left,keep=keep,right=right,left_first=left>=max(keep,right)))
        finally:env.close()
    return rows

if __name__=='__main__':
    with ProcessPoolExecutor(6) as pool:rows=sum(pool.map(run,itertools.product(('E0','E1','E2','E3'),(11,22,33))),[])
    assert len(rows)==130
    with (OUT/'training_replay/successful_recoveries.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    import statistics as st
    for a in ['all','E0','E1','E2','E3']:
        g=[r for r in rows if a=='all' or a==r['arm']]
        print(a,len(g),'stops',sum(r['stopped_steps']>0 for r in g),'100+',sum(r['last_distance']>=100 for r in g),'300+',sum(r['last_distance']>=300 for r in g),'top',sum(r['left_first'] for r in g),'medprob',[round(st.median(r[k] for r in g),4) for k in ('left','keep','right')], 'dist',round(st.median(r['last_distance'] for r in g),1))
