import json,tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from utils.jys3 import environment,torch
from utils.jys3_dagger_one import policy
from utils.jys3_ppo_compare import OUT
D=OUT/'ppo_only'
def run(row):
    torch.set_num_threads(1)
    a=policy(D/'policy_0.pt')
    result={'case':row['case'],'seed':row['seed'],'lane':row['lane'],'route':row['route']}
    with tempfile.TemporaryDirectory() as tmp:
        e=environment('high_real_ratio',11,Path(tmp))
        try:
            for mode in ('original','keep'):
                rt=f"route_{row['route']:02d}"
                e.ego_depart_lane=str(row['lane'])
                obs,_=e.reset(seed=row['seed'],options={'route':rt})
                assert list(row['start'])==[e.snapshot[0],e.snapshot[1],e.snapshot[7],e.snapshot[-1]]
                cmd=[0,0,0]; actual=0;lo=1.;hi=-1.;edges=[]
                for t in range(1,1001):
                    pre=e.snapshot
                    if not pre[0].startswith(':') and (not edges or edges[-1]!=pre[0]):edges.append(pre[0])
                    action=a.predict(obs,deterministic=True)
                    if mode=='keep':action[e.lane_change_index]=0
                    lo=min(lo,float(action[e.accel_index]));hi=max(hi,float(action[e.accel_index]))
                    obs,_,done,cut,info=e.step(action)
                    cmd[1+info['lane_change']]+=1
                    post=e.snapshot
                    actual+=bool(post and post[0]==pre[0] and not pre[0].startswith(':') and post[1]!=pre[1])
                    if done or cut:break
                result[mode]=dict(steps=t,arrived=info['arrived'],commands=cmd,actual=actual,accel=[lo,hi],edges=edges)
                if mode=='original':assert t==row['steps'] and info['arrived'] and edges==list(e.routes[rt])
        finally:e.close()
    return result
if __name__=='__main__':
    cases=[r for r in json.loads((D/'hold_0.json').read_bytes()) if r['outcome']=='arrival']
    with ProcessPoolExecutor(5) as pool:rows=list(pool.map(run,cases))
    (D/'initial_arrivals_audit.json').write_text(json.dumps(rows,indent=2))
    print([(r['route'],r['original']['commands'],r['original']['actual'],r['keep']['arrived']) for r in rows])
