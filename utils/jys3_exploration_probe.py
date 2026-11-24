from utils.jys3_ppo_long import DIR,ARM,paired
from utils.jys3 import environment,torch
from utils.jys3_dagger_one import policy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile,json
def run(i):
 torch.set_num_threads(1)
 models={s:policy(paired.ck(ARM,s)) for s in (32768,131072,196608,300000)}
 with tempfile.TemporaryDirectory() as d:
  e=environment('high_real_ratio',11,Path(d))
  out=[]
  try:
   for sample in range(5):
    seed=32110016+100*i
    e.ego_depart_lane='0';o,_=e.reset(seed=seed,options={'route':'route_16'})
    torch.manual_seed(98200+10*i+sample)
    probes={};cross=False;first=None
    for _ in range(1000):
     edge,l,_,_,_,_,_,_,rem=e.snapshot
     if sample==0 and edge=='86198728' and l==0:
      for t in (300,100,10,1):
       if rem<=t and t not in probes:
        with torch.no_grad():
         probes[t]={s:m.policy(torch.from_numpy(o).unsqueeze(0))[2].softmax(-1)[0].tolist() for s,m in models.items()}
     act=models[300000].predict(o,deterministic=sample==0)
     o,_,done,cut,info=e.step(act)
     if e.snapshot:
      if edge=='86198728' and l==0 and e.snapshot[0]==edge and e.snapshot[1]>0 and first is None:first=rem
      if e.snapshot[0]=='86198725':cross=True
     if done or cut:break
    out.append(dict(seed=seed,sample=sample,probes=probes,left_m=first,passage=cross,outcome='arrival' if info['arrived'] else 'collision' if info['collided'] else 'timeout'))
  finally:e.close()
 return out
if __name__=='__main__':
 with ProcessPoolExecutor(4) as pool:r=[x for batch in pool.map(run,range(4)) for x in batch]
 (DIR/'exploration16.json').write_text(json.dumps(r,indent=2))
 from collections import Counter
 for group in (False,True):
  rows=[x for x in r if bool(x['sample'])==group];print(group,len(rows),sum(x['left_m'] is not None for x in rows),sum(x['passage'] for x in rows),Counter(x['outcome'] for x in rows))
 for x in r:
  if not x['sample']:
   print(x['seed'],[(t,[round(p*100,1) for p in z[300000]]) for t,z in x['probes'].items()])
