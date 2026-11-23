from . import local_driver_ppo as d
import sys,json
from collections import deque
def probe(nm):
    d.setup(d.model(nm));infile=d.read(d.D/(nm+'_screen.json'))['rows']
    for r in [next(r for r in infile if r['outcome']=='timeout' and r['route']==rt) for rt in ('route_13','route_14')]:
        obs,_=d.env.reset(seed=r['seed'],options={'route':r['route']})
        frames=[]
        while True:
            t=d.torch.from_numpy(obs).unsqueeze(0)
            with d.torch.no_grad(): mu,_,lg,_=d.agent.policy(t);j=lg.argmax().item()
            if d.x.audit.V.getLaneID('ego')=='170059994_3' and (len(frames)<12 or d.env.step_count>=2999):
                lg=lg[0]-lg[0,2].item();frames.append([d.env.step_count,round(float(obs[23]),2),round(float(obs[24]),2),round(float(obs[13])*120,1),round(float(-obs[15])*120,1),lg.tolist()])
            obs,_,dn,cut,_=d.env.step([float(mu),1-j])
            if dn or cut:break
        print(r['route'],frames[:3],frames[-1:],flush=True)
    d.env.close()
if __name__=='__main__':probe(sys.argv[1])
