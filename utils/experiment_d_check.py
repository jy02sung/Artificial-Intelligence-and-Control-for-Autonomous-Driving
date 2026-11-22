"""Fixed SUMO reward replays and preregistration."""
import hashlib,json
import numpy as np
from utils.experiment_b import agent_for,digest,m,save,STAGES
from utils.experiment_c_check import place
from utils.experiment_c_env import OUT as C,SEEDS
from utils.experiment_d_env import OUT,environment

SELECT=(850000,850100)
HOLD=(960000,960100,960200,960300)
def check():
    results=[]
    for arm in ('D0','D1'):
        env=environment('none',arm,11,OUT/'check_maps',audit=True)
        try:
            for route,road,correct,wrong,initial,departure,back in [
                ('route_13','86198728',0,1,0,480,370),('route_13','86198728',0,1,0,200,100),
                ('route_13','86198728',0,1,1,480,370),('route_16','86198728',1,0,1,480,370),
                ('route_04','1063527748',2,1,2,200,1),('route_14','1063527748',2,1,2,200,1)]:
                place(env,route,road,initial,departure)
                for t in range(250):
                    edge,l,*_,rem=env.snapshot
                    cmd=wrong-correct if t==0 and initial==correct else correct-wrong if l==wrong and rem<=back else 0
                    _,_,done,cut,info=env.step(np.array([.5,cmd],np.float32))
                    if cmd==correct-wrong and env.snapshot and env.snapshot[1]==correct:
                        expected=.3 if initial==wrong or arm=='D0' and departure>400 else 0.
                        assert abs(info['c_prep']-expected)<1e-6,(arm,route,info)
                        results.append(dict(condition=arm,route=route,start=initial,depart=departure,requested_return=back,actual_return=rem,payment=info['c_prep'],cycles=list(env.d_cycles)))
                        break
                    assert not(done or cut)
                else:raise RuntimeError('missing return')
        finally:env.close()
    save(OUT/'reward_probe.json',results)

def prepare():
    decision=json.loads((OUT/'decision.json').read_text())
    assert decision['arms']==['D0','D1']
    assert not list(OUT.glob('D*_*/*episodes*.jsonl'))
    check()
    base=json.loads((C/'design.json').read_text())
    checks={}
    for stage in STAGES:
        cfg=m.prepare_scenario(stage,OUT/'configs')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cfg.parent.iterdir() if p.is_file()}
        assert hashes==base['checks'][stage]['hashes']
        checks[stage]=dict(base['checks'][stage],config=str(cfg),hashes=hashes)
    for seed in SEEDS:
        h=[]
        for a in decision['arms']:
            d=OUT/f'{a}_{seed}';d.mkdir(exist_ok=True)
            agent=agent_for(seed)
            assert not agent.optimizer.state and agent.num_timesteps==0
            h.append(digest(agent))
            agent.save(str(d/'policy_0.pt'))
        assert len(set(h))==1
        checks[str(seed)]=h
    save(OUT/'design.json',dict(arms=decision['arms'],ppo=base['ppo'],checks=checks,pairs=base['pairs'],seeds=SEEDS,select=SELECT,hold=HOLD,
        selection='High traffic route13 from lane 1 AND route16 from lane 0 with identical SUMO seed; fewer collisions across all 16 routes; higher arrivals; earliest step.',
        note='Both departure lanes use each SUMO seed. Different requested lanes may cause different actual insertion times and speeds; log starts. Deterministic fixed-policy evaluation.'))
if __name__=='__main__':prepare()
