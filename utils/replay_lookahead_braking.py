"""Compare the preserved source and learned lookahead policy on the same merge."""
from utils import run_junction_lookahead_trials as trial
s=trial.s
t=s.t


def main():
    s.a.b.torch.set_num_threads(1)
    m=t.read(s.OUT/'runs/ahead150/manifest.json')
    for file,digest in m['source_hashes'].items():
        assert s.a.b.digest(s.ROOT/file)==digest
    for label,cp in [('source',s.SOURCE),('ahead150_51200',s.OUT/'runs/ahead150/policy-51200.pt')]:
        agent=s.a.b.PPO(obs_dim=63,act_dim=2,**m['hp']);agent.load(str(cp))
        env=s.make_env('ahead150',evaluation=True,none=False)
        v=s.a.b.traci.vehicle;trace=[]
        try:
            obs,_=env.reset(seed=79103,options={'route':'route_01'})
            while True:
                action=agent.predict(obs,deterministic=True)
                ablated=obs.copy();ablated[60:63]=0
                counterfactual=agent.predict(ablated,deterministic=True)
                row=dict(t=env.step_count*env.step_length,speed=v.getSpeed(env.ego_id),risk=env.conflict_features()[0],gap=env.safety_context(action),action=action.tolist(),counterfactual_action_without_merge_input=counterfactual.tolist())
                obs,reward,done,cut,info=env.step(action)
                if env.ego_id in v.getIDList():row['after_speed']=v.getSpeed(env.ego_id)
                trace.append(row)
                if done or cut or env.step_count*env.step_length>=20:
                    result=dict(label=label,checkpoint_sha=s.a.b.digest(cp),seconds=env.step_count*env.step_length,
                                outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'diagnostic_window_end',
                                first_risk=next((x for x in trace if x['risk']>0),None),
                                first_brake_after_risk=next((x for x in trace if x['risk']>0 and x['action'][0]<0),None),
                                route_overrides=env.route_override_count,trace=trace)
                    assert env.route_override_count==0
                    t.save(s.OUT/('braking-replay-'+label+'.json'),result)
                    print({k:v for k,v in result.items() if k!='trace'},flush=True)
                    break
        finally:env.close()

if __name__=='__main__':main()
