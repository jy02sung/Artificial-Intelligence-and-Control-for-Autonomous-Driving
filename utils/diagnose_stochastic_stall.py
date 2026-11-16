"""Diagnostic only: stochastic rollouts of one deterministic timeout case."""
from utils import run_safety_strength_trials as trial
s=trial.s
t=s.t

def main():
    s.a.b.torch.set_num_threads(1)
    cp=s.OUT/'runs/collision20/policy-100352.pt'
    meta=t.read(cp.parent/'manifest.json')
    for file,digest in meta['source_hashes'].items():
        assert s.a.b.digest(s.ROOT/file)==digest
    results=[]
    for sampling_seed in (1101,1102,1103):
        agent=s.a.b.PPO(obs_dim=63,act_dim=2,**meta['hp'])
        agent.load(str(cp))
        env=s.make_env('collision20',evaluation=True,none=True)
        try:
            obs,_=env.reset(seed=79101,options={'route':'route_05'})
            s.a.b.torch.manual_seed(sampling_seed)
            while True:
                action=agent.predict(obs,deterministic=False)
                obs,_,done,cut,info=env.step(action)
                if done or cut or env.step_count>=4500:
                    outcome='collision' if info['collided'] else 'arrival' if info['arrived'] else 'timeout' if cut else 'wrong_exit' if done else 'diagnostic_limit'
                    row=dict(sampling_seed=sampling_seed,outcome=outcome,seconds=env.step_count*env.step_length,route_overrides=env.route_override_count)
                    assert env.route_override_count==0
                    results.append(row)
                    print(row,flush=True)
                    break
        finally:
            env.close()
    t.save(s.OUT/'stochastic-stall-diagnostic.json',dict(checkpoint_sha=s.a.b.digest(cp),route='route_05',traffic='none',simulation_seed=79101,policy='stochastic',diagnostic_limit_seconds=900,results=results,interpretation='Diagnostic only; not comparable to deterministic 48-case benchmark or proof of safety.'))

if __name__=='__main__':main()
