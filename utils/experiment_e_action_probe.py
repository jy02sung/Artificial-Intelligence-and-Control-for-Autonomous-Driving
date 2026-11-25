"""Matched 200k policy evaluation: stochastic lane, acceleration, or both."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools, json
import torch
from utils.experiment_e import OUT, agent, make
from utils.experiment_b_eval import evaluate_episode
from utils.experiment_b import save

def run(job):
    arm,seed=job
    torch.set_num_threads(1)
    model=agent(seed)
    model.load(str(OUT/f"{arm}_{seed}/policy_200000.pt"))
    model.policy.eval()
    folder=OUT/'training_replay'/f"{arm}_{seed}"/"matched_probe"
    folder.mkdir(exist_ok=True,parents=True)
    env=make("high_real_ratio",arm,seed,folder/"map")
    rows=[]
    try:
        for sim in (980000,980100,980200,980300):
            for mode in ("greedy","sample_lane","sample_accel","sample_both"):
                for rep in range(1 if mode=="greedy" else 3):
                    torch.manual_seed(40000000+sim+rep)
                    class Policy:
                        @torch.no_grad()
                        def predict(self,obs,deterministic=True):
                            mu,std,logits,_=model.policy(torch.as_tensor(obs)[None])
                            # Draw both in a fixed order for paired action randomness.
                            accel=torch.normal(mu,std)
                            lane=torch.distributions.Categorical(logits=logits).sample()
                            if mode not in ("sample_both","sample_accel"):accel=mu
                            if mode not in ("sample_both","sample_lane"):lane=logits.argmax(-1)
                            return [float(accel.item()),model.policy.LANE_CMDS[lane.item()]]
                    r=evaluate_episode(env,Policy(),"route_16",sim,0)
                    rows.append(dict(sim=sim,mode=mode,rep=rep,passed=r['first']['actual']==r['first']['expected'],left=env.c_commands.get('1',0),outcome=r['outcome'],start=r['start']))
            save(folder/"results.json",rows)
    finally:env.close()
if __name__=='__main__':
    with ProcessPoolExecutor(3) as pool:list(pool.map(run,itertools.product(('E0','E1','E2','E3'),(11,22,33))))
