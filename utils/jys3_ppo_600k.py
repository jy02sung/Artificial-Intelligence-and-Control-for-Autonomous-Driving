"""Matched 600k PPO runs with independent endpoint validation."""
import os
os.environ["IC_LIBSUMO"] = "1"
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import time
from utils import jys3_ppo_compare as paired
from utils.jys3 import HP, SCHEMA, OBS_DIM, m, environment, ScheduledEpisodes, atomic_torch, torch

ROOT = paired.OUT / "four_600k"
ARMS = ("ppo_only", "bc", "dagger", "bc_dagger")
MARKS = (0, 32768, 100000, 200000, 300000, 400000, 500000, 600000)
BASE = 34110000

def train(arm):
    folder = ROOT / arm
    torch.set_num_threads(1)
    if arm == "ppo_only":
        agent = m.PPO(obs_dim=OBS_DIM, act_dim=2, buffer=m.RolloutBuffer(), seed=11, **HP)
    else:
        agent = paired.policy(m.ROOT / "models" / "jys3" / paired.MODELS[arm])
    agent.policy.train()
    env = environment("high_real_ratio", 11, folder / "map")
    try:
        schedule = ScheduledEpisodes(env, 11, folder / "episodes.jsonl")
        obs, _ = schedule.reset()
        stats = [0., 0]
        for target in MARKS:
            while agent.num_timesteps < target:
                steps = min(2048, target - agent.num_timesteps)
                obs, last, _ = agent.collect_rollout(schedule, obs, ep_stats=stats, steps=steps, timeout_is_failure=True)
                adv, ret = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
                agent.update(adv, ret)
                paired.save(folder / "progress.json", dict(step=agent.num_timesteps, time=time.time()))
            state = agent.policy.state_dict()
            atomic_torch(dict(schema=SCHEMA, policy=state), folder / f"policy_{target}.pt")
            if target in (0,32768):
                old = torch.load(paired.ck(arm,target), weights_only=True)["policy"]
                paired.save(folder / f"matches_old_{target}.json", all(torch.equal(v,old[k]) for k,v in state.items()))
    finally:
        env.close()

def validate(job):
    arm,rep = job
    paired.BASE["final600"] = BASE
    return paired.evaluate((f"four_600k/{arm}",600000,"final600",rep))

def main():
    ROOT.mkdir(exist_ok=False)
    for arm in ARMS:
        (ROOT / arm).mkdir()
    paired.save(ROOT / "protocol.json",dict(seed=11,marks=MARKS,hp=HP,traffic="high_real_ratio",fresh_base=BASE,initial_models=paired.MODELS))
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(train, ARMS))
    jobs = [(a,r) for a in ARMS for r in range(4)]
    rows = {a:[] for a in ARMS}
    with ProcessPoolExecutor(8) as pool:
        for (arm,rep),batch in zip(jobs,pool.map(validate,jobs)):
            rows[arm].extend(batch)
            paired.save(ROOT / arm / f"final_rep_{rep}.json",batch)
    ref = None
    for arm, items in rows.items():
        keys = [(r['case'],r['route'],r['seed'],r['start']) for r in items]
        assert len(set((r['case'],r['route']) for r in items)) == len(items) == 64
        assert ref is None or ref == keys
        ref = keys
    paired.save(ROOT / "summary.json",{arm:dict(Counter(r['outcome'] for r in items)) for arm,items in rows.items()})

if __name__ == "__main__":
    main()
