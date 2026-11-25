"""PPO-only tuning of the corrected-environment 50-feature BC+DAgger policy."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

from utils import jys3_clear_stop_trial as trial
from utils import jys3_train_physical_imitation as imitation

v3 = trial.v3
ROOT = imitation.ROOT.parent / 'jys3_physical_imitation_ppo'
INITIAL = imitation.ROOT / 'BC_DAgger' / 'dagger1.pt'
HASH = '6690df50d70c82c1b61230e2e90d112c6ddb03af61516385e1d8c56f66a83eb3'
INTERVAL = 8192


@contextmanager
def environment():
    with TemporaryDirectory() as tmp:
        meta = imitation.data.bc.environment('high_real_ratio', 11, Path(tmp)/'map')
        env = trial.Env(trial.ARM, Path(tmp)/'physical', 11)
        env.sim.signs, env.sim.road_flags = meta.signs, meta.road_flags
        try:
            yield env
        finally:
            env.close()


class Episodes(v3.j.ScheduledEpisodes):
    def reset(self):
        obs, info = super().reset()
        return obs[:50], info

    def step(self, act):
        obs, reward, done, cut, info = super().step(act)
        return obs[:50], reward, done, cut, info


def checkpoint(step):
    return ROOT/f'policy_{step}.pt' if step else INITIAL


def evaluate(step, panel, rep):
    imitation.torch.set_num_threads(1)
    model = imitation.dagger.policy(checkpoint(step))
    previous = v3.read(imitation.ROOT/'BC_DAgger'/panel/f'{rep}.json')
    rows = []
    with environment() as env:
        for ref in previous:
            _, row = imitation.replay.drive(env, model, ref['route'], ref['seed'], ref['depart_lane'], True)
            assert ref['start'] == row['start']
            if step == 0 and row['outcome'] != 'clear_stop':
                assert (row['steps'],row['outcome']) == (ref['steps'],ref['outcome'])
            rows.append(dict(**row, route=ref['route'], seed=ref['seed'], case=rep))
    return rows


def panel_counts(rows):
    return Counter(row['outcome'] for row in rows)


def decision(baseline, current):
    changes = []
    for panel in imitation.PANELS:
        before, after = map(panel_counts, (baseline[panel], current[panel]))
        changes.append((after['arrival']-before['arrival'], after['collision']-before['collision']))
    if all(a >= 0 and c <= 0 for a,c in changes) and sum(a for a,c in changes) >= 4:
        return 'improved'
    if all(a <= -4 and c >= 0 for a,c in changes):
        return 'worse'
    return 'extend'


def main():
    assert hashlib.sha256(INITIAL.read_bytes()).hexdigest() == HASH
    ROOT.mkdir(exist_ok=False)
    imitation.torch.set_num_threads(1)
    model = imitation.dagger.policy(INITIAL)
    model.policy.train()
    assert model.optimizer.param_groups[0]['lr'] == .0001
    jobs = {}
    def schedule(pool, step):
        jobs[step] = {panel:[pool.submit(evaluate,step,panel,rep) for rep in range(4)]
                      for panel in imitation.PANELS}
    def finish(step):
        rows={panel:[r for f in shards for r in f.result()] for panel,shards in jobs[step].items()}
        assert all(len(v)==64 for v in rows.values())
        v3.save(ROOT/f'eval_{step}.json', rows)
        return rows
    with ProcessPoolExecutor(6) as pool, environment() as env:
        schedule(pool,0)
        episodes = Episodes(env,11,ROOT/'episodes.jsonl')
        obs,_ = episodes.reset()
        stats=[0.,0]
        for step in range(2048,65537,2048):
            env.sums.clear()
            obs,last,_=model.collect_rollout(episodes, obs, ep_stats=stats, steps=2048, timeout_is_failure=True)
            adv,returns=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
            metrics=model.update(adv,returns)
            assert model.num_timesteps==step
            with (ROOT/'updates.jsonl').open('a') as fp:
                print(json.dumps(dict(step=step,rewards=dict(env.sums),**metrics)),file=fp)
            if step%INTERVAL:
                continue
            v3.j.atomic_torch(dict(schema=imitation.dagger.SCHEMA,policy=model.policy.state_dict()),checkpoint(step))
            schedule(pool,step)
            if step==INTERVAL:
                baseline=finish(0)
            if step==32768:
                current=finish(step)
                verdict=decision(baseline,current)
                v3.save(ROOT/'decision.json',dict(step=step,decision=verdict))
                if verdict!='extend':
                    break
        for mark in jobs:
            if not (ROOT/f'eval_{mark}.json').exists():
                finish(mark)


if __name__=='__main__':
    main()
