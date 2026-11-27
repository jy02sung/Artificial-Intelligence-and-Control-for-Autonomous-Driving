"""Replay the published 50-input imitation policies with lane-driven fork traversal."""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import jys3_dagger_one as dagger
from utils import jys3_physical_forks as physical

v3 = physical.v3
ROOT = dagger.BC.parent / "jys3_imitation_physical"
MODELS = {
    "BC": (dagger.BC / "bc.pt", dagger.BC / "student.json"),
    "BC_DAgger": (dagger.OUT / "dagger1.pt", dagger.OUT / "evaluation.json"),
}


def drive(env, model, route, seed, lane, corrected):
    env.ego_depart_lane = str(lane)
    obs, _ = env.reset(seed=seed, options={"route": f"route_{route:02d}"})
    start = [env.snapshot[i] for i in (0, 1, 7, -1)]
    initial = obs[:50].copy()
    for t in range(1, 1001):
        assert obs.shape == ((52,) if corrected else (50,))
        if corrected:
            assert tuple(obs[50:]) == (0., 0.)
        before = env.snapshot
        act = model.predict(obs[:50], deterministic=True)
        obs, _, done, cut, info = env.step(act)
        if done or cut:
            pos = env.snapshot or before
            outcome = (info['outcome'] if corrected else
                       'arrival' if info['arrived'] else 'collision' if info['collided']
                       else 'stall' if info['stalled'] else 'timeout')
            return initial, dict(start=start, outcome=outcome, steps=t, edge=pos[0], lane=pos[1],
                                 detail=info.get('off_route_detail'))
    raise AssertionError("episode exceeded original time limit")


def batch(job):
    name, rep = job
    weights, original = MODELS[name]
    dagger.torch.set_num_threads(1)
    model = dagger.policy(weights)
    targets = v3.read(original)[16*rep:16*(rep+1)]
    assert [r['route'] for r in targets] == list(range(1,17))
    records, starts = [], []
    with TemporaryDirectory() as tmp:
        legacy = dagger.environment('high_real_ratio',11,Path(tmp)/'legacy')
        try:
            for r in targets:
                initial, row = drive(legacy,model,r['route'],r['seed'],r['lane'],False)
                assert (row['steps'],row['outcome']) == (r['steps'],r['outcome']), (job,r,row)
                starts.append(initial)
                records.append(dict(route=r['route'],seed=r['seed'],depart_lane=r['lane'],case=rep,old=row))
        finally:
            legacy.close()
        env = physical.Env(v3.ARMS[0],Path(tmp)/'physical',11)
        # Preserve the original meaning and order of the imitation policy's road inputs.
        env.sim.signs = legacy.signs
        env.sim.road_flags = legacy.road_flags
        try:
            for r, initial in zip(records, starts):
                obs, row = drive(env,model,r['route'],r['seed'],r['depart_lane'],True)
                assert (obs==initial).all() and r['old']['start']==row['start']
                r['new']=row
        finally:
            env.close()
    v3.save(ROOT/f"{name}_{rep}.json",dict(sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), rows=records))


if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(batch, [(name,rep) for rep in range(4) for name in MODELS]):
            pass
