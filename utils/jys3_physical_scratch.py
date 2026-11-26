"""Run matched S1 and S2 from step zero after the frozen comparison is complete."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import msvcrt
from pathlib import Path
import time

from utils import jys3_physical_validation as validation
from utils import jys3_physical_forks as physical

v3 = physical.v3
ROOT = validation.ROOT.parent / "from_scratch_600k"
SEEDS = validation.SEEDS
END = 600000


def gate_ready():
    for arm, seed, step, panel, rep in validation.jobs():
        path = validation.result_path(arm, seed, step, panel, rep)
        if not path.is_file() or not (validation.source(arm,seed) / path.name).is_file():
            return False
    return True


def initialize():
    initial = {}
    for seed in SEEDS:
        model = v3.core.agent(seed)
        initial[seed] = v3.v1.digest(model)
        for arm in v3.ARMS:
            folder = ROOT / f"{arm}_{seed}"
            folder.mkdir(exist_ok=True)
            zero = folder / "policy_0.pt"
            if not zero.exists():
                v3.j.atomic_torch({"policy": model.policy.state_dict()}, zero)
            else:
                model.policy.load_state_dict(v3.j.torch.load(zero, weights_only=True)["policy"])
                assert v3.v1.digest(model) == initial[seed]
    return initial


def run(workers):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "manager.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        print("Waiting for the complete paired frozen evaluation.", flush=True)
        while not gate_ready():
            time.sleep(10)
        assert validation.report()
        initial = initialize()
        v3.save(ROOT / "protocol.json",dict(environment="lane-driven forks", baseline_environment_commit="b61447e",
                roots="independent random PPO, no demonstrations or restored historical weights",
                steps=END, checkpoint_interval=50000, seeds=SEEDS, initial_sha256=initial,
                collision=dict(zip(v3.ARMS,(-5,-7))), remaining_rewards=v3.protocol()["shared_rewards"],
                ppo=v3.j.HP, traffic="high_real_ratio", vph=15000, screen=v3.SCREEN,
                panels=validation.PANELS[END], workers=workers))
        print("Verified 1,536 frozen comparison drives. Starting six fresh policies.", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            screen=v3.screen_stage(v3.ARMS,SEEDS,END,ROOT,pool,from_stage1=False,env_type=physical.Env)
            selection={}
            pending=[]
            for arm in v3.ARMS:
                for seed in SEEDS:
                    assert v3.read(ROOT/f"{arm}_{seed}"/"initial.json")["sha256"]==initial[seed]
                def score(n):
                    c=Counter()
                    for seed in SEEDS:
                        c.update(screen[f"{arm}_{seed}_{n}"]["outcomes"])
                    return c["arrival"],-c["collision"],-n
                selection[arm]=max(v3.MARKS,key=score)
                for n in sorted({selection[arm], END}):
                    for panel in validation.PANELS[END]:
                        for seed in SEEDS:
                            for rep in range(4):
                                target=ROOT/f"{arm}_{seed}"/f"eval_{panel}_{n}_{rep}.json"
                                if not target.exists():
                                    pending.append(pool.submit(v3.eval_job,ROOT,arm,seed,n,rep,panel,env_type=physical.Env))
            v3.save(ROOT/"selection.json",selection)
            for job in pending:
                job.result()
        final={}
        for arm,best in selection.items():
            for step in sorted({best,END}):
                for panel in validation.PANELS[END]:
                    rows=[r for seed in SEEDS for r in v3.v2.snapshot_rows(ROOT,arm,seed,step,panel)]
                    assert len(rows)==192
                    final[f"{arm}_{step}_{panel}"]=v3.v1.aggregate(rows)
        v3.save(ROOT/"final_summary.json",final)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=12)
    run(parser.parse_args().workers)
