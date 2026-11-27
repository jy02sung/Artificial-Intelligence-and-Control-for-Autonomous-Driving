"""Reproduce the final team policies using the distributed scenario and source."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

from run import ROOT, model_path


def pure_ppo():
    from utils import jys3_m5ft_targeted as last
    ladder = last.L
    q = ladder.q
    seed = 33
    initializer = partial(q.p.agent, seed, 2048, 64)
    initial_hash = q.p.digest(initializer().policy)
    for step in range(50000, 250001, 50000):
        dest = q.folder("Q1", seed)
        if not (dest / f"policy_{step}.pt").exists():
            q.p.train("Q1", step, directory=dest, make_agent=initializer, make_episodes=q.Episodes,
                      expected_hash=initial_hash, training_seed=seed, update=q.update)
    ladder.train("M5-FT", seed, end=125000)
    dest = ROOT / "results/pure_ppo_final"
    last.train(("A", seed, "A"), directory=dest, marks=(12500, 25000, 50000, 75000, 87500, 100000),
               collision_reward=-14., gae_lambda=.95)
    print(dest / "policy_100000.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="dagger5",
                        choices=("dagger5", "bc50", "bc100", "uniform_dagger5", "dagger5_ppo", "pure_ppo"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["IC_LIBSUMO"] = "1"
    if args.model == "pure_ppo":
        pure_ppo()
        return
    if args.model == "dagger5_ppo":
        from utils import jys3_relu_ppo_comparison as ppo
        ppo.ORIG.update(BC=model_path("bc50"), R5=model_path("dagger5"))
        ppo.train("B", fixed_budget=True)
        print(ppo.ROOT / "B/policy_32768.pt")
        return

    from utils import jys3_final_relu_dagger as final
    final.ROOT.mkdir(parents=True, exist_ok=True)
    final.torch.set_num_threads(2)
    with ProcessPoolExecutor(args.workers) as pool:
        if args.model == "dagger5":
            result = final.pipeline(pool)
            print(final.checkpoint(result["best"]))
            return
        selected, train, valid = final.collect_demos(pool)
        x, y = final.dataset(train, "train")
        vx, vy = final.dataset(valid, "valid")
        final.fit(0, x, y, vx, vy, [])
        previous = final.checkpoint(0)
        if args.model == "bc100":
            final.fit(0, x, y, vx, vy, [], extended_bc=True)
            previous = final.ROOT / "bc100/bc.pt"
        if args.model == "uniform_dagger5":
            dest = final.ROOT.parent / "jys3_uniform_dagger"
            dest.mkdir(exist_ok=True)
            for k in range(1, 6):
                data = dest / f"R{k}"
                rows = final.schedule(pool, data, 78000000 + 100000 * k, 9, selected, previous, collect=True)
                xx, yy = final.dataset(rows, data)
                x, y = final.torch.cat((x, xx)), final.torch.cat((y, yy))
                final.fit(k, x, y, vx, vy, [], uniform_dir=dest, initial=previous)
                previous = dest / f"dagger{k}.pt"
        print(previous)


if __name__ == "__main__":
    main()
