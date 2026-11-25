"""Compare lane-choice probabilities on matched held-out starting observations."""
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from utils.jys3_fork1314_pilot import Pilot, panel, ROOT, TRAIN_SEEDS, j, pure, write


def main():
    j.torch.set_num_threads(1)
    cases = panel("holdout")
    with TemporaryDirectory() as tmp:
        env = Pilot(Path(tmp), 11, False)
        try:
            states = [env.reset(case)[0].copy() for case in cases]
        finally:
            env.env.close()
    for first, second in zip(states[::2], states[1::2]):
        assert np.flatnonzero(first != second).tolist() == [34]
    inputs = j.torch.as_tensor(np.stack(states), dtype=j.torch.float32)
    for seed in TRAIN_SEEDS:
        model = pure.agent(seed)
        assert tuple(model.policy.LANE_CMDS) == (1., 0., -1.)
        for mark in (0, 50000):
            folder = ROOT / f"seed{seed}"
            model.policy.load_state_dict(j.torch.load(folder/f"policy_{mark}.pt", weights_only=True))
            model.policy.eval()
            with j.torch.no_grad():
                probabilities = j.torch.softmax(model.policy(inputs)[2], dim=-1).numpy()
            rows = [dict(route=route, lane=lane, traffic_seed=seed_id, left=float(probs[0]),
                         keep=float(probs[1]), right=float(probs[2]))
                    for (route, lane, seed_id), probs in zip(cases, probabilities)]
            write(folder/f"start_prob_{mark}.json", rows)


if __name__ == "__main__":
    main()
