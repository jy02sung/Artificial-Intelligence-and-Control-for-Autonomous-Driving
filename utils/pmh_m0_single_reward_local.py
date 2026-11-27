"""Run the M0 single-reward control (utils.jys3_m0_single_reward) on this PC, with a local M0 (pmh).

Why: on macOS arm64 the M0 initial policies of seeds 11/22/33 do not match the
team's SHA-256 in results/jys3_ppo_clean_ladder/protocol.json (orthogonal
init goes through LAPACK QR, which differs between platforms), so the runner's
--check stops. The user chose to train M0 here as well, from the same local
initial weights as the candidate, so the paired comparison differs only in the
one reward term. The team's M0 can still be paired later with --summarize of
the original runner.

Everything else is the original runner: reward table, gamma .99, shared critic,
256x256 ReLU policy, PPO settings, training traffic formula, 250k steps, marks
every 50k, 128-episode screen panel (95380000), selection rule, and the three
holdouts (81260100 x64, 91000000 x256, 104320000 x256 = 576). The only change
is that the expected initial hash is this PC's own digest (recorded in
protocol_local.json together with the team hash).

Usage: python -m utils.pmh_m0_single_reward_local arrival|M0 [--workers 4]
       python -m utils.pmh_m0_single_reward_local --summarize-local arrival
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import jys3_m0_single_reward as sr

ladder = sr.ladder
sr.CHANGES.setdefault('M0', ('ARRIVAL_REWARD', 5.))          # M0 itself: the base value, no change
sr.NAMES.setdefault('M0', 'M0 기준')
_config = sr.reward_config
sr.reward_config = lambda arm: dict(sr.BASE) if arm == 'M0' else _config(arm)
LOCAL_BASELINE = sr.ROOT / 'local_baseline'


def local_hash(seed):
    return sr.p.digest(ladder.fresh('M0', seed).policy)


def train(arm, seed, n):
    out = sr.folder(arm, seed)
    if (out / f'policy_{n}.pt').is_file():
        return
    sr.p.train(arm, n, directory=out, make_agent=lambda: ladder.fresh('M0', seed), make_episodes=sr.Episodes,
               expected_hash=local_hash(seed), training_seed=seed, update=sr.c.verified_update)


def run(arm, workers):
    team = sr.read(ladder.ROOT / 'protocol.json')['initial']
    config = dict(arm=arm, rewards=sr.reward_config(arm), gamma=.99, progress=2., off_route=-5., clear_stop=False,
                  shared_critic=True, seed=sr.SEEDS, steps=250000, marks=sr.MARKS, panels=sr.old.PANELS,
                  initial_local={s: local_hash(s) for s in sr.SEEDS},
                  initial_team={s: team[str(s)]['shared'] for s in sr.SEEDS},
                  note='initial weights generated on this PC (macOS arm64); team hash does not match')
    sr.save(sr.ROOT / arm / 'protocol_local.json', config)
    with ProcessPoolExecutor(workers) as pool:
        for n in sr.MARKS:
            for job in as_completed([pool.submit(train, arm, s, n) for s in sr.SEEDS]):
                job.result()
            sr.collect(pool, arm, [(s, n) for s in sr.SEEDS], 'screen')
        checks = [(s, sr.pick(sr.folder(arm, s))) for s in sr.SEEDS]
        for h in sr.c.HOLDS:
            sr.collect(pool, arm, checks, h)


def summarize_local(arm):
    """Pair the candidate with the M0 trained here (folder layout the runner expects: root/M0_s<seed>)."""
    LOCAL_BASELINE.mkdir(parents=True, exist_ok=True)
    for s in sr.SEEDS:
        link = LOCAL_BASELINE / f'M0_s{s}'
        if not link.exists():
            os.symlink(sr.folder('M0', s), link)
    sr.summarize(arm, LOCAL_BASELINE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=sorted(sr.CHANGES))
    p.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    p.add_argument('--summarize-local', action='store_true')
    a = p.parse_args()
    ladder.torch.set_num_threads(1)
    if a.summarize_local:
        summarize_local(a.arm)
    else:
        run(a.arm, a.workers)


if __name__ == '__main__':
    main()
