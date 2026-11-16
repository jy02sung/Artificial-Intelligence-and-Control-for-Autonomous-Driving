"""Evaluate a frozen checkpoint without modifying its active training directory."""
import argparse
import importlib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices=('junction', 'lane'))
    parser.add_argument('condition')
    parser.add_argument('step', type=int)
    args = parser.parse_args()
    module = importlib.import_module('utils.run_junction_conflict_trials' if args.family == 'junction' else 'utils.run_lane_followup')
    s = module.s
    t = s.t
    assert args.condition in s.CONDITIONS
    s.a.b.torch.set_num_threads(1)
    source = s.OUT / 'runs' / args.condition
    checkpoint = source / f'policy-{args.step}.pt'
    assert checkpoint.exists(), checkpoint
    digest = s.a.b.digest(checkpoint)
    dest = s.OUT / 'early-evaluation' / str(args.step) / args.condition
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / 'evaluation.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        manifest = t.read(source / 'manifest.json')
        for file, expected in t.read(s.OUT / 'manifest.json')['source_hashes'].items():
            assert s.a.b.digest(s.ROOT / file) == expected, file
        t.save(dest / 'manifest.json', manifest)
        output = dest / (checkpoint.stem + '-evaluation.json')
        rows = t.read(output) if output.exists() else s.evaluate(args.condition, checkpoint, dest)
        assert len(rows) == 64
        assert len({(r['route'], r['seed'], r['traffic']) for r in rows}) == 64
        assert all(r['condition'] == args.condition and r['checkpoint'] == checkpoint.name and r['route_overrides'] == 0 for r in rows)
        assert s.a.b.digest(checkpoint) == digest
        counts = {mode: {outcome: sum(r['traffic'] == mode and r['outcome'] == outcome for r in rows)
                         for outcome in ('arrival', 'collision', 'timeout', 'wrong_exit')}
                  for mode in ('none', 'reference')}
        result = dict(condition=args.condition, checkpoint=checkpoint.name, checkpoint_sha=digest, counts=counts)
        t.save(dest / 'complete.json', result)
        print(result, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
