"""Finish evaluation of completed training, reusing only exact full early suites."""
import argparse
import importlib
import os
from utils.evaluate_live_checkpoint import FAMILIES


def validate_suite(rows, condition, checkpoint, seeds):
    expected = {(f'route_{i:02d}', mode, seed) for i in range(1, 17)
                for mode in ('none', 'reference') for seed in (seeds[:1] if mode == 'none' else seeds)}
    assert len(rows) == len(expected) == 64
    assert {(r['route'], r['traffic'], r['seed']) for r in rows} == expected
    assert all(r['condition'] == condition and r['checkpoint'] == checkpoint and r['route_overrides'] == 0 for r in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices=FAMILIES)
    parser.add_argument('condition')
    args = parser.parse_args()
    s = importlib.import_module(FAMILIES[args.family]).s
    t = s.t
    assert args.condition in s.CONDITIONS
    s.a.b.torch.set_num_threads(1)
    directory = s.OUT / 'runs' / args.condition
    completed = t.read(directory / 'training-complete.json')
    assert completed['steps'] == 200000
    assert s.a.b.digest(directory / 'model.pt') == completed['model_sha']
    manifest = t.read(s.OUT / 'manifest.json')
    for file, digest in manifest['source_hashes'].items():
        assert s.a.b.digest(s.ROOT / file) == digest, file
    for mode, files in manifest['inputs'].items():
        for file, digest in files.items():
            assert s.a.b.digest(s.OUT / 'inputs' / mode / file) == digest, (mode, file)
    lock = s.OUT / f'evaluation-{args.condition}.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        count = 0
        for checkpoint in sorted(directory.glob('policy-*.pt'), key=lambda p: int(p.stem.split('-')[1]), reverse=True):
            dest = directory / (checkpoint.stem + '-evaluation.json')
            if dest.exists():
                rows = t.read(dest)
            else:
                matches = [p for p in (s.OUT / 'early-evaluation').rglob(dest.name) if p.parent.name == args.condition]
                if matches:
                    source = matches[0]
                    assert t.read(source.parent / 'manifest.json') == t.read(directory / 'manifest.json')
                    provenance = t.read(source.parent / 'complete.json')
                    if 'checkpoint_sha' in provenance:
                        assert provenance['checkpoint_sha'] == s.a.b.digest(checkpoint)
                    rows = t.read(source)
                    validate_suite(rows, args.condition, checkpoint.name, s.EVAL_SEEDS)
                    t.save(dest, rows)
                    t.save(directory / (checkpoint.stem + '-evaluation-reuse.json'), dict(source=str(source), checkpoint_sha=s.a.b.digest(checkpoint)))
                else:
                    rows = s.evaluate(args.condition, checkpoint, directory)
            validate_suite(rows, args.condition, checkpoint.name, s.EVAL_SEEDS)
            count += len(rows)
        result = dict(condition=args.condition, steps=completed['steps'], evaluations=count, model_sha=completed['model_sha'])
        t.save(directory / 'complete.json', result)
        t.save(directory / 'progress.json', dict(stage='complete', **result))
        s.publish()
        print(result, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
