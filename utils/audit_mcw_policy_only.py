"""Check the no-gate contract and provenance of a concrete PPO checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmstart', required=True)
    parser.add_argument('--trained-checkpoint', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--evaluation-manifest', required=True)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_mcw_policy_only')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise RuntimeError('Policy contract tests failed')

    original = torch.load(args.warmstart, map_location='cpu', weights_only=True)
    trained = torch.load(args.trained_checkpoint, map_location='cpu', weights_only=True)
    candidate = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    evaluation = json.loads(Path(args.evaluation_manifest).read_text(encoding='utf8'))
    baseline = json.loads(Path(args.baseline_manifest).read_text(encoding='utf8'))
    assert digest(args.warmstart) == manifest['initial_model_sha256']
    assert digest(args.checkpoint) == evaluation['evaluated_model_sha256']
    assert trained.keys() == candidate.keys() == original.keys()
    assert all(torch.equal(trained[k], candidate[k]) for k in candidate), 'Evaluation changed trained weights'
    assert manifest['scenario_hashes'] == evaluation['scenario_hashes'] == baseline['scenario_hashes']
    assert not evaluation['action']['lane_change_safety']['enabled']
    assert not evaluation['teacher_labels']
    root = Path(__file__).resolve().parents[1]
    for name, sha256 in evaluation['code_hashes'].items():
        assert digest(root/name) == sha256, f'Runtime source changed: {name}'

    columns = manifest['risk_adaptation_columns']
    allowed = {'trunk.0.weight', 'v_head.weight', 'v_head.bias'}
    changes = {}
    for name, tensor in candidate.items():
        assert tensor.shape == original[name].shape, name
        changes[name] = int(torch.count_nonzero(tensor != original[name]))
        if columns and name not in allowed:
            assert torch.equal(tensor, original[name]), name
    if columns:
        other = [i for i in range(candidate['trunk.0.weight'].shape[1]) if i not in columns]
        assert torch.equal(candidate['trunk.0.weight'][:, other], original['trunk.0.weight'][:, other])
    else:
        assert not manifest['args']['adapt_risk']
        assert changes['lane_head.weight'] > 0
    removed = list(range(31, 34)) + list(range(37, 47))
    assert not torch.count_nonzero(candidate['trunk.0.weight'][:, removed])
    assert changes['trunk.0.weight'] > 0
    payload = dict(
        contract_tests_run=tested.testsRun, contract_tests_passed=True,
        update_scope='traffic_inputs_and_value' if columns else 'full_ppo_policy',
        trained_and_evaluated_weights_identical=True,
        scenario_identical_to_baseline=True, runtime_source_hashes_match=True,
        safety_gate_enabled=False, teacher_labels=False,
        removed_lane_hint_weights_zero=True,
        changed_parameter_counts=changes, trainable_input_columns=columns,
        warmstart_sha256=digest(args.warmstart),
        trained_checkpoint_sha256=digest(args.trained_checkpoint),
        evaluated_checkpoint_sha256=digest(args.checkpoint))
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
