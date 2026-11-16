"""Offline paired-input audit; action differences are not rollout success rates."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from algorithms.ppo import PPO

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    source = ROOT / 'results/traffic_recovery_auto_20261116/round-02/runs/route'
    manifest = json.loads((source / 'manifest.json').read_text())
    agent = PPO(obs_dim=63, act_dim=2, **manifest['hp'])
    agent.load(str(source / 'policy-200000.pt'))
    specs = [
        ('future', 'lane_followup_20261116/early-evaluation/51200/future/policy-51200-evaluation.json', (47, 50)),
        ('junction', 'junction_conflict_trials_20261116/early-evaluation/51200/combined/policy-51200-evaluation.json', (60, 63)),
        ('lateral', 'lane_followup_20261116/early-evaluation/51200/lateral/policy-51200-evaluation.json', (57, 60)),
    ]
    result = {}
    for name, file, (start, end) in specs:
        path = ROOT / 'results' / file
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        states = [state for r in rows for state in [r['first_state'], *r['tail']] if r['first_state']]
        active = [state for state in states if np.any(np.asarray(state['obs'])[start:end] != 0)]
        zeroed = copy.deepcopy(agent)
        with torch.no_grad():
            zeroed.policy.trunk[0].weight[:, start:end] = 0
        delta = []
        residual = []
        for state in active:
            obs = np.asarray(state['obs'], dtype=np.float32)
            old = obs.copy()
            old[start:end] = 0
            baseline = agent.predict(old, deterministic=True)
            added = agent.predict(obs, deterministic=True)
            preserved = zeroed.predict(obs, deterministic=True)
            delta.append(np.abs(added - baseline))
            residual.append(float(np.max(np.abs(preserved - baseline))))
        assert max(residual, default=0) < 1e-6
        differences = np.asarray(delta)
        result[name] = dict(source=str(path.relative_to(ROOT)), sampled_states=len(states), active_states=len(active),
                            mean_absolute_action_change=differences.mean(axis=0).tolist() if len(active) else [],
                            lane_action_changes=int(np.sum(differences[:, 1] > 1e-6)) if len(active) else 0,
                            added_column_weight_norm=float(agent.policy.trunk[0].weight[:, start:end].detach().norm()),
                            zero_column_max_action_difference=max(residual, default=0),
                            limitation='Paired inputs on saved states, not independent samples or a causal rollout outcome comparison.')
    output = ROOT / 'results/observation_initialization_audit_20261116.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
