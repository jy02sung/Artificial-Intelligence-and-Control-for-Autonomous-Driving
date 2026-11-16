"""Activate previously zero observation slots without changing initial outputs."""
import torch


def preserve_existing_inputs(policy, columns):
    with torch.no_grad():
        policy.trunk[0].weight[:, list(columns)] = 0
