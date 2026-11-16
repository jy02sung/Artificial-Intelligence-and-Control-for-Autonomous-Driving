"""Reset only the scalar acceleration exploration parameter."""
import math
import torch


def reset_acceleration_std(policy, std):
    if not math.isfinite(std) or std <= 0:
        raise ValueError('std must be finite and positive')
    if policy.log_std.numel()!=1:
        raise ValueError('Expected one acceleration exploration parameter')
    with torch.no_grad():
        policy.log_std.fill_(math.log(std))
