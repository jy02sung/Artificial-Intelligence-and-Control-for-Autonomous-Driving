"""Copy training metadata with one explicit learning-rate change."""
import math


def with_learning_rate(manifest, rate):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    return dict(manifest, hp=dict(manifest['hp'], lr=rate))
