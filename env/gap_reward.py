"""Gap selection on observable adjacent lanes, shared by training labels and reward."""
from typing import NamedTuple

import numpy as np


class GapOpportunity(NamedTuple):
    direction: int
    available: bool
    approach_speed: float
    front_gap: float
    front_speed: float
    immediate: bool


def opportunity(obs, vmax, visibility):
    immediate = int(round(float(obs[31]) * 3))
    prepare = int(round(float(obs[33]) * 3)) if 0 <= obs[32] <= .7 else 0
    direction = int(np.sign(immediate or prepare))
    if not direction or obs[51] < .22 or obs[22 if direction > 0 else 24] < 0:
        return None
    first = 5 if direction > 0 else 13
    front_gap = float(obs[first]) * visibility
    rear_gap = -float(obs[first + 2]) * visibility
    speed = float(obs[0]) * vmax
    front_speed = max(0., speed + float(obs[first + 1]) * vmax)
    rear_speed = max(0., speed + float(obs[first + 3]) * vmax)
    front_required = 3. + .35 * speed + 1.25 * max(0., speed - front_speed)
    rear_required = 3. + .35 * rear_speed + 1.25 * max(0., rear_speed - speed)
    front_clear = front_gap >= visibility - .01 or front_gap > front_required
    rear_clear = rear_gap >= visibility - .01 or rear_gap > rear_required
    remaining = float(obs[52]) * 300.
    approach_speed = min(12., max(1.25, (remaining - 10.) / 5.))
    if not front_clear:
        approach_speed = min(approach_speed, max(.9, front_speed - 2.8))
    if not rear_clear:
        if front_gap > front_required + 8. and rear_gap > 1.5 and rear_speed <= speed + .7 and remaining > 20.:
            approach_speed = max(approach_speed, min(speed + 2., float(obs[34]) * vmax))
        else:
            approach_speed = min(approach_speed, max(1.2, rear_speed - 3.))
    return GapOpportunity(direction, front_clear and rear_clear, approach_speed, front_gap, front_speed, bool(immediate))


def gap_reward(gap, speed, commanded, actual, step):
    """Score lateral progress on the same edge using the gap before the manoeuvre."""
    if gap is None:
        return -.9 * step if commanded else 0.
    if commanded and (commanded != gap.direction or not gap.available):
        return -.9 * step
    if actual == gap.direction and gap.available:
        return 1.25
    if actual == gap.direction and not gap.available:
        return -.9 * step
    if gap.available:
        return -.40 * step
    if gap.immediate:
        return -.55 * step * min(1., max(0., speed - gap.approach_speed) / 5.)
    return 0.
