"""Lane-preparation shaping terms; independent of SUMO for validation."""


class PrepareCredit:
    """One-time credit when the lane gap to the route-compatible lanes shrinks."""

    def __init__(self, horizon=300.0, weight=0.1):
        self.horizon = horizon
        self.weight = weight
        self.paid = set()

    def award(self, edge, decision, targets, before, after, distance):
        if not targets or distance is None or not 0 <= distance <= self.horizon:
            return 0.0
        old = min(abs(before - target) for target in targets)
        new = min(abs(after - target) for target in targets)
        reward = 0.0
        for level in range(new, old):
            key = (edge, decision, tuple(sorted(targets)), level)
            if key not in self.paid:
                self.paid.add(key)
                reward += self.weight
        return reward


class PreparePressure:
    """Per-step penalty while lane changes are still owed near a split.

    penalty = weight * remaining_changes * (1 - distance / horizon)

    Zero in a compatible lane and zero beyond the horizon, so it only
    removes the "wait in the wrong lane for free" option. Grows as the split
    gets closer and stays at its maximum while stopped at the lane end,
    which is the no-traffic deadlock seen in pmh_prepare300_fresh.

    weight=0.01 at 0.2s steps: one owed change at the split costs 0.05/s,
    so ~100s of waiting equals the collision penalty (5.0). Keeping this
    above typical gap-waiting time avoids teaching forced cut-ins.
    """

    def __init__(self, horizon=300.0, weight=0.01):
        self.horizon = horizon
        self.weight = weight

    def penalty(self, targets, lane, distance):
        if not targets or distance is None or not 0 <= distance <= self.horizon:
            return 0.0
        remaining = min(abs(lane - target) for target in targets)
        return self.weight * remaining * (1.0 - distance / self.horizon)


def lane_change_risk(front, rear, ego_speed, min_gap=5.0, horizon=3.0):
    """Risk in [0, 1] of moving into a target lane with these neighbours.

    front/rear are (gap_m, speed_mps) of the nearest target-lane leader and
    follower, or None. Each side scores the worse of a gap term (gap below
    min_gap) and a TTC term (time to contact below horizon at the current
    closing speed). Used as a penalty, not a gate: the policy still decides.
    """
    def side(neighbour, closing):
        if neighbour is None:
            return 0.0
        gap = neighbour[0]
        gap_term = min(max((min_gap - gap) / min_gap, 0.0), 1.0)
        rate = max(closing(neighbour[1]), 0.0)
        ttc_term = 0.0 if rate <= 0 else min(max((horizon - max(gap, 0.0) / rate) / horizon, 0.0), 1.0)
        return max(gap_term, ttc_term)

    return max(side(front, lambda speed: ego_speed - speed),
               side(rear, lambda speed: speed - ego_speed))


def approach_risk(leader, ego_speed, horizon=3.0):
    """Same-lane rear-end risk in [0, 1] from time to contact with the leader.

    leader is (gap_m, speed_mps) of the visible same-lane leader, or None.
    Zero when not closing in (equal or slower ego speed), so queues and
    steady following are free; close following itself is left to the
    existing close_gap_penalty. Braking lowers the closing speed and the
    penalty at once, giving a dense signal to slow down early.
    """
    if leader is None:
        return 0.0
    gap, leader_speed = leader
    closing = ego_speed - leader_speed
    if closing <= 0:
        return 0.0
    ttc = max(gap, 0.0) / closing
    return min(max((horizon - ttc) / horizon, 0.0), 1.0)
