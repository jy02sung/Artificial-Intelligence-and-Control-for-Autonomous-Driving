"""One-time lane progress credit; independent of SUMO for validation."""


class PrepareCredit:
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
