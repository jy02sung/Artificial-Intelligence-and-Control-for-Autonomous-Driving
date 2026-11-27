class ProgressReward:
    def __init__(self): self.paid=set()
    def award(self,edge,targets,before,after,distance):
        if not targets or distance is None or not 0<=distance<=100: return 0.
        a=min(abs(before-t) for t in targets); b=min(abs(after-t) for t in targets)
        reward=0.
        for level in range(b,a):
            key=(edge,tuple(sorted(targets)),level)
            if key not in self.paid:
                self.paid.add(key); reward+=.1
        return reward
