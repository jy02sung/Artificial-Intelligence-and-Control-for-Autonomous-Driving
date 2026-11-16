"""Change initial acceleration exploration only; preserve all deterministic actions."""
import math
import sys
from pathlib import Path
from utils import run_safety_strength_trials as original
from utils.acceleration_exploration import reset_acceleration_std
s=original.s
t=s.t
s.OUT=s.ROOT/'results/braking_exploration_trial_20261116'
t.OUT=s.OUT
s.DOC=s.ROOT/'docs/experiments/ic-braking-exploration-results-20261116.md'
t.DOC=s.DOC
s.CONDITIONS=('anticipatory',)
t.CONDITIONS=s.CONDITIONS
s.SOURCE_FILES=(*s.SOURCE_FILES,'utils/acceleration_exploration.py','utils/run_braking_exploration_trial.py')
t.SOURCE_FILES=s.SOURCE_FILES
BasePPO=s.a.b.PPO


class ExplorationPPO(BasePPO):
    def load(self,path):
        result=super().load(path)
        if Path(path).resolve()==s.SOURCE.resolve():
            reset_acceleration_std(self.policy,math.exp(-.5))
        return result


s.a.b.PPO=ExplorationPPO

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--run-condition':
        original.run_one(sys.argv[2])
    else:
        s.main()
