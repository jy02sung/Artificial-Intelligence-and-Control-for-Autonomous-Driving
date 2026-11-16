"""Change only timeout GAE boundaries relative to unchanged-policy continuation."""
import sys
from utils import run_junction_conflict_trials as original
from utils.timeout_rollout import TimeoutBoundaryMixin
s=original.s
t=s.t
s.OUT=s.ROOT/'results/timeout_boundary_trial_20261116'
t.OUT=s.OUT
s.DOC=s.ROOT/'docs/experiments/ic-timeout-boundary-results-20261116.md'
t.DOC=s.DOC
s.CONDITIONS=('control',)
t.CONDITIONS=s.CONDITIONS
s.SOURCE_FILES=(*s.SOURCE_FILES,'utils/timeout_rollout.py','utils/run_timeout_boundary_trial.py')
t.SOURCE_FILES=s.SOURCE_FILES
BasePPO=s.a.b.PPO
class BoundaryPPO(TimeoutBoundaryMixin,BasePPO):
    pass
s.a.b.PPO=BoundaryPPO

if __name__=='__main__':
    s.main()
