"""Single added reward term versus the anticipatory safety condition."""
import sys
from utils import run_safety_strength_trials as original
from env.risk_action_trial import RiskActionEnv
s=original.s
t=s.t
s.OUT=s.ROOT/'results/risk_action_trial_20261116'
t.OUT=s.OUT
s.DOC=s.ROOT/'docs/experiments/ic-risk-action-results-20261116.md'
t.DOC=s.DOC
s.CONDITIONS=('anticipatory',)
t.CONDITIONS=s.CONDITIONS
t.TrialEnv=RiskActionEnv
s.SOURCE_FILES=(*s.SOURCE_FILES,'env/risk_action_trial.py','utils/run_risk_action_trial.py')
t.SOURCE_FILES=s.SOURCE_FILES
if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--run-condition':
        original.run_one(sys.argv[2])
    else:
        s.main()
