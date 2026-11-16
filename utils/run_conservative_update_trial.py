"""Single-factor learning-rate comparison for the unchanged anticipatory condition."""
import sys
from pathlib import Path
from utils import run_safety_strength_trials as original
from utils.conservative_update import with_learning_rate
s=original.s
t=s.t
s.OUT=s.ROOT/'results/conservative_update_trial_20261116'
t.OUT=s.OUT
s.DOC=s.ROOT/'docs/experiments/ic-conservative-update-results-20261116.md'
t.DOC=s.DOC
s.CONDITIONS=('anticipatory',)
t.CONDITIONS=s.CONDITIONS
s.SOURCE_FILES=(*s.SOURCE_FILES,'utils/conservative_update.py','utils/run_conservative_update_trial.py')
t.SOURCE_FILES=s.SOURCE_FILES
hp_source=(s.a.r.sources('baseline',22)/'manifest.json').resolve()
original_read=s.read


def read_training_metadata(path):
    metadata=original_read(path)
    return with_learning_rate(metadata,.00003) if Path(path).resolve()==hp_source else metadata


s.read=read_training_metadata

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--run-condition':
        original.run_one(sys.argv[2])
    else:
        s.main()
