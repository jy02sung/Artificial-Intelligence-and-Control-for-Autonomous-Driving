"""Four controlled junction-conflict trials on the successful route candidate."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['IC_RECOVERY_CONFIG']=str(ROOT/'results/traffic_recovery_auto_20261116/round-02/config.json')
from utils import run_traffic_recovery_auto as a
from env.junction_conflict_trials import ConflictEnv
s=a.s;t=s.t
s.OUT=ROOT/'results/junction_conflict_trials_20261116';t.OUT=s.OUT
s.DOC=ROOT/'docs/experiments/ic-junction-conflict-results-20261116.md';t.DOC=s.DOC
s.SOURCE=ROOT/'results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000.pt'
s.CONDITIONS=('control','observation','reward','combined');t.CONDITIONS=s.CONDITIONS
s.t.TrialEnv=ConflictEnv
s.SOURCE_FILES=(*s.SOURCE_FILES,'env/junction_conflict_trials.py','utils/run_junction_conflict_trials.py');t.SOURCE_FILES=s.SOURCE_FILES
s.t.EVAL_SEEDS=(79101,79102,79103);s.EVAL_SEEDS=s.t.EVAL_SEEDS
_publish=s.publish

def publish():
    _publish()
    text=s.DOC.read_text(encoding='utf8').replace('59101~59103','79101~79103').replace('lane/policy-151552.pt','round-02/route/policy-200000.pt').replace('안전 기준점에서 길찾기 보상 추가 비교','교차 충돌 관측·보상 독립 및 조합 실험')
    s.DOC.write_text(text,encoding='utf8')
s.publish=publish
if __name__=='__main__':s.main()
