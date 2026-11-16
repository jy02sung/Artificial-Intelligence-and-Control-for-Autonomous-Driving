"""Reuse the same base/control; compare future-turn and lateral-risk observations."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['IC_RECOVERY_CONFIG']=str(ROOT/'results/traffic_recovery_auto_20261116/round-02/config.json')
from utils import run_traffic_recovery_auto as a
from env.lane_followup_observation import LaneFollowupEnv
s=a.s;t=s.t
s.OUT=ROOT/'results/lane_followup_20261116';t.OUT=s.OUT
s.DOC=ROOT/'docs/experiments/ic-lane-followup-results-20261116.md';t.DOC=s.DOC
s.SOURCE=ROOT/'results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000.pt'
s.CONDITIONS=('future','lateral');t.CONDITIONS=s.CONDITIONS;t.TrialEnv=LaneFollowupEnv
s.SOURCE_FILES=(*s.SOURCE_FILES,'env/lane_followup_observation.py','utils/run_lane_followup.py');t.SOURCE_FILES=s.SOURCE_FILES
s.EVAL_SEEDS=(79101,79102,79103);t.EVAL_SEEDS=s.EVAL_SEEDS
_publish=s.publish

def publish():
    _publish()
    text=s.DOC.read_text(encoding='utf8').replace('59101~59103','79101~79103').replace('lane/policy-151552.pt','round-02/route/policy-200000.pt').replace('안전 기준점에서 길찾기 보상 추가 비교','늦은 차로변경 관측 보완 독립 비교')
    s.DOC.write_text(text,encoding='utf8')
s.publish=publish
if __name__=='__main__':s.main()
