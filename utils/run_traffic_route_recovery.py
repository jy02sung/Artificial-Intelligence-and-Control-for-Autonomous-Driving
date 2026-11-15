"""Three parallel warm-start trials using the established training/evaluation pipeline."""
from utils import run_traffic_safety_trials as s
from env.traffic_route_recovery import RouteRecoveryEnv

def configure():
    s.OUT=s.ROOT/'results/traffic_route_recovery_20261116'
    s.DOC=s.ROOT/'docs/experiments/ic-traffic-route-recovery-results-20261116.md'
    s.CONDITIONS=('control','route','opportunity')
    s.SOURCE=s.ROOT/'results/traffic_safety_warmstart_20261115/runs/lane/policy-151552.pt'
    s.configure()
    s.t.TrialEnv=RouteRecoveryEnv
    s.t.EVAL_SEEDS=(59101,59102,59103)
    s.EVAL_SEEDS=s.t.EVAL_SEEDS
    s.SOURCE_FILES=(*s.t.SOURCE_FILES,'env/traffic_route_recovery.py','utils/run_traffic_route_recovery.py')
    s.t.SOURCE_FILES=s.SOURCE_FILES

configure()
_original_publish=s.publish
def publish():
    _original_publish()
    text=s.DOC.read_text(encoding='utf8')
    text=text.replace('교통 정책 이어학습: 안전보상 독립 비교 및 조합 탐색','안전 기준점에서 길찾기 보상 추가 비교')
    text=text.replace('원본 교통정책20만 스텝 보존.', '출발 정책: lane/policy-151552.pt 고정·보존.')
    text=text.replace('39101~39103','59101~59103')
    s.DOC.write_text(text,encoding='utf8')
s.publish=publish
if __name__=='__main__':s.main()
