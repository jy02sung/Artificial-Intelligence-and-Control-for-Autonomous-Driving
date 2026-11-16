"""Independent safe route bonus and preserved lateral input, plus their combination."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['IC_RECOVERY_CONFIG'] = str(ROOT / 'results/traffic_recovery_auto_20261116/round-02/config.json')
from utils import run_traffic_recovery_auto as a
from env.safe_route_reward_trials import SafeRouteRewardEnv
from utils.policy_input_init import preserve_existing_inputs
s = a.s
t = s.t
s.OUT = ROOT / 'results/safe_route_reward_trials_20261116'
t.OUT = s.OUT
s.DOC = ROOT / 'docs/experiments/ic-safe-route-reward-results-20261116.md'
t.DOC = s.DOC
s.SOURCE = ROOT / 'results/traffic_recovery_auto_20261116/round-02/runs/route/policy-200000.pt'
s.CONDITIONS = ('safe_reward', 'lateral_zero', 'combined')
t.CONDITIONS = s.CONDITIONS
t.TrialEnv = SafeRouteRewardEnv
s.EVAL_SEEDS = (79101, 79102, 79103)
t.EVAL_SEEDS = s.EVAL_SEEDS
s.SOURCE_FILES = (*s.SOURCE_FILES, 'env/safe_route_reward_trials.py', 'utils/policy_input_init.py', 'utils/run_safe_route_reward_trials.py')
t.SOURCE_FILES = s.SOURCE_FILES
BasePPO = s.a.b.PPO


class PreservedLateralPPO(BasePPO):
    def load(self, path):
        result = super().load(path)
        if Path(path).resolve() == s.SOURCE.resolve():
            preserve_existing_inputs(self.policy, range(57, 60))
        return result


s.a.b.PPO = PreservedLateralPPO
_publish = s.publish


def publish():
    _publish()
    text = s.DOC.read_text(encoding='utf8').replace('59101~59103', '79101~79103').replace('lane/policy-151552.pt', 'round-02/route/policy-200000.pt').replace('안전 기준점에서 길찾기 보상 추가 비교', '위험한 변경의 경로 보상 지급 조건 비교')
    s.DOC.write_text(text, encoding='utf8')


s.publish = publish


def run_one(condition):
    assert condition in s.CONDITIONS
    manifest = t.read(s.OUT / 'manifest.json')
    for c in s.CONDITIONS:
        assert (s.OUT / 'smoke' / c / 'complete.json').exists()
        assert t.read(s.OUT / 'smoke' / c / 'manifest.json')['source_hashes'] == manifest['source_hashes']
    lock = s.OUT / (condition + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        print(s.train(condition), flush=True)
        publish()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--run-condition':
        run_one(sys.argv[2])
    else:
        s.main()
