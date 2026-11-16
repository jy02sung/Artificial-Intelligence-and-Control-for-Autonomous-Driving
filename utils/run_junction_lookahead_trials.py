"""Same observation and reward; only along-route conflict lookahead changes."""
import os
import sys
from pathlib import Path
from utils import run_junction_conflict_trials as original
from env.junction_lookahead_trials import JunctionLookaheadEnv
from utils.policy_input_init import preserve_existing_inputs
s = original.s
t = s.t
s.OUT = s.ROOT / 'results/junction_lookahead_trials_20261116'
t.OUT = s.OUT
s.DOC = s.ROOT / 'docs/experiments/ic-junction-lookahead-results-20261116.md'
t.DOC = s.DOC
s.CONDITIONS = ('near50', 'ahead150')
t.CONDITIONS = s.CONDITIONS
t.TrialEnv = JunctionLookaheadEnv
s.SOURCE_FILES = (*s.SOURCE_FILES, 'env/junction_lookahead_trials.py', 'utils/policy_input_init.py', 'utils/run_junction_lookahead_trials.py')
t.SOURCE_FILES = s.SOURCE_FILES
BasePPO = s.a.b.PPO


class PreservedConflictPPO(BasePPO):
    def load(self, path):
        result = super().load(path)
        if Path(path).resolve() == s.SOURCE.resolve():
            preserve_existing_inputs(self.policy, range(60, 63))
        return result


s.a.b.PPO = PreservedConflictPPO
_publish = s.publish


def publish():
    _publish()
    text = s.DOC.read_text(encoding='utf8').replace('교차 충돌 관측·보상 독립 및 조합 실험', '합류 검색 거리50m/150m 독립 비교')
    text += '\n두 조건의 물리 감지거리는50m로 동일하다. 새 입력60:63 연결만0으로 초기화했다.\n'
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
