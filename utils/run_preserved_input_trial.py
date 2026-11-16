"""Future navigation ablation: only newly activated input columns start at zero."""
from pathlib import Path
from utils import run_lane_followup as original
from utils.policy_input_init import preserve_existing_inputs

s = original.s
t = s.t
ROOT = original.ROOT
s.OUT = ROOT / 'results/preserved_input_trial_20261116'
t.OUT = s.OUT
s.DOC = ROOT / 'docs/experiments/ic-preserved-input-results-20261116.md'
t.DOC = s.DOC
s.CONDITIONS = ('future',)
t.CONDITIONS = s.CONDITIONS
s.SOURCE_FILES = (*s.SOURCE_FILES, 'utils/policy_input_init.py', 'utils/run_preserved_input_trial.py')
t.SOURCE_FILES = s.SOURCE_FILES
BasePPO = s.a.b.PPO


class PreservedInputPPO(BasePPO):
    def load(self, path):
        result = super().load(path)
        # Restore trained checkpoints verbatim; initialize only the original source.
        if Path(path).resolve() == s.SOURCE.resolve():
            preserve_existing_inputs(self.policy, range(47, 50))
        return result


s.a.b.PPO = PreservedInputPPO
_publish = s.publish


def publish():
    _publish()
    text = s.DOC.read_text(encoding='utf8').replace('늦은 차로변경 관측 보완 독립 비교', '새 관측의 행동 보존 초기화 비교')
    text += '\n새 입력47:50 연결만0으로 초기화했다. 출발 가중치 파일은 같지만 초기 정책 텐서 해시는 기존 future 조건과 다르다. 저장 모델 평가 시에는 초기화하지 않는다.\n'
    s.DOC.write_text(text, encoding='utf8')


s.publish = publish
if __name__ == '__main__':
    s.main()
