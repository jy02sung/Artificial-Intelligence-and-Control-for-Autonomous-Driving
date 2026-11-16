"""C observes existing bounded merge features; actions/rewards remain unchanged."""
from types import MethodType
from utils.curriculum_data import s
from utils.curriculum_env import CurriculumEnv
from env.junction_lookahead_trials import JunctionLookaheadEnv

def enable_merge_observation(env):
    assert env.visibility==50
    env.planning_horizon=150
    env.conflict_condition='observation'
    env.conflict_features=MethodType(JunctionLookaheadEnv.conflict_features,env)
    return env

def make_merge_env(evaluation=False,none=False):
    return enable_merge_observation(s.make_env('control',evaluation=evaluation,none=none))

class MergeCurriculumEnv(CurriculumEnv):
    def __init__(self,seed):
        super().__init__(seed)
        enable_merge_observation(self.env)
