from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def simulator():
    from utils.driving_env import environment
    from utils import physical_forks as physical
    from utils.route_rewards import ARMS
    with TemporaryDirectory() as tmp:
        meta = environment('high_real_ratio',11,Path(tmp)/'map')
        env = physical.Env(ARMS[0],Path(tmp)/'physical',11)
        env.sim.signs,env.sim.road_flags=meta.signs,meta.road_flags
        try:
            yield env
        finally:
            env.close()
