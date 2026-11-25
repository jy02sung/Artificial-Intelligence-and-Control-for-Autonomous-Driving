"""Run the two fork-reward arms on a team member's machine, using paired road cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
from utils import jys3_terminal_long as long

# Windows workers import this module during spawn, so configure them at import time.
long.ARMS=("collision5","collision5_fork1")
long.ROOT=long.ratio.ROOT/"fork_reward_team_600k"
long.STRICT_REFERENCE=False

def check(arm):
    long.ratio.setup(arm)
    expected=long.ratio.params(arm)
    j=long.e.j
    assert (j.ARRIVAL_REWARD,j.COLLISION_REWARD,j.PASS_REWARD)==expected
    for seed in long.SEEDS:
        reference=j.torch.load(long.reference(arm,seed),weights_only=True)["policy"]
        model=long.e.agent(seed)
        model.policy.load_state_dict(reference)
    model=long.e.agent(11)
    digest=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()
    with TemporaryDirectory() as tmp:
        env=long.ratio.Env(arm,Path(tmp),11)
        try:
            obs,_=env.reset(seed=41110016,options={"route":"route_16"})
            assert len(obs)==52
            _,_,_,_,info=env.step(model.predict(obs,deterministic=True))
            assert "progress_bonus" in info
        finally:env.close()
    return arm,expected,digest

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Check SUMO, reference weights, rewards and identical starts.")
    args=parser.parse_args()
    with ProcessPoolExecutor(2) as pool:
        checked=list(pool.map(check,long.ARMS))
    assert checked[0][2]==checked[1][2]
    for entry in checked:print(*entry,flush=True)
    if not args.check:long.main()
