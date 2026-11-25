"""Matched high-traffic route-13/14 suffix pilot. Run from the repository root."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

os.environ.setdefault("IC_LIBSUMO", "1")
from utils import jys3_physical_forks as physical
from utils import jys3_pure_f as pure

j = physical.j
ROOT = j.m.ROOT / "results" / "jys3_fork1314_pilot"
FORK = "1063527748"
NEXT = {13: "169728835", 14: "797789364"}
MARKS = (0, 16384, 32768, 50000)
TRAIN_SEEDS = (11, 22, 33)


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    tmp.replace(path)


def append(path, data):
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(data) + "\n")


class Pilot:
    def __init__(self, folder, seed, training):
        self.env = physical.Env(physical.ARMS[0], folder / "map", seed)
        self.folder = folder
        self.training = training
        self.rng = random.Random(seed)
        self.seed = seed
        self.block = 0
        self.queue = []
        self.full_length = {}
        self.env.sim.ego_depart_speed = "10"
        self.env.sim.eval_mode = not training
        cfg = Path(self.env.sim.cfg_path)
        routes_file = cfg.parent / "highway.rou.xml"
        routes = ET.parse(routes_file)
        root = routes.getroot()
        first_route = list(root).index(root.find("route"))
        for route, dest in NEXT.items():
            full = self.env.routes[f"route_{route:02d}"]
            suffix = full[full.index(FORK):]
            assert suffix[1] == dest
            nav = self.env.navigation
            self.full_length[route] = sum(nav.lengths[e] for e in full) + sum(
                nav.link_lengths.get(pair, 0.) for pair in zip(full, full[1:]))
            alias = f"pilot_{route}"
            self.env.routes[alias] = suffix
            root.insert(first_route, ET.Element("route", id=alias, edges=" ".join(suffix)))
        routes.write(routes_file, encoding="utf8", xml_declaration=True)
        assert self.env.links[FORK][NEXT[13]] == {0, 1} and self.env.links[FORK][NEXT[14]] == {2}
        assert self.env.sim.max_steps == 1000 and self.env.sim.step_length == .2
        assert not self.env.lc_safety_enabled

    def reset(self, case=None):
        if case is None:
            if not self.queue:
                offset = 500000000 + self.seed * 1000000 + self.block * 3
                self.queue = [(r, l, offset+l) for l in range(3) for r in NEXT]
                self.rng.shuffle(self.queue)
                self.block += 1
            case = self.queue.pop()
        route, lane, seed = case
        env = self.env
        env.ego_depart_lane = str(lane)
        obs, info = env.reset(seed=seed, options={"route":f"pilot_{route}"})
        # The original environment divides progress by remaining episode length.
        # Rescale its cap so one metre always uses the full original route length.
        env.progress_cap = 2. * (env.total - env.start) / self.full_length[route]
        edge, actual_lane, path, index, *rest = env.snapshot
        assert (edge, actual_lane, index) == (FORK, lane, 0)
        assert path[1] == NEXT[route] and env.sim.route_intact
        assert env.sim.paid_forks == set() and math.isclose(env.high,env.start)
        assert obs.shape == (52,) and obs[34] != 0 and obs[-2:].tolist() == [0.,0.]
        vehicle = j.traci.vehicle
        ego = env.ego_id
        assert vehicle.getSpeedMode(ego) == vehicle.getLaneChangeMode(ego) == 0
        self.row = dict(route=route, lane=lane, traffic_seed=seed,
                        change_needed=lane not in env.links[FORK][NEXT[route]],
                        expected=NEXT[route], actual=None, crossing_lane=None,
                        left=0, right=0, changes=0, progress=0., fork_reward=0., direction=float(obs[34]))
        if not self.training:
            hv = [(vid,vehicle.getLaneID(vid),vehicle.getLanePosition(vid),vehicle.getSpeed(vid))
                  for vid in sorted(vehicle.getIDList()) if vid != ego]
            self.row["start"] = [edge, lane, vehicle.getLanePosition(ego),vehicle.getSpeed(ego),
                                 j.traci.simulation.getTime(),len(hv),hashlib.sha256(json.dumps(hv).encode()).hexdigest()]
        return obs, info

    def step(self, action):
        env = self.env
        before = env.snapshot
        obs, reward, done, cut, info = env.step(action)
        after = env.snapshot
        r = self.row
        if before[0] == FORK:
            r["left"] += int(info["lane_change"] > 0)
            r["right"] += int(info["lane_change"] < 0)
            if after and after[0] == FORK:
                r["changes"] += int(before[1] != after[1])
            self.last_fork_lane = before[1]
        if after and after[0] != FORK and not after[0].startswith(":") and r["actual"] is None:
            r["actual"] = after[0]
            r["crossing_lane"] = self.last_fork_lane
        r["progress"] += float(info["progress_bonus"])
        r["fork_reward"] += float(info["fork_passed"])
        if done or cut:
            r.update(outcome=info["outcome"], steps=env.sim.step_count)
            assert r["progress"] <= env.progress_cap + 1e-5 and r["fork_reward"] <= 1.
            if r["outcome"] == "off_route":
                assert r["actual"] != r["expected"] and info["off_route_detail"]["actual"] == r["actual"]
                assert reward == -5.
            if r["outcome"] == "arrival":
                assert r["actual"] == r["expected"] and r["fork_reward"] == 1.
            if self.training:
                append(self.folder/"episodes.jsonl", r)
        return obs,reward,done,cut,info


def panel(kind):
    base, repeats = (800000000, 3) if kind == "screen" else (900000000, 8)
    return [(route,lane,base+i*10+lane) for i in range(repeats) for lane in range(3) for route in NEXT]


def evaluate(folder, seed, mark, kind):
    model = pure.agent(seed)
    model.policy.load_state_dict(j.torch.load(folder/f"policy_{mark}.pt",weights_only=True))
    model.policy.eval()
    env = Pilot(folder/f"eval_{mark}_{kind}",seed,False)
    rows = []
    try:
        for case in panel(kind):
            obs,_ = env.reset(case)
            while True:
                obs,_,done,cut,_ = env.step(model.predict(obs, deterministic=True))
                if done or cut:
                    rows.append(env.row)
                    break
    finally:
        env.env.close()
    for a,b in zip(rows[::2],rows[1::2]):
        assert a["start"] == b["start"] and a["direction"] == -b["direction"], (a,b)
    write(folder/f"{kind}_{mark}.json",rows)


def train(folder, seed):
    folder.mkdir(parents=True,exist_ok=False)
    env=Pilot(folder,seed,True)
    model=pure.agent(seed)
    write(folder/"initial.json",dict(seed=seed,initial_policy_hash=physical.v3.v1.digest(model),
                                    ppo=j.HP, gamma=model.gamma,gae_lambda=model.gae_lambda,
                                    full_route_lengths=env.full_length))
    j.atomic_torch(model.policy.state_dict(),folder/"policy_0.pt")
    try:
        obs,_=env.reset()
        stats=[0.,0]
        for mark in MARKS[1:]:
            while model.num_timesteps < mark:
                obs,last,_=model.collect_rollout(env,obs,ep_stats=stats,steps=min(2048,mark-model.num_timesteps),
                                                 timeout_is_failure=True)
                adv,ret=model.buffer.compute_gae(last,model.gamma,model.gae_lambda)
                metrics=model.update(adv,ret)
                assert all(math.isfinite(float(v)) for v in metrics.values())
                append(folder/"updates.jsonl",dict(step=model.num_timesteps,**metrics))
            j.atomic_torch(model.policy.state_dict(),folder/f"policy_{mark}.pt")
    finally:
        env.env.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["train","screen","holdout","check"])
    parser.add_argument("seed",type=int,choices=TRAIN_SEEDS)
    parser.add_argument("mark",type=int,choices=MARKS,nargs="?",default=0)
    args=parser.parse_args()
    folder=ROOT/f"seed{args.seed}"
    if args.mode=="train":
        train(folder,args.seed)
    elif args.mode=="check":
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            test=Pilot(Path(tmp),args.seed,False)
            try:
                for lane in range(3):
                    states=[]
                    for route in NEXT:
                        test.reset((route,lane,700000000+lane))
                        states.append((test.row["start"],test.row["direction"]))
                        # Isolate lane connectivity in this preflight only, after recording traffic.
                        for vid in j.traci.vehicle.getIDList():
                            if vid != test.env.ego_id:
                                j.traci.vehicle.remove(vid)
                        for i in range(1000):
                            _,_,done,cut,_=test.step([1.,0.])
                            if done or cut:break
                        print(route,lane,test.row["actual"],test.row["outcome"],flush=True)
                        assert test.row["actual"] == (NEXT[14] if lane==2 else NEXT[13])
                    assert states[0][0]==states[1][0] and states[0][1]==-states[1][1]
            finally:test.env.close()
    else:
        evaluate(folder,args.seed,args.mark,args.mode)


