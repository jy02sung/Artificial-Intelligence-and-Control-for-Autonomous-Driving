"""Paired replay of the frozen mcw2 80-case panel. Run as a script from jys3."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

COMMIT = "3dee85f1e3f8e857158d0b8edd69e0e8d0000c39"
SHA = "4dab15658aa3f85880df9165d655f932be3e092c4997758e515b9cb6092e4f7e"
REFERENCE = Path("results/mcw_runs/eval_mcw2_occlusion_12288_80ep")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results/jys3_mcw_physical_holdout"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


class LaneDrivenForks:
    """Keep mission-relative navigation while setting only SUMO's executable route."""

    def __init__(self, env, vehicle, types):
        self.env = env
        self.vehicle = vehicle
        self.types = types
        self.ego = env.ego_id
        self.mission = env.routes[env.episode_route]
        self.index = 0
        self.offset = 0
        self.alternate = None
        self.switches = 0
        self.offset_steps = 0
        self.branch_lane = None
        self.actual_index = vehicle.getRouteIndex
        self.paths = {}
        for route in env.routes.values():
            for i, pair in enumerate(zip(route, route[1:])):
                if len(route) - i > len(self.paths.get(pair, ())):
                    self.paths[pair] = route[i:]

    def mission_index(self, vehicle_id):
        raw = self.actual_index(vehicle_id)
        return raw + self.offset if vehicle_id == self.ego else raw

    def choose_route(self):
        edge = self.vehicle.getRoadID(self.ego)
        exits = self.env.links.get(edge, {})
        if len(exits) < 2 or self.index + 1 == len(self.mission):
            return
        assert edge == self.mission[self.index]
        lane = self.vehicle.getLaneIndex(self.ego)
        expected = self.mission[self.index + 1]
        choices = [dst for dst, lanes in exits.items() if lane in lanes]
        alternate = choices[0] if choices and expected not in choices else None
        self.branch_lane = lane
        if alternate != self.alternate:
            path = self.paths.get((edge, alternate), [edge, alternate]) if alternate else self.mission[self.index:]
            self.vehicle.setRoute(self.ego, path)
            self.offset = self.index - self.actual_index(self.ego)
            self.alternate = alternate
            self.switches += 1
        if self.offset:
            self.offset_steps += 1

    def observe_edge(self):
        edge = self.vehicle.getRoadID(self.ego)
        src = self.mission[self.index]
        if edge == src or edge.startswith(":"):
            return None
        runtime = self.vehicle.getRoute(self.ego)
        end = self.actual_index(self.ego)
        # Include sub-step external edges, using the executed route to verify every hop.
        start = end - (self.mission_index(self.ego) - self.index)
        assert 0 <= start < end and runtime[start] == src and runtime[end] == edge
        for actual in runtime[start + 1:end + 1]:
            expected = self.mission[self.index + 1]
            if actual != expected:
                kinds = self.types.get(expected), self.types.get(actual)
                reason = {("highway.motorway_link", "highway.motorway"): "missed_exit",
                          ("highway.motorway", "highway.motorway_link"): "unwanted_exit"}
                return dict(source=self.mission[self.index], expected=expected, actual=actual,
                            observed_edge=edge, lane=self.branch_lane, kind=reason.get(kinds, "other_external"))
            self.index += 1
        self.alternate = None
        self.branch_lane = None
        return None


def drive(m, env, agent, row, corrected, types):
    obs, _ = env.reset(seed=row["seed"], options={"route": row["route"]})
    v = m.traci.vehicle
    ego = env.ego_id
    assert env.ego_depart_lane == "best" and v.getLaneChangeMode(ego) == v.getSpeedMode(ego) == 0
    assert obs.shape == (47,) and tuple(v.getRoute(ego)) == tuple(env.routes[row["route"]])
    start = dict(obs_sha256=digest(obs.tobytes()), edge=v.getRoadID(ego), lane=v.getLaneIndex(ego),
                 speed=v.getSpeed(ego), position=v.getLanePosition(ego), time=m.traci.simulation.getTime())
    fork = LaneDrivenForks(env, v, types) if corrected else None
    speeds = []
    detail = None
    for step in range(1, env.max_steps + 1):
        action = agent.predict(obs, deterministic=True)
        if step == 1:
            start["action"] = action.tolist()
        if fork:
            fork.choose_route()
        # All native mcw2 navigation calls read the original mission index. SUMO
        # continues to execute the lane-selected route during simulationStep().
        with patch.object(v, "getRouteIndex", fork.mission_index) if fork else nullcontext():
            obs, _, done, cut, info = env.step(action)
        assert info["policy_action_raw"] == info["environment_action_raw"] == action.tolist()
        if info["speed"] is not None:
            speeds.append(info["speed"])
        if fork and not info["collided"] and ego in v.getIDList():
            assert ego not in m.traci.simulation.getStartingTeleportIDList()
            assert ego not in m.traci.simulation.getEndingTeleportIDList()
            detail = fork.observe_edge()
        if done or cut or detail:
            outcome = ("collision" if info["collided"] else "off_route" if detail else
                       "arrival" if info["arrived"] else "timeout" if cut else "other")
            if fork and outcome == "arrival":
                assert fork.index == len(fork.mission) - 1
            return dict(start=start, outcome=outcome, steps=step,
                        mean_speed=float(m.np.mean(speeds)) if speeds else 0.,
                        route_switches=fork.switches if fork else 0,
                        index_offset_fork_steps=fork.offset_steps if fork else 0, off_route=detail)
    raise AssertionError("episode exceeded its configured time limit")


def batch(job):
    source, rows = job
    sys.path.insert(0, str(source))
    os.environ["IC_LIBSUMO"] = "1"
    from utils import mcw_train as m
    assert Path(m.__file__).resolve().parents[1] == source
    m.torch.set_num_threads(1)
    manifest = read(source / REFERENCE / "manifest.json")
    args = SimpleNamespace(**manifest["args"])
    model = source / args.resume
    assert digest(model.read_bytes()) == SHA
    agent = m.PPO(obs_dim=47, act_dim=2, buffer=m.RolloutBuffer(), **(manifest["hparams"] | {"device": "cpu"}))
    agent.load(str(model))
    agent.policy.eval()
    with tempfile.TemporaryDirectory() as temp:
        cfg = m.prepare_scenario(args.stage, Path(temp), args.color_by_type, args.gui_delay)
        net = cfg.parent / "network.net.xml"
        routes = cfg.parent / "highway.rou.xml"
        road, nav = m.load_network_metadata(net, routes, args.nav_horizon)
        types = {e.get("id"): e.get("type") for e in ET.parse(net).findall("edge")}
        configs = m.build_configs(args)
        assert json.loads(json.dumps(configs[0])) == manifest["action"] and configs[1] == manifest["reward"]
        env = m.make_env(cfg, road, nav, args, *configs)
        env.eval_mode = True
        assert args.stage == "high_real_ratio" and env.use_route_conn and env.agenda2 and env.occlusion
        assert env.obs_noise and env.visibility == 120 and not env.lc_safety_enabled
        assert env.occ_truck_rate == env.occ_bus_rate == 1 and env.occ_delivery_rate == 0.5
        assert not env.cross_lane and not env.scripted and env.lc_reverse_cooldown == 0
        assert env.max_steps == 3000 and env.step_length == .2 and env.warmup_steps == 300
        assert env.ego_depart_speed == "max" and env.use_ttc_reward and env.use_prep_reward
        results = []
        try:
            for row in rows:
                old = drive(m, env, agent, row, False, types)
                new = drive(m, env, agent, row, True, types)
                assert old["start"] == new["start"], row
                results.append(dict(route=row["route"], seed=row["seed"], published=row, old=old, new=new))
        finally:
            env.close()
        return dict(rows=results, sumo=m.traci.getVersion(), torch=str(m.torch.__version__),
                    map_sha256=digest(net.read_bytes()), traffic_sha256=digest(routes.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcw-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.mcw_root.resolve()
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert rev == COMMIT, rev
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True)
    original = read(source / REFERENCE / "eval_16routes.json")["rows"]
    assert [(r["route"], r["seed"]) for r in original] == [
        (f"route_{r:02d}", 40000 + rep * 100 + r) for r in range(1,17) for rep in range(5)]
    jobs = [(source, original[i:i+5]) for i in range(0, 80, 5)]
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        batches = list(pool.map(batch, jobs))
    rows = [r for result in batches for r in result["rows"]]
    metadata = {k: v for k, v in batches[0].items() if k != "rows"}
    assert all(all(b[k] == v for k, v in metadata.items()) for b in batches)
    exact = sum((r["old"]["steps"], r["old"]["outcome"]) ==
                (r["published"]["steps"], r["published"]["outcome"]) for r in rows)
    counts = {cond: dict(Counter(r[cond]["outcome"] for r in rows)) for cond in ("old", "new")}
    report = dict(commit=rev, weights_sha256=SHA, **metadata, workers=args.workers, exact=exact,
                  matched_starts=len(rows), counts=counts, rows=rows)
    temp = args.out / "paired.tmp"
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    temp.replace(args.out / "paired.json")
    print(exact, counts, flush=True)


if __name__ == "__main__":
    main()
