"""Run the jys3 road-direction and front-TTC PPO comparison in isolated workers."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from utils import jys3_reward_mix as v1
from utils import jys3_pure_f as core
from utils.jys3_pure_f_report import aggregate

j = core.j
ROOT = v1.ROOT.parent / "reward_mix_v2"
ARMS = ("P1_dir_clean", "P2_dir_ttc")
MARKS = v1.MARKS
SCREEN, PAIRED, FRESH = v1.SCREEN, v1.PAIRED, v1.FRESH
read = v1.read
save = core.save

from collections import defaultdict
import xml.etree.ElementTree as ET


def fork_offsets(net, connections):
    """Rank road geometry using edge IDs and lane-shape averages only."""
    shapes = {}
    for edge in ET.parse(net).findall("edge"):
        if edge.get("function"):
            continue
        lanes = [[tuple(map(float, point.split(",")[:2])) for point in lane.get("shape").split()]
                 for lane in edge.findall("lane")]
        def center(i):
            return tuple(sum(lane[i][axis] for lane in lanes) / len(lanes) for axis in (0, 1))
        shapes[edge.get("id")] = center(0), center(-2), center(-1)
    exits = defaultdict(set)
    for link in connections:
        exits[link[0]].add(link[1])
    offsets = {}
    for src, destinations in exits.items():
        if len(destinations) < 2:
            continue
        _, (px, py), (x, y) = shapes[src]
        norm = math.hypot(x-px, y-py)
        if not norm:
            raise ValueError(f"Degenerate incoming road tangent: {src}")
        for dst in destinations:
            dx, dy = shapes[dst][0]
            offsets[src, dst] = (-(y-py)*(dx-x) + (x-px)*(dy-y)) / norm
    return offsets


def lateral_directions(net, connections):
    offsets = fork_offsets(net, connections)
    exits = defaultdict(list)
    for src, dst in offsets:
        exits[src].append(dst)
    signs = {}
    for src, destinations in exits.items():
        ordered = sorted(destinations, key=lambda dst: offsets[src, dst])
        for i, dst in enumerate(ordered):
            if i and math.isclose(offsets[src, dst], offsets[src, ordered[i-1]], abs_tol=1e-6):
                raise ValueError(f"Indistinguishable road branches: {src}")
            signs[src, dst] = 2*i/(len(ordered)-1)-1
    return signs


def configure_roads(sim, net):
    # The direction function sees only ordered pairs of roads, never the ego or a lane ID.
    pairs = [(src, dst) for src, exits in sim.links.items() for dst in exits]
    sim.signs = lateral_directions(net, pairs)
    flags = {}
    for src, exits in sim.links.items():
        slots = sim.no_links.copy().reshape(j.MAX_LANES, 2)
        for dst, lanes in exits.items():
            sign = sim.signs.get((src, dst))
            for lane in lanes:
                if sign is None:
                    slots[lane, :] = 1.
                else:
                    slots[lane, int(sign > 0)] = 1.
        flags[src] = slots.ravel()
    sim.road_flags = flags


def idle_clear(speed, gap, terminal):
    if not terminal and speed is not None and speed < .5 and (gap is None or gap > 25.):
        return -.03
    return 0.


def front_phi(gap, closing):
    if gap is None or closing <= .5:
        return 0.
    return -v1.risk(gap, closing, 2.5)


def ttc_bonus(previous, current, collided):
    return 0. if collided else .99 * current - previous


class Env(v1.BASE_ENV):
    progress_cap = 3.

    def __init__(self, arm, folder, seed):
        if arm not in ARMS:
            raise ValueError(f"Unknown v2 arm {arm}")
        super().__init__("F1", folder, seed)
        self.arm = arm
        self.sums = Counter()
        configure_roads(self.sim, folder / "high_real_ratio" / "network.net.xml")
        assert self.progress_enabled and not self.nav_enabled and self.sim.max_steps == 1000
        assert self.visibility == 120 and self.obs_noise and self.occlusion and not self.lc_safety_enabled

    def potential(self):
        if self.snapshot is None or self.snapshot[0].startswith(":"):
            return 0.
        leader = j.traci.vehicle.getLeader(self.ego_id, self.visibility)
        if not leader or not leader[0]:
            return 0.
        return front_phi(leader[1], self.snapshot[7] - j.traci.vehicle.getSpeed(leader[0]))

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.previous = self.potential() if self.arm == ARMS[1] else 0.
        assert len(obs) == 52 and tuple(obs[-2:]) == (0., 0.)
        return obs, info

    def step(self, action):
        obs, reward, done, cut, info = super().step(action)
        terminal = done or cut
        current = self.potential() if self.arm == ARMS[1] and not terminal else 0.
        pot = ttc_bonus(self.previous, current, info["collided"]) if self.arm == ARMS[1] else 0.
        self.previous = current
        parts = dict(progress=info["progress_bonus"], arrival=j.ARRIVAL_REWARD*info["arrived"],
                     fork=j.PASS_REWARD*info["fork_passed"], collision=j.COLLISION_REWARD*info["collided"],
                     timeout=info["timeout_reward"], idle_clear=idle_clear(info["speed"], info["gap"], terminal),
                     pot=pot)
        total = reward + parts["idle_clear"] + pot
        assert math.isclose(total, sum(parts.values()), abs_tol=1e-7)
        self.sums.update(parts)
        return obs, total, done, cut, info


def setup():
    core.FactorEnv = Env
    j.ARRIVAL_REWARD, j.PASS_REWARD = 5., 1.
    j.COLLISION_REWARD, j.TIMEOUT_REWARD = -9., -8.


def audit_map(env, folder):
    net = folder / "high_real_ratio" / "network.net.xml"
    root = ET.parse(net)
    # Independent ground truth for testing only: SUMO lane 0 is rightmost.
    oracle = defaultdict(lambda: defaultdict(set))
    edges = {e.get("id") for e in root.findall("edge") if not e.get("function")}
    for link in root.findall("connection"):
        src, dst = link.get("from"), link.get("to")
        if src in edges and dst in edges:
            oracle[src][dst].add(int(link.get("fromLane")))
    pairs = [(src, dst) for src in oracle for dst in oracle[src]]
    offsets = fork_offsets(net, pairs)
    old = j.directions(net, ((a,b,None,None) for a,b in pairs))
    forks = []
    flipped = set()
    expected = {}
    for src, exits in sorted(oracle.items()):
        if len(exits) == 1:
            continue
        assert len(exits) == 2
        right, left = sorted(exits, key=lambda dst: sum(exits[dst]) / len(exits[dst]))
        assert min(exits[right]) <= min(exits[left]) and max(exits[right]) < max(exits[left])
        expected[src, right], expected[src, left] = -1., 1.
        assert env.signs[src, right] == -1 and env.signs[src, left] == 1
        slots = env.road_flags[src].reshape(j.MAX_LANES, 2)
        for lane, (r, l) in enumerate(slots):
            assert bool(r) == (lane in exits[right]) and bool(l) == (lane in exits[left])
        changed = old[src, right] != -1.
        if changed:
            flipped.add(src)
        forks.append(dict(src=src, right=right, left=left, right_m=offsets[src, right],
                          left_m=offsets[src, left], right_lanes=sorted(exits[right]),
                          left_lanes=sorted(exits[left]), flipped=changed, slots_ok=True))
    assert {row["src"] for row in forks} == {
        "1063527748","169929488#0","170059994","52873356","544293888#0","781184662#1","86198728"}
    assert flipped == {"170059994", "1063527748", "169929488#0"}
    assert len(env.routes) == 16
    counts = Counter({"wrong": 0})
    for path in env.routes.values():
        for pair in zip(path, path[1:]):
            if pair in expected:
                counts["RIGHT" if expected[pair] < 0 else "LEFT"] += 1
                counts["wrong"] += int(env.signs[pair] != expected[pair])
    assert counts["wrong"] == 0
    return forks, dict(counts)


def check():
    setup()
    assert (j.TIMEOUT_REWARD,j.COLLISION_REWARD,j.PASS_REWARD,j.ARRIVAL_REWARD,
            Env.progress_cap,j.HP["lr"],j.STALL_SECONDS) == (-8.,-9.,1.,5.,3.,1e-4,30.)
    assert (j.HP["lane_entropy_coef"],j.HP["lane_keep_bias"]) == (.002,1.)
    assert [idle_clear(v,g,t) for v,g,t in [(0,None,False),(.49,26,False),(.49,25,False),
                                            (.5,30,False),(0,None,True)]] == [-.03,-.03,0.,0.,0.]
    assert front_phi(4,0) == front_phi(10,4) == front_phi(None,3) == 0.
    hazard = front_phi(2, 3)
    assert -1. <= hazard < 0 and ttc_bonus(0.,hazard,False) < 0
    assert ttc_bonus(0.,0.,False) == ttc_bonus(hazard,0.,True) == 0.
    assert ttc_bonus(hazard,0.,False) > 0
    with TemporaryDirectory() as temp:
        folder = Path(temp) / "geometry"
        env = Env(ARMS[0], folder, 11)
        try:
            forks, counts = audit_map(env, folder)
            cases = []
            for route, sign in ((13,-1.),(16,1.)):
                for lane in (0,1):
                    env.ego_depart_lane = str(lane)
                    obs, _ = env.reset(seed=111003, options={"route":f"route_{route:02d}"})
                    for _ in range(100):
                        assert env.snapshot[:2] == ("86198728",lane)
                        if env.snapshot[4] is not None:
                            break
                        obs, _, done, cut, _ = env.step([0., 0])
                        assert not (done or cut)
                    assert env.snapshot[4] == 0 and obs[34] == sign
                    assert (obs[40:50] == env.road_flags[env.snapshot[0]]).all()
                    cases.append(dict(route=route, edge=env.snapshot[0], lane=lane, traffic_seed=111003, remaining_m=float(env.snapshot[-1]), sign=float(obs[34])))
        finally:
            env.close()
        smoke = {}
        for arm in ARMS:
            initial, update = train_job(arm, 11, 2048, Path(temp))
            assert abs(update["approx_kl"]) < .1
            smoke[arm] = dict(initial_sha256=initial, **update)
        assert len({r["initial_sha256"] for r in smoke.values()}) == 1
    ROOT.mkdir(parents=True, exist_ok=True)
    core.save(ROOT/"preflight_checks.json",dict(forks=forks, route_sign_counts=counts, ego_cases=cases,
              smoke=smoke, reward_tests=True, side_action_used_in_potential=False))

def train_job(arm, seed, end, stage):
    setup()
    folder = stage / f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    model = core.agent(seed)
    env = Env(arm, folder / "map", seed)
    try:
        initial = v1.digest(model)
        save(folder / "initial.json", {"sha256": initial})
        assert model.gamma == .99 and model.gae_lambda == .95
        schedule = j.ScheduledEpisodes(env, seed, folder / "episodes.jsonl")
        obs, _ = schedule.reset()
        episode_stats = [0., 0]
        marks = [end] if end == 2048 else [step for step in MARKS if step <= end]
        for step in marks:
            while model.num_timesteps < step:
                env.sums.clear()
                batch = min(2048, step - model.num_timesteps)
                obs, last, _ = model.collect_rollout(schedule, obs, ep_stats=episode_stats,
                                                     steps=batch, timeout_is_failure=True)
                adv, ret = model.buffer.compute_gae(last, model.gamma, model.gae_lambda)
                metrics = model.update(adv, ret)
                assert all(math.isfinite(x) for x in metrics.values())
                lr = model.optimizer.param_groups[0]["lr"]
                assert lr == .0001
                record = dict(step=model.num_timesteps, batch=batch, lr=lr, rewards=dict(env.sums), **metrics)
                with (folder / "train_updates.jsonl").open("a", encoding="utf8") as f:
                    print(json.dumps(record), file=f)
            j.atomic_torch({"policy": model.policy.state_dict()}, folder / f"policy_{step}.pt")
        return initial, record
    finally:
        env.close()


def eval_job(stage, arm, seed, step, rep, panel):
    setup()
    lane_departures = {}
    class EvalEnv(Env):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self.eval_route = kwargs["options"]["route"]
            lane_departures[self.eval_route] = False
            return result

        def step(self, action):
            before = self.snapshot
            result = super().step(action)
            after = self.snapshot
            if (self.eval_route == "route_16" and before and after and before[3] == 0
                and before[0] == after[0] == "86198728" and before[1] == 1 and after[1] == 0):
                lane_departures[self.eval_route] = True
            return result
    core.FactorEnv, core.ROOT = EvalEnv, stage
    core.evaluate((arm, seed, step, rep, panel))
    path = stage / f"{arm}_{seed}" / f"eval_{panel}_{step}_{rep}.json"
    rows = read(path)
    assert [r["route"] for r in rows] == list(range(1, 17))
    for row in rows:
        if row["route"] == 16:
            row["r16_lane1_to0"] = lane_departures["route_16"]
    save(path, rows)


def snapshot_rows(stage, arm, seed, step, base):
    return v1.rows(stage, arm, seed, step, base)


def verify_rerun(arm, seed, step):
    prefix = f"{arm}_{seed}/policy_{step}.pt"
    old, new = [j.torch.load(ROOT / s / prefix, weights_only=True)["policy"] for s in ("stage1","stage2")]
    assert old.keys() == new.keys() and all(j.torch.equal(old[k], new[k]) for k in old), (arm,seed,step)


def stage_run(arms, seeds, end):
    import shutil
    stage = ROOT / ("stage1" if end == 300000 else "stage2")
    stage.mkdir(exist_ok=True)
    scheduled = set()
    runs = [(arm, seed) for arm in arms for seed in seeds]
    with ProcessPoolExecutor(max_workers=12) as pool:
        pending = set()
        for arm, seed in runs:
            folder = stage / f"{arm}_{seed}"
            if not (folder / f"policy_{end}.pt").exists():
                if folder.exists():
                    folder.rename(folder.with_name(folder.name + ".interrupted-" + datetime.now().strftime("%Y%m%d%H%M%S")))
                pending.add(pool.submit(train_job, arm, seed, end, stage))
        while True:
            for n in MARKS:
                if n > end:
                    break
                for arm, seed in runs:
                    folder = stage / f"{arm}_{seed}"
                    key = arm, seed, n
                    if key in scheduled or not (folder / f"policy_{n}.pt").exists():
                        continue
                    replay = stage.name == "stage2" and seed in (11,22) and n <= 300000
                    if replay:
                        verify_rerun(arm, seed, n)
                    for rep in range(4):
                        name = f"eval_{SCREEN}_{n}_{rep}.json"
                        if replay:
                            shutil.copyfile(ROOT / "stage1" / folder.name / name, folder / name)
                        elif not (folder / name).exists() or "r16_lane1_to0" not in read(folder / name)[15]:
                            pending.add(pool.submit(eval_job, stage, arm, seed, n, rep, SCREEN))
                    scheduled.add(key)
            if not pending:
                if len(scheduled) == len(runs) * (end // 50000):
                    break
                raise RuntimeError("Expected policy checkpoint missing after training")
            done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for f in done:
                f.result()
        screen = {f"{a}_{s}_{n}": aggregate(snapshot_rows(stage,a,s,n,SCREEN))
                  for a,s in runs for n in MARKS if n <= end}
        save(stage / "screen.json", screen)
        if stage.name == "stage1":
            for s in seeds:
                assert len({read(stage/f"{a}_{s}"/"initial.json")["sha256"] for a in arms}) == 1
            ranking = {}
            for a in arms:
                arrivals = []
                counts = Counter()
                passes = 0
                for s in seeds:
                    group = [r for n in (200000,250000,300000) for r in snapshot_rows(stage,a,s,n,SCREEN)]
                    c = Counter(r["outcome"] for r in group)
                    counts.update(c)
                    arrivals.append(c["arrival"])
                    passes += sum((r["route"] == 13 and "169728835" in r["forks"]) or
                                  (r["route"] == 16 and "86198725" in r["forks"]) for r in group)
                ranking[a] = [sum(arrivals), min(arrivals), -counts["collision"], -counts["timeout"], passes]
            best = max(arms, key=ranking.get)
            save(ROOT/"stage1_summary.json", dict(promoted=[best],ranking=ranking,screen=screen))
            return best
        a, = arms
        def score(n):
            counts = Counter()
            for s in seeds:
                counts.update(screen[f"{a}_{s}_{n}"]["outcomes"])
            return counts["arrival"], -counts["collision"], -n
        best = max(MARKS, key=score)
        save(ROOT/"selection.json", {a:best})
        jobs = []
        for n in sorted({best,end}):
            for b in (PAIRED,FRESH):
                for s in seeds:
                    for r in range(4):
                        out = stage/f"{a}_{s}"/f"eval_{b}_{n}_{r}.json"
                        if not out.exists() or "r16_lane1_to0" not in read(out)[15]:
                            jobs.append(pool.submit(eval_job,stage,a,s,n,r,b))
        for f in jobs:
            f.result()
    return best


def protocol():
    baseline = read(v1.ROOT / "protocol.json")
    return dict(base_commit="8343fb712578eca5193c75e93052a419d9f9cd20", arms=ARMS,
                environment=baseline["environment"], ppo=dict(j.HP,gamma=.99,gae_lambda=.95,initialization="random"),
                panels=baseline["panels"], fresh_note="held out from v2 training and selection; also used for v1 final testing",
                rewards=dict(progress_cap=3,arrival=5,fork=1,collision=-9,timeout=-8,stall=-8,idle_clear=-.03),
                P2=dict(front_ttc_s=2.5,minimum_closing_mps=.5,gamma=.99,scale=1,collision_pot=0),
                stage1=dict(seeds=[11,22],steps=300000,interval=50000,rank="lexicographic: total arrivals, min per-seed arrivals, fewest collisions, fewest ordinary timeouts, most route 13 exit and 16 main passes across both starts; 200/250/300k pooled",promote=1),
                stage2=dict(seeds=[11,22,33],steps=600000,interval=50000,
                            replay="11 and 22 rerun from zero; verify six checkpoint tensors, then reuse identical stage1 screening drives",
                            rank="most arrivals, fewest collisions, earliest step"))



# The report reads archived trajectories so every table can be reproduced offline.
OUTCOME_NAMES = {"arrival": "완주", "collision": "충돌", "stall": "끝 정지",
                 "timeout": "일반 시간초과", "other": "기타"}


def report_drives(stage, arm, seed, step, panel):
    """Read a complete 64-drive panel or return None while evaluation is running."""
    paths = [stage / f"{arm}_{seed}" / f"eval_{panel}_{step}_{i}.json" for i in range(4)]
    if not all(p.is_file() for p in paths):
        return None
    drives = [drive for path in paths for drive in json.loads(path.read_bytes())]
    assert len(drives) == 64
    assert {(r["route"], r["case"]) for r in drives} == {
        (route, case) for route in range(1, 17) for case in range(4)}
    assert all(r["traffic_seed"] == panel + 100 * (r["case"] // 2) + r["route"] for r in drives)
    return drives


def road_results(drives):
    counts = Counter(r["outcome"] for r in drives)
    assert set(counts) <= OUTCOME_NAMES.keys()
    r16 = [r for r in drives if r["route"] == 16]
    n = len(drives)
    passed = [sum(r["start"][1] == lane and "86198725" in r["forks"] for r in r16) for lane in (0, 1)]
    fail16 = sum(r["start"][1] == 1 and r["r16_lane1_to0"] and not r["forks"]
                 and r["outcome"] == "stall" for r in r16)
    routes = [sum(r["route"] == route and target in r["forks"] for r in drives)
              for route, target in [(13, "169728835"), (14, "797789364")]]
    return dict(n=n, outcomes=counts, moves=sum(r["moves"] for r in drives)/n,
                passed16=passed, fail16=fail16, routes=routes)


def road_result_row(arm, step, label, result):
    r = result
    c = r["outcomes"]
    rates = [f"{c[k]} ({c[k]/r['n']:.1%})" for k in ("arrival", "collision", "stall", "timeout")]
    return (f"|{arm}|{step:,}|{label}|{r['n']}|" + "|".join(rates)
            + f"|{c['other']}|{r['moves']:.2f}|{r['routes'][0]}/{r['n']//16}|{r['routes'][1]}/{r['n']//16}|"
            + "|".join(f"{v}/{r['n']//32}" for v in [*r["passed16"], r["fail16"]]) + "|")


def road_table(stage, arms, seeds, steps, panel):
    lines = ["|조건|스텝|학습 시드|주행 수|완주|충돌|끝 정지|일반 시간초과|기타|실제 변경/주행|13번 진출|14번 통과|16번 0차로 복귀 통과|16번 1차로 본선 통과|16번 1→0 후 끝 정지|",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    series = {}
    for arm in arms:
        for step in steps:
            combined = []
            for seed in seeds:
                drives = report_drives(stage, arm, seed, step, panel)
                if drives is None or any(r['route'] == 16 and 'r16_lane1_to0' not in r for r in drives):
                    continue
                combined.extend(drives)
                r = road_results(drives)
                lines.append(road_result_row(arm, step, seed, r))
                series[arm, step, seed] = r
            if len(combined) == len(seeds)*64:
                r = road_results(combined)
                lines.append(road_result_row(arm, step, "합산", r))
                series[arm, step, "합산"] = r
    return (lines if series else ["완료된 평가 시드 묶음을 기다리고 있다."]), series


def render_screen_svg(series, arm, path):
    """One panel per seed and a pooled panel; shared axes in percentages."""
    from xml.sax.saxutils import escape
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="610" viewBox="0 0 960 610" role="img">',
           '<title>고교통 중간 평가 추이</title><rect width="960" height="610" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#17232a">',
           f'<text x="20" y="27" font-size="18">{escape(arm)} · 고교통 중간 평가 (%)</text>']
    outcomes = [("arrival", "#087758"), ("collision", "#b43a2c"),
                ("stall", "#315fa4"), ("timeout", "#82652b")]
    for i, (key, color) in enumerate(outcomes):
        x = 25 + 162*i
        svg.extend([f'<path d="M{x} 50h28" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{x+36}" y="55">{OUTCOME_NAMES[key]}</text>'])
    for i, seed in enumerate([11, 22, 33, "합산"]):
        x, y = 62 + (i%2)*473, 108 + (i//2)*254
        w, h = 373, 169
        svg.append(f'<text x="{x}" y="{y-11}" font-weight="bold">{"세 시드 합산" if seed=="합산" else "학습 시드 "+str(seed)}</text>')
        for pct in (0, 25, 50, 75, 100):
            yy = y + h * (1-pct/100)
            svg.extend([f'<path d="M{x} {yy}h{w}" stroke="#cdd9df"/>',
                        f'<text x="{x-7}" y="{yy+4}" text-anchor="end">{pct}</text>'])
        for n in (50000, 200000, 400000, 600000):
            xx = x + w*(n-50000)/550000
            svg.append(f'<text x="{xx}" y="{y+h+19}" text-anchor="middle">{n//10000}만</text>')
        records = sorted((step, data) for (a, step, s), data in series.items() if a==arm and s==seed)
        for key, color in outcomes:
            points = [(x+w*(step-50000)/550000, y+h*(1-r["outcomes"][key]/r["n"])) for step,r in records]
            coords = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
            svg.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="2.4" fill="none"/>')
            svg.extend(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.6" fill="{color}"/>' for xx,yy in points)
    svg.append('<text x="480" y="596" text-anchor="middle">PPO 학습 스텝</text></g></svg>')
    path.write_text("\n".join(svg), encoding="utf8")


def report_matching(stage, arm, best_step):
    base = ROOT.parent / "terminal_ratio" / "long_600k_4"
    old_mix = ROOT.parent / "reward_mix_v1"
    lines = ["### 동일한 192회 주행에서 종료 원인이 바뀐 횟수", "",
             "43110000 시드군에서 학습 시드·경로·출발 조건·교통 시드·실제 출발 도로·차로·속도·잔여거리를 대조했다. 각 행은 기존 종료별 건수다.", "",
             "|기존 모델|시점|기존→후보 스텝|기존 종료|완주로|충돌로|끝 정지로|일반 시간초과로|기타로|",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for old, name, folder, selection in [
        ("R3_pot_safe", "R3", old_mix/"stage2", old_mix),
        ("collision5", "충돌 −5", base, base), ("F1", "F1", base, base)]:
        old_best = json.loads((selection / "selection.json").read_bytes())[old]
        for label, left, right in [("대표", old_best, best_step), ("60만", 600000, 600000)]:
            counts = Counter()
            for seed in (11, 22, 33):
                previous = report_drives(folder, old, seed, left, PAIRED)
                current = report_drives(stage, arm, seed, right, PAIRED)
                assert previous is not None and current is not None
                def key(r):
                    return r["case"], r["route"], r["traffic_seed"], tuple(r["start"])
                indexed = {key(r): r for r in previous}
                assert set(indexed) == {key(r) for r in current} and len(indexed) == 64
                counts.update((indexed[key(r)]["outcome"], r["outcome"]) for r in current)
            assert sum(counts.values()) == 192
            for outcome, description in OUTCOME_NAMES.items():
                row = [counts[outcome, target] for target in OUTCOME_NAMES]
                if sum(row):
                    lines.append(f"|{name}|{label}|{left:,}→{right:,}|{description}|"+"|".join(map(str,row))+"|")
    return lines


def report():
    """Regenerate the v2 results and report only completed panels."""
    doc = Path(__file__).resolve().parents[1] / "docs/experiments/jys3-reward-mix-v2.md"
    heading = "## 측정 결과"
    prefix = doc.read_text(encoding="utf-8-sig").split(heading, 1)[0].rstrip()
    lines = [prefix, "", heading, ""]
    checks = ROOT / "preflight_checks.json"
    if checks.exists():
        checked = json.loads(checks.read_bytes())
        lines += ["### 분기 7곳과 차로 연결 슬롯", "",
                  "오프셋은 진입 도로 끝 중심점 기준 왼쪽을 양수로 잡은 거리(m)다. 두 도로 사이의 상대 순서로 부호를 정한다.", "",
                  "|진입 도로|오른쪽 도로 (−1)|좌측 오프셋(m)|출발 차로|왼쪽 도로 (+1)|좌측 오프셋(m)|출발 차로|이전 부호 변경|슬롯 일치|",
                  "|---|---|---:|---|---|---:|---|---|---|"]
        for f in checked["forks"]:
            lines.append(f"|{f['src']}|{f['right']}|{f['right_m']:.2f}|{f['right_lanes']}|{f['left']}|{f['left_m']:.2f}|{f['left_lanes']}|{'반전' if f['flipped'] else '유지'}|{'일치' if f['slots_ok'] else '불일치'}|")
        lines += ["", "16개 경로에서 분기별 좌우 불일치 " + str(checked["route_sign_counts"]["wrong"]) + "건. "
                  + " · ".join(f"{e['route']}번 {e['lane']}차로 {e['sign']:+.0f}" for e in checked["ego_cases"])
                  + ". [사전검증 기록](../../results/jys3_ppo_comparison/pure_f/reward_mix_v2/preflight_checks.json)", ""]
    lines += ["### 1단계 중간 평가", "", "13번·14번은 두 번째 목표 분기의 통과를 센다. 16번은 첫 본선 통과를 출발 차로별로 센다. 마지막 열은 1차로 출발 후 첫 분기 전에 1→0으로 바뀌어 첫 분기 전에 끝 정지한 횟수다. 실제 변경은 동일 도로에서 관찰한 차로변경의 주행당 평균이다.", ""]
    table, stage1 = road_table(ROOT/"stage1", ARMS, (11, 22), range(50000,300001,50000), SCREEN)
    lines += table
    summary = ROOT/"stage1_summary.json"
    if summary.exists():
        selected = json.loads(summary.read_bytes())["promoted"]
        lines += ["", "#### 승격 판단", "",
                  "|조건|20·25·30만 합산 완주 / 384|낮은 시드의 완주 / 192|충돌|일반 시간초과|13번+16번 통과|시드 22의 30만 실제 변경/주행|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            late = [stage1[a,n,"합산"] for n in (200000,250000,300000)]
            worst = min(sum(stage1[a,n,s]["outcomes"]["arrival"] for n in (200000,250000,300000)) for s in (11,22))
            counts = sum((r["outcomes"] for r in late), Counter())
            junctions = sum(r["routes"][0] + sum(r["passed16"]) for r in late)
            lines.append(f"|{a}|{counts['arrival']}|{worst}|{counts['collision']}|{counts['timeout']}|{junctions}|{stage1[a,300000,22]['moves']:.2f}|")
        lines += ["", "사전 규칙으로 승격한 조건은 **"+", ".join(selected)+"**이다. [판정 기록](../../results/jys3_ppo_comparison/pure_f/reward_mix_v2/stage1_summary.json)"]
        for a in selected:
            lines += ["", "### 2단계 고교통 중간 평가", ""]
            table, screen = road_table(ROOT/"stage2", [a], (11,22,33), MARKS, SCREEN)
            lines += table
            if screen:
                graph = ROOT/f"{a}_screen.svg"
                render_screen_svg(screen, a, graph)
                lines += ["", f"![선정 조건 시드별 주행 추이](../../results/jys3_ppo_comparison/pure_f/reward_mix_v2/{graph.name})"]
            selection = ROOT/"selection.json"
            if not selection.exists():
                continue
            best = json.loads(selection.read_bytes())[a]
            lines += ["", f"대표는 **{best:,}스텝**이다. 3개 시드 합산 중간 평가에서 완주 최다, 충돌 최소, 이른 시점 순으로 골랐다."]
            finals = {}
            for panel in (PAIRED, FRESH):
                lines += ["", f"### 최종 검증 · {panel} 시드군", ""]
                table, final = road_table(ROOT/"stage2", [a], (11,22,33), sorted({best,600000}), panel)
                lines += table
                finals[panel] = final
            if all((a, n, "합산") in finals[b] for b in (PAIRED,FRESH) for n in (best,600000)):
                lines += ["", *report_matching(ROOT/"stage2",a,best), "", "### 60만 이후 판단", ""]
                for panel in (PAIRED, FRESH):
                    c = finals[panel][a,600000,"합산"]["outcomes"]
                    lines.append(f"- {panel}에서 60만 모델은 완주 {c['arrival']}/192 ({c['arrival']/192:.1%}), 충돌 {c['collision']}/192 ({c['collision']/192:.1%}), 끝 정지 {c['stall']}건, 일반 시간초과 {c['timeout']}건이다.")
                goal = all((finals[b][a,600000,"합산"]["outcomes"]["arrival"]/192 >= .95 and
                            finals[b][a,600000,"합산"]["outcomes"]["collision"]/192 < .05) for b in (PAIRED,FRESH))
                late = [screen[a,n,"합산"]["outcomes"]["arrival"] for n in (500000,550000,600000)]
                reason = ("두 검증에서 목표에 도달해 60만 모델을 동결한다." if goal
                          else "세 시점 모두 완주가 늘었다. 후속 시험에서는 같은 설정의 연장 학습을 비교할 근거가 있다." if late[0] < late[1] < late[2]
                          else "후반 세 시점에서 완주가 연속으로 늘지 않아 60만에서 동결한다. 후속 설계는 16번 출발 차로별 정지와 기존 완주가 실패로 바뀐 사례를 기준으로 검토한다.")
                lines += ["", "50만·55만·60만의 중간 평가 완주는 각각 "+"·".join(map(str,late))+"건이다. "+reason]
    temp = doc.with_suffix(".md.partial")
    temp.write_text("\n".join(lines)+"\n", encoding="utf8")
    temp.replace(doc)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    for flag in ("check","stage1","stage2","report","auto"):
        task.add_argument("--"+flag, action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.report:
        report()
        return
    import msvcrt
    with (ROOT/"manager.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if args.check or args.auto:
            save(ROOT/"protocol.json", protocol())
            check()
        if args.stage1 or args.stage2 or args.auto:
            assert (ROOT/"preflight_checks.json").is_file()
        if args.stage1 or args.auto:
            stage_run(ARMS,(11,22),300000)
        if args.auto or args.stage2:
            promoted = read(ROOT/"stage1_summary.json")["promoted"]
            if args.stage2 and args.arms != promoted:
                parser.error("Use the promoted arm from stage1_summary.json")
            stage_run(promoted,(11,22,33),600000)
        if args.stage1 or args.stage2 or args.auto:
            from utils.jys3_reward_mix_v2 import report as write_report
            write_report()


if __name__ == "__main__":
    main()
