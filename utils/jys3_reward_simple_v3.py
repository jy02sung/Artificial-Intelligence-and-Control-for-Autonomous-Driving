"""Pure PPO under simple task rewards, matched route observations and actual road departures."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

# Use SUMO's in-process TraCI backend in the manager, smoke runs and spawned workers.
os.environ.setdefault("IC_LIBSUMO", "1")
from utils import jys3_reward_mix_v2 as v2

j, core, v1 = v2.j, v2.core, v2.v1
ROOT = v2.ROOT.parent / "reward_simple_v3"
ARMS = ("S1_dir_col5","S2_dir_col7")
MARKS = v2.MARKS
SCREEN, PAIRED, FRESH, NOVEL = v2.SCREEN, v2.PAIRED, v2.FRESH, 77110000
save, read = v2.save, v2.read


from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import xml.etree.ElementTree as ET


def setup(arm):
    j.ARRIVAL_REWARD, j.PASS_REWARD, j.TIMEOUT_REWARD = 5., 1., -5.
    j.COLLISION_REWARD = dict(zip(ARMS, (-5., -7.)))[arm]
    core.FactorEnv = Env


class Env(v1.BASE_ENV):
    progress_cap = 2.
    off_route_penalty = -5.
    idle_clear = pot = 0.

    def __init__(self, arm, folder, seed):
        setup(arm)
        super().__init__("F1", folder, seed)
        self.arm = arm
        self.sums = Counter()
        self.outcome_log = None
        net = folder / "high_real_ratio" / "network.net.xml"
        v2.configure_roads(self.sim, net)
        self.road_types = {e.get("id"): e.get("type","") for e in ET.parse(net).findall("edge")}
        assert self.progress_enabled and not self.nav_enabled and self.sim.max_steps == 1000
        assert self.visibility == 120 and self.obs_noise and self.occlusion and not self.lc_safety_enabled

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_off_route = None
        self.assigned_path = self.routes[self.episode_route]
        self.current_seed = kwargs.get("seed")
        assert len(obs) == 52 and tuple(obs[-2:]) == (0., 0.)
        return obs, info

    def departure(self, previous, previously_intact, info):
        if self.snapshot is None or info["collided"] or not previously_intact:
            return None
        actual = self.snapshot[0]
        index, src = previous
        if actual == src or actual.startswith(":"):
            return None
        path = self.assigned_path
        expected = path[index+1] if index+1 < len(path) else None
        if actual == expected:
            return None
        if (self.ego_id in j.traci.simulation.getStartingTeleportIDList()
                or self.ego_id in j.traci.simulation.getEndingTeleportIDList()):
            return None
        expected_kind = self.road_types.get(expected)
        actual_kind = self.road_types.get(actual)
        kind = "other_external"
        if expected_kind == "highway.motorway_link" and actual_kind == "highway.motorway":
            kind = "missed_exit"
        if expected_kind == "highway.motorway" and actual_kind == "highway.motorway_link":
            kind = "unwanted_exit"
        return dict(src=src, expected=expected, actual=actual, kind=kind)

    def step(self, action):
        # The inner simulator updates last_external on this very step.
        previous, intact = self.sim.last_external, self.sim.route_intact
        obs, reward, done, cut, info = super().step(action)
        self.last_off_route = self.departure(previous, intact, info)
        off_route = self.last_off_route is not None
        if off_route:
            reward, done, cut = self.off_route_penalty, True, False
            info.update(arrived=False, fork_passed=False, stalled=False, timeout_reward=0.,
                        progress_bonus=0., route_intact=False)
        outcome = None
        if done or cut:
            outcome = ("collision" if info["collided"] else "off_route" if off_route
                       else "arrival" if info["arrived"] else "stall" if info["stalled"]
                       else "timeout" if cut else "other")
        info.update(off_route=off_route, off_route_detail=self.last_off_route, outcome=outcome)
        parts = dict(progress=info["progress_bonus"], arrival=j.ARRIVAL_REWARD*info["arrived"],
                     fork=j.PASS_REWARD*info["fork_passed"], collision=j.COLLISION_REWARD*info["collided"],
                     timeout=info["timeout_reward"], off_route=self.off_route_penalty*off_route,
                     idle_clear=0., pot=0.)
        assert math.isclose(reward, sum(parts.values()), abs_tol=1e-7)
        self.sums.update(parts)
        if outcome and self.outcome_log is not None:
            record = dict(route=self.episode_route, traffic_seed=self.current_seed, outcome=outcome,
                          steps=self.sim.step_count, off_route=self.last_off_route)
            with self.outcome_log.open("a", encoding="utf8") as f:
                print(json.dumps(record), file=f)
        return obs, reward, done, cut, info


def historical_audit(routes):
    roots = dict(reward_mix_v2=v2.ROOT, long_600k_4=v1.ROOT.parent/"terminal_ratio"/"long_600k_4")
    audit = {}
    for name, root in roots.items():
        files = sorted(root.rglob("eval_*.json"))
        counts = Counter()
        outcomes = Counter()
        examples = []
        for file in files:
            rows = json.loads(file.read_bytes())
            for row in rows:
                counts["rows"] += 1
                outcomes[row["outcome"]] += 1
                path = routes[f"route_{row['route']:02d}"]
                terminal = row.get("timeout_distances") or {}
                trace = row.get("track") or []
                counts["tracked_rows"] += bool(trace)
                counts["timeout_false_intact"] += terminal.get("intact") is False
                observed = [sample[1] for sample in trace] + [terminal.get("end_edge")]
                wrong = next((e for e in observed if e and not e.startswith(":") and e not in path), None)
                if wrong is not None:
                    counts["confirmed_wrong_external"] += 1
                    if len(examples) < 8:
                        examples.append(dict(file=file.relative_to(root).as_posix(), route=row["route"], edge=wrong))
                elif row["outcome"] == "arrival" or terminal.get("intact") is True:
                    counts["certified_intact"] += 1
                else:
                    counts["indeterminate"] += 1
        assert files and counts["rows"] == 16 * len(files)
        assert counts["rows"] == sum(counts[k] for k in ("confirmed_wrong_external","certified_intact","indeterminate"))
        keys = ("rows","confirmed_wrong_external","certified_intact","indeterminate","timeout_false_intact","tracked_rows")
        audit[name] = dict(files=len(files), outside_route_examples=examples, outcomes=dict(outcomes),
                           **{k:counts[k] for k in keys})
    return audit


def check():
    for arm, penalty in zip(ARMS, (-5.,-7.)):
        setup(arm)
        assert (j.ARRIVAL_REWARD, j.PASS_REWARD, j.TIMEOUT_REWARD, j.COLLISION_REWARD,
                Env.progress_cap, Env.idle_clear, Env.pot, j.HP["lr"]) == (5.,1.,-5.,penalty,2.,0.,0.,1e-4)
    assert (j.STALL_SECONDS, j.HP["lane_entropy_coef"], j.HP["lane_keep_bias"]) == (30.,.002,1.)
    with TemporaryDirectory() as temp:
        folder = Path(temp) / "geometry"
        env = Env(ARMS[0], folder, 11)
        try:
            forks, counts = v2.audit_map(env, folder)
            cases = []
            for route, sign in ((13,-1.),(16,1.)):
                for lane in (0,1):
                    env.ego_depart_lane = str(lane)
                    obs, _ = env.reset(seed=111003, options={"route":f"route_{route:02d}"})
                    for _ in range(100):
                        assert env.snapshot[:2] == ("86198728",lane)
                        if env.snapshot[4] is not None:
                            break
                        obs, _, done, cut, info = env.step([0.,0.])
                        assert not (done or cut or info["off_route"])
                    assert env.snapshot[4] == 0 and obs[34] == sign
                    assert (obs[40:50] == env.road_flags[env.snapshot[0]]).all()
                    cases.append(dict(route=route, edge=env.snapshot[0], lane=lane, sign=float(obs[34])))
            departures = []
            # Deliberately change the simulator route in the test fixture only. Keep the
            # mission unchanged, place ego on the original approach and drive across the split.
            for route, alternate, lane, kind in ((13,16,1,"missed_exit"),(16,13,0,"unwanted_exit")):
                env.ego_depart_lane = str(lane)
                env.reset(seed=111003, options={"route":f"route_{route:02d}"})
                veh = j.traci.vehicle
                # Isolate collision-free geometry in this explicitly induced fault probe.
                for vehicle_id in veh.getIDList():
                    if vehicle_id != env.ego_id:
                        veh.remove(vehicle_id)
                alternate_path = env.routes[f"route_{alternate:02d}"]
                veh.setRoute(env.ego_id, alternate_path)
                lid = f"86198728_{lane}"
                # Position close enough to cross the fork without several hundred metres of warm-up.
                veh.moveTo(env.ego_id, lid, j.traci.lane.getLength(lid)-5.)
                veh.setSpeed(env.ego_id, 8.)
                for _ in range(80):
                    _, reward, done, cut, info = env.step([0.,0.])
                    if done or cut:
                        break
                assert done and not cut and reward == -5. and info["outcome"] == "off_route", (route,info)
                assert env.last_off_route["kind"] == kind and info["progress_bonus"] == 0.
                departures.append(dict(route=route, **env.last_off_route, reward=reward))
            audit = historical_audit(env.routes)
        finally:
            env.close()
        resume_check = check_resume_equivalence(Path(temp)/"resume_validation")
        smoke = {}
        for arm in ARMS:
            initial, update = train_job(arm, 11, 2048, Path(temp))
            assert abs(update["approx_kl"]) < .1 and update["rewards"]["idle_clear"] == update["rewards"]["pot"] == 0.
            smoke[arm] = dict(initial_sha256=initial, **update)
        assert len({s["initial_sha256"] for s in smoke.values()}) == 1
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT/"preflight_checks.json",dict(forks=forks, route_sign_counts=counts, ego_cases=cases,
         induced_off_route=departures, historical_audit=audit, smoke=smoke, resume_equivalence=resume_check, reward_tests=True))

import os
import random
import numpy as np


def train_job(arm, seed, end, stage, *, checkpoint_steps=None, env_type=Env):
    setup(arm)
    folder = stage / f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    resume_path = folder / "resume.pt"
    state = j.torch.load(resume_path, weights_only=False) if resume_path.is_file() else None
    model = core.agent(seed)
    initial = v1.digest(model)
    assert model.gamma == .99 and model.gae_lambda == .95
    logs = ("episodes.jsonl", "train_updates.jsonl", "train_outcomes.jsonl")
    if state:
        assert (arm, seed, initial) == (state["arm"], state["seed"], state["initial"])
        assert state["step"] <= end
        # Discard only records written since the last successful atomic checkpoint.
        for name in logs:
            path = folder/name
            offset = state["log_bytes"][name]
            assert path.stat().st_size >= offset
            if path.stat().st_size != offset:
                with path.open("r+b") as fp:
                    fp.truncate(offset)
        model.policy.load_state_dict(state["policy"])
        model.optimizer.load_state_dict(state["optimizer"])
        model.num_timesteps = state["step"]
        if not (folder/f"policy_{state['step']}.pt").exists():
            j.atomic_torch({"policy": state["policy"]},folder/f"policy_{state['step']}.pt")
    else:
        assert not any((folder/name).exists() for name in logs), "Partial run without resumable state"
        save(folder/"initial.json", dict(sha256=initial))
        for name in logs:
            (folder/name).touch()
    env = env_type(arm, folder/"map", seed)
    try:
        episodes = j.ScheduledEpisodes(env, seed, Path(os.devnull) if state else folder/"episodes.jsonl",
                                    state["next_episode"]-1 if state else 0)
        obs, _ = episodes.reset()
        stats = [0., 0]
        record = None
        if state:
            # Replay only the active drive, with the episode-specific SUMO and sensor seeds.
            for action in state["actions"]:
                obs, _, done, cut, _ = episodes.step(action)
                assert not (done or cut), "Restored episode terminated early"
            assert np.array_equal(obs, state["obs"])
            assert env.sim._noise_rng.bit_generator.state == state["noise_rng"]
            assert env.high == state["progress"] and env.sim.stopped_steps == state["stopped_steps"]
            assert (env.sim.route_intact, sorted(env.sim.paid_forks)) == state["route_state"]
            episodes.folder = folder/"episodes.jsonl"
            stats = state["stats"]
            record = state["last_update"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["np_rng"])
            j.torch.set_rng_state(state["torch_rng"])
        env.outcome_log = folder/"train_outcomes.jsonl"
        if checkpoint_steps is None:
            checkpoint_steps = [n for n in MARKS if n <= end]
            if not checkpoint_steps or checkpoint_steps[-1] < end:
                checkpoint_steps.append(end)
        for n in checkpoint_steps:
            if n <= model.num_timesteps:
                continue
            while model.num_timesteps < n:
                env.sums.clear()
                batch = min(2048, n-model.num_timesteps)
                obs, last, _ = model.collect_rollout(episodes, obs, ep_stats=stats, steps=batch,
                                                     timeout_is_failure=True)
                adv, ret = model.buffer.compute_gae(last, model.gamma, model.gae_lambda)
                metrics = model.update(adv, ret)
                assert all(math.isfinite(x) for x in metrics.values())
                lr = model.optimizer.param_groups[0]["lr"]
                assert lr == .0001
                record = dict(step=model.num_timesteps, batch=batch, lr=lr, rewards=dict(env.sums), **metrics)
                with (folder/"train_updates.jsonl").open("a", encoding="utf8") as fp:
                    print(json.dumps(record), file=fp)
            j.atomic_torch(dict(arm=arm, seed=seed, initial=initial, step=n,
                                policy=model.policy.state_dict(), optimizer=model.optimizer.state_dict(),
                                obs=obs, actions=episodes.actions, next_episode=episodes.n, stats=stats,
                                progress=env.high, stopped_steps=env.sim.stopped_steps,
                                route_state=(env.sim.route_intact, sorted(env.sim.paid_forks)),
                                torch_rng=j.torch.get_rng_state(), np_rng=np.random.get_state(),
                                python_rng=random.getstate(), noise_rng=env.sim._noise_rng.bit_generator.state,
                                last_update=record, log_bytes={name:(folder/name).stat().st_size for name in logs}),
                           resume_path)
            j.atomic_torch({"policy": model.policy.state_dict()}, folder/f"policy_{n}.pt")
        return initial, record
    finally:
        env.close()


def check_resume_equivalence(root):
    """Two actual PPO updates, compared bit for bit with a process-style restart between them."""
    continuous = root/"continuous"
    resumed = root/"resumed"
    arm, seed = ARMS[0], 11
    train_job(arm,seed,192,continuous,checkpoint_steps=(96,192))
    train_job(arm,seed,96,resumed,checkpoint_steps=(96,))
    train_job(arm,seed,192,resumed,checkpoint_steps=(96,192))
    name = f"{arm}_{seed}"
    a, b = (j.torch.load(base/name/"resume.pt",weights_only=False) for base in (continuous,resumed))
    assert all(j.torch.equal(t,b["policy"][k]) for k,t in a["policy"].items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for k, tensors in a["optimizer"]["state"].items():
        assert all(j.torch.equal(t,b["optimizer"]["state"][k][tag]) for tag,t in tensors.items())
    assert a["noise_rng"] == b["noise_rng"] and np.array_equal(a["obs"],b["obs"])
    assert j.torch.equal(a["torch_rng"],b["torch_rng"]) and a["stats"] == b["stats"]
    for fname in ("episodes.jsonl","train_updates.jsonl","train_outcomes.jsonl"):
        assert (continuous/name/fname).read_bytes() == (resumed/name/fname).read_bytes()
    return dict(step=192, resumed_at=96, equal_weights=True, equal_optimizer=True,
                equal_observations=True, equal_logs=True)


def eval_job(stage, arm, seed, step, rep, panel, *, env_type=Env, checkpoint=None):
    """Run the existing deterministic 16-route driver; publish enriched rows atomically."""
    setup(arm)
    terminal = {}
    departures = {}
    class EvalEnv(env_type):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self.route_key = self.episode_route
            departures[self.route_key] = False
            return result

        def step(self, action):
            before = self.snapshot
            result = super().step(action)
            _, _, done, cut, info = result
            after = self.snapshot
            if (self.route_key == "route_16" and before and after and before[3] == 0
                    and before[0] == after[0] == "86198728" and before[1] == 1 and after[1] == 0):
                departures[self.route_key] = True
            if done or cut:
                terminal[self.route_key] = (info["outcome"], info.get("off_route_detail"))
            return result

    def save_episodes(path, rows):
        assert len(terminal) == len(rows) == 16
        for row in rows:
            name = f"route_{row['route']:02d}"
            outcome, detail = terminal[name]
            assert outcome == "off_route" or outcome == row["outcome"], (outcome, row)
            row.update(outcome=outcome, off_route_detail=detail)
            if row["route"] == 16:
                row["r16_lane1_to0"] = departures[name]
        save(path, rows)

    old_env, old_root, old_save = core.FactorEnv, core.ROOT, core.save
    try:
        core.FactorEnv, core.ROOT, core.save = EvalEnv, stage, save_episodes
        core.evaluate((arm,seed,step,rep,panel), checkpoint=checkpoint)
    finally:
        core.FactorEnv, core.ROOT, core.save = old_env, old_root, old_save


def import_stage1(arm, seed, stage):
    """Restore 300k once. A marker distinguishes partial copies from training resumption."""
    src = ROOT / "stage1" / f"{arm}_{seed}"
    dest = stage / src.name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "promotion.json").exists():
        return
    required = ["initial.json","episodes.jsonl","train_updates.jsonl","train_outcomes.jsonl","resume.pt"]
    for step in MARKS[:6]:
        required.append(f"policy_{step}.pt")
        required.extend(f"eval_{SCREEN}_{step}_{rep}.json" for rep in range(4))
    for name in required:
        tmp = dest / (name + ".copying")
        shutil.copyfile(src/name, tmp)
        tmp.replace(dest/name)
    save(dest / "promotion.json", dict(arm=arm,seed=seed,step=300000,
                                       method="restore PPO optimizer, RNG and replay unfinished SUMO episode"))


def screen_stage(arms, seeds, end, stage, pool, *, from_stage1=True, env_type=Env):
    futures = set()
    scheduled = set()
    pairs = [(a,s) for a in arms for s in seeds]
    marks = [n for n in MARKS if n <= end]
    if end == 600000 and from_stage1:
        for arm in arms:
            for seed in (11,22):
                import_stage1(arm, seed, stage)
    for arm,seed in pairs:
        folder = stage / f"{arm}_{seed}"
        if not (folder / f"policy_{end}.pt").exists() or not (folder / "resume.pt").exists():
            futures.add(pool.submit(train_job,arm,seed,end,stage,env_type=env_type))
    while True:
        for n in marks:
            for a,s in pairs:
                folder = stage / f"{a}_{s}"
                if (a,s,n) in scheduled or not (folder / f"policy_{n}.pt").exists():
                    continue
                for r in range(4):
                    if not (folder / f"eval_{SCREEN}_{n}_{r}.json").exists():
                        futures.add(pool.submit(eval_job,stage,a,s,n,r,SCREEN,env_type=env_type))
                scheduled.add((a,s,n))
        if not futures:
            assert len(scheduled) == len(pairs)*len(marks)
            break
        done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
        for f in done:
            f.result()
    screen = {f"{a}_{s}_{n}": v1.aggregate(v2.snapshot_rows(stage,a,s,n,SCREEN))
              for a,s in pairs for n in marks}
    save(stage / "screen.json", screen)
    return screen


def promote(stage):
    ranking = {}
    for s in (11,22):
        assert len({read(stage/f"{a}_{s}"/"initial.json")["sha256"] for a in ARMS}) == 1
    for a in ARMS:
        by_seed, counts, forks = [], Counter(), 0
        for s in (11,22):
            drives = [r for n in (200000,250000,300000) for r in v2.snapshot_rows(stage,a,s,n,SCREEN)]
            c = Counter(r["outcome"] for r in drives)
            by_seed.append(c["arrival"])
            counts.update(c)
            forks += sum((r["route"] == 13 and "169728835" in r["forks"]) or
                         (r["route"] == 16 and "86198725" in r["forks"]) for r in drives)
        ranking[a] = [sum(by_seed),min(by_seed),-counts["collision"],
                      -sum(counts[k] for k in ("stall","timeout","off_route")),forks]
    winner = max(ARMS, key=ranking.get)
    save(ROOT/"stage1_summary.json",dict(promoted=[winner],ranking=ranking))
    return winner


def run_stage(arms, seeds, end, output_root=ROOT):
    stage = output_root / ("stage1" if end == 300000 else "stage2")
    stage.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        screen = screen_stage(arms,seeds,end,stage,pool)
        if end == 300000:
            return promote(stage)
        arm, = arms
        def score(n):
            c = Counter()
            for seed in seeds:
                c.update(screen[f"{arm}_{seed}_{n}"]["outcomes"])
            return c["arrival"],-c["collision"],-n
        best = max(MARKS,key=score)
        save(output_root/"selection.json",{arm:best})
        pending = []
        for panel in (PAIRED,FRESH,NOVEL):
            for n in sorted({best,end}):
                for s in seeds:
                    for r in range(4):
                        if not (stage/f"{arm}_{s}"/f"eval_{panel}_{n}_{r}.json").exists():
                            pending.append(pool.submit(eval_job,stage,arm,s,n,r,panel))
        for f in pending:
            f.result()


def protocol():
    p = v2.protocol()
    return dict(base_commit="42968ce273ed40bbc6d9361bf565be2cc8f129cb", arms=ARMS,
                training_backend="libsumo",environment=p["environment"],ppo=p["ppo"],
                shared_rewards={"arrival":5,"route_progress_cap":2,"first_valid_fork":1,
                                "stall":-5,"timeout":-5,"off_route":-5,"idle":0,"potential":0},
                collision={ARMS[0]:-5,ARMS[1]:-7},
                panels={"screen":SCREEN,"hold_paired":PAIRED,"hold_fresh":FRESH,"previously_unused":NOVEL,
                        "cases_per_seed":64,"note":"45110000 held out for v3 and reused from v1 evaluation; 77110000 additionally locked before training"},
                stage1=dict(seeds=[11,22], steps=300000,interval=50000,
                            ranking="20/25/30만 합산 완주 최다, 최소 시드 완주 최다, 충돌 최소, 끝 정지+일반 시간초과+경로 이탈 최소, 13·16번 통과 최다 순. 모두 같으면 S1"),
                stage2=dict(seeds=[11,22,33],steps=600000,interval=50000,
                            continuation="11/22 resume from stage1 with optimizer and RNG; 33 random initialization",
                            selection="highest total arrivals, fewest collisions, earliest checkpoint"),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())



# All counts and transitions below come from completed 64-drive trajectory panels.
SIMPLE_OUTCOMES = {"arrival":"완주", "collision":"충돌", "stall":"끝 정지",
                   "timeout":"일반 시간초과", "off_route":"실제 경로 이탈", "other":"기타"}


def simple_tables(folder, arms, seeds, steps, panel):
    lines = ["|조건|스텝|학습 시드|주행 수|완주|충돌|끝 정지|일반 시간초과|실제 경로 이탈|기타|차로변경/주행|13번 진출|16번 0차로 본선|16번 1차로 본선|",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    series = {}
    for arm in arms:
        for n in steps:
            groups = {}
            for seed in seeds:
                rows = v2.report_drives(folder, arm, seed, n, panel)
                if rows is not None:
                    groups[seed] = rows
            if len(groups) == len(seeds):
                groups["합산"] = [r for group in groups.values() for r in group]
            for seed, rows in groups.items():
                c = Counter(r["outcome"] for r in rows)
                assert set(c) <= SIMPLE_OUTCOMES.keys()
                rr = [r for r in rows if r["route"]==16]
                main = [sum("86198725" in r["forks"] and r["start"][1]==lane for r in rr) for lane in (0,1)]
                counts = [f"{c[k]} ({c[k]/len(rows):.1%})" for k in SIMPLE_OUTCOMES]
                moves = sum(r["moves"] for r in rows)/len(rows)
                ex = sum(r["route"]==13 and "169728835" in r["forks"] for r in rows)
                lines.append(f"|{arm}|{n:,}|{seed}|{len(rows)}|"+"|".join(counts)
                             +f"|{moves:.2f}|{ex}/{len(rr)}|{main[0]}/{len(rr)//2}|{main[1]}/{len(rr)//2}|")
                series[arm,n,seed] = dict(outcomes=c, n=len(rows), moves=moves, junctions=ex+sum(main))
    return (lines if series else ["완료된 64회 평가 묶음을 기다리고 있다."]), series


def simple_transition(lines, old, new):
    """Pair by measurement identity, additionally verify exact start state."""
    old_keys = {(r["route"], r["case"], r["traffic_seed"]): r for r in old}
    assert len(old_keys) == len(old) == len(new)
    keys = {(r["route"], r["case"], r["traffic_seed"]) for r in new}
    assert keys == set(old_keys)
    for r in new:
        before = old_keys[r["route"],r["case"],r["traffic_seed"]]
        assert before["start"] == r["start"]
        lines[before["outcome"],r["outcome"]] += 1


def pair_tables(old_root, old_arm, old_step, new_root, new_arm, new_step, panel, seeds):
    counts = Counter()
    for s in seeds:
        left = v2.report_drives(old_root,old_arm,s,old_step,panel)
        right = v2.report_drives(new_root,new_arm,s,new_step,panel)
        if left is None or right is None:
            return []
        simple_transition(counts,left,right)
    assert sum(counts.values()) == len(seeds)*64
    rows = []
    for source, label in SIMPLE_OUTCOMES.items():
        destinations = [counts[source,k] for k in SIMPLE_OUTCOMES]
        if sum(destinations):
            rows.append(f"|{old_arm}→{new_arm}|{old_step:,}→{new_step:,}|{label}|"
                        +"|".join(map(str,destinations))+f"|{sum(destinations)}|")
    return rows


def report():
    doc = Path(__file__).resolve().parents[1]/"docs/experiments/jys3-reward-simple-v3.md"
    heading = "## 측정 결과"
    lines = [doc.read_text(encoding="utf-8-sig").split(heading,1)[0].rstrip(), "", heading, ""]
    base = ROOT.parent/"terminal_ratio"/"long_600k_4"
    mix1, mix2 = [ROOT.parent/f"reward_mix_v{i}" for i in (1,2)]
    checks = ROOT/"preflight_checks.json"
    if checks.exists():
        audit = json.loads(checks.read_bytes())["historical_audit"]
        lines += ["### 이전 평가의 실제 경로 이탈 감사", "",
                  "체크포인트별 주행 기록을 모두 센다. 같은 모델·조건의 재평가도 한 기록으로 집계한다. 과거 궤적은 일부 도로·시간 간격만 저장하므로 확인 범위를 함께 표시했다.", "",
                  "|원자료|평가 파일|주행 기록|다른 외부 도로 진입 확인|경로 유지 확인|기록만으로 판별 불가|",
                  "|---|---:|---:|---:|---:|---:|"]
        for key in ("reward_mix_v2","long_600k_4"):
            a = audit[key]
            lines.append(f"|{key}|{a['files']}|{a['rows']}|{a['confirmed_wrong_external']}|{a['certified_intact']}|{a['indeterminate']}|")
        lines += ["", "v3은 매 스텝 외부 도로와 지정 경로를 대조한다. 실제 경로 이탈은 이후 표에서 별도 종료 원인으로 표시한다. [분기·차로 독립성·경로 이탈·학습 사전검증](../../results/jys3_ppo_comparison/pure_f/reward_simple_v3/preflight_checks.json)", ""]
    lines += ["### 1단계 고교통 중간 평가", "",
              "주행당 차로변경은 실제로 옆 차로로 이동한 횟수의 평균이다. 13번은 두 번째 목표 분기 진출, 16번은 첫 본선 통과를 출발 차로별로 집계했다.", ""]
    table, first = simple_tables(ROOT/"stage1", ARMS, (11,22), range(50000,300001,50000),SCREEN)
    lines += table
    transition_header = ["|기존→후보|스텝|기존 종료|완주로|충돌로|끝 정지로|일반 시간초과로|경로 이탈로|기타로|합계|",
                         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    completed = [a for a in ARMS if (a,300000,"합산") in first]
    if completed:
        lines += ["","### 30만 스텝의 동일 주행 128회 전환", "",
                  "각 모델 쌍에서 경로·교통 시드·출발 도로·차로·속도·위치를 대조했다.", "", *transition_header]
        for a in completed:
            for arm, folder in (("collision5",base),("P1_dir_clean",mix2/"stage1")):
                matched = pair_tables(folder,arm,300000,ROOT/"stage1",a,300000,SCREEN,(11,22))
                assert matched
                lines += matched
    s1 = ROOT/"stage1_summary.json"
    if s1.exists():
        selected = json.loads(s1.read_bytes())["promoted"]
        lines += ["", "### 1단계 승격 근거", "",
                  "|조건|완주 / 384|낮은 시드 완주 / 192|충돌|끝 정지+시간초과+경로 이탈|13·16번 통과|시드 22 30만 완주 / 64|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            late = (200000,250000,300000)
            reports = [first[a,n,"합산"] for n in late]
            c = sum((r["outcomes"] for r in reports),Counter())
            weakest = min(sum(first[a,n,s]["outcomes"]["arrival"] for n in late) for s in (11,22))
            fails = c["stall"]+c["timeout"]+c["off_route"]
            lines.append(f"|{a}|{c['arrival']}|{weakest}|{c['collision']}|{fails}|{sum(r['junctions'] for r in reports)}|{first[a,300000,22]['outcomes']['arrival']}|")
        lines += ["", "20만·25만·30만 합계에 사전 순위를 적용한 승격 조건은 **"+", ".join(selected)+"**이다. [선별 기록](../../results/jys3_ppo_comparison/pure_f/reward_simple_v3/stage1_summary.json)"]
        for a in selected:
            stage = ROOT/"stage2"
            lines += ["", "### 2단계 고교통 중간 평가", ""]
            table, screen = simple_tables(stage,[a],(11,22,33),MARKS,SCREEN)
            lines += table
            if screen:
                graph = ROOT/f"{a}_screen.svg"
                v2.render_screen_svg(screen,a,graph)
                lines += ["", f"![학습 시드별 완주·충돌·끝 정지·일반 시간초과 추이](../../results/jys3_ppo_comparison/pure_f/reward_simple_v3/{graph.name})"]
            selection = ROOT/"selection.json"
            if not selection.exists():
                continue
            best = json.loads(selection.read_bytes())[a]
            lines += ["",f"중간 평가에서 선택한 대표는 **{best:,}스텝**이다."]
            holds = {}
            marks = sorted({best,600000})
            for panel in (PAIRED,FRESH,NOVEL):
                lines += ["",f"### 최종 검증 · 교통 시드군 {panel}", ""]
                table, stats = simple_tables(stage,[a],(11,22,33),marks,panel)
                lines += table
                holds[panel] = stats
            if not all((a,n,"합산") in holds[p] for n in marks for p in (PAIRED,FRESH,NOVEL)):
                continue
            refs = [("collision5",base,base),("F1",base,base),("R3_pot_safe",mix1/"stage2",mix1)]
            if (mix2/"selection.json").exists() and "P1_dir_clean" in json.loads((mix2/"selection.json").read_bytes()):
                refs.append(("P1_dir_clean",mix2/"stage2",mix2))
            for tag,new_step in (("대표",best),("60만",600000)):
                lines += ["",f"### 동일 주행 192회 · {tag} 모델끼리 비교", "", *transition_header]
                for ref, folder, parent in refs:
                    old_step = json.loads((parent/"selection.json").read_bytes())[ref] if tag=="대표" else 600000
                    rows = pair_tables(folder,ref,old_step,stage,a,new_step,PAIRED,(11,22,33))
                    if rows:
                        lines += rows
                    else:
                        assert ref=="P1_dir_clean", (ref, old_step)
                        lines.extend(["","P1의 동일 조건 최종 평가 파일이 일부 남아 있다.",""])
            late = [screen[a,n,"합산"]["outcomes"]["arrival"] for n in (500000,550000,600000)]
            lines += ["","### 60만 이후 판단", "",
                      "50만·55만·60만의 192회 중간 평가 완주는 각각 "+"·".join(map(str,late))+"건이었다."]
            if best == 600000:
                lines.append("대표와 최종이 같은 60만 스텝이므로 세 교통 시드군에서 각각 한 번씩 검증했다.")
            else:
                for p, rr in holds.items():
                    c = rr[a,600000,"합산"]["outcomes"]
                    cb = rr[a,best,"합산"]["outcomes"]
                    lines.append(f"시드군 {p}에서 대표→60만의 완주는 {cb['arrival']}→{c['arrival']}, 충돌은 {cb['collision']}→{c['collision']}, 끝 정지는 {cb['stall']}→{c['stall']}건이었다.")
            goal = all(h[a,600000,"합산"]["outcomes"]["arrival"]/192>=.95 and
                       h[a,600000,"합산"]["outcomes"]["collision"]/192<.05 for h in holds.values())
            improved = late[0]<late[1]<late[2] and all(h[a,600000,"합산"]["outcomes"]["arrival"]>=h[a,best,"합산"]["outcomes"]["arrival"] for h in holds.values())
            if goal:
                reason = "세 시드군에서 60만 모델이 완주율과 충돌률 기준을 모두 충족해 여기서 동결한다."
            elif best == 600000:
                crashes = [screen[a,n,"합산"]["outcomes"]["collision"] for n in (550000,600000)]
                reason = (f"55만에서 60만으로 갈 때 완주는 {late[1]}→{late[2]}건, 충돌은 {crashes[0]}→{crashes[1]}건이었다. "
                          "세 교통 시드군을 대상으로 한 최종 검증에서 목표를 충족하지 못해 이번 실험은 60만에서 마쳤다.")
            elif improved:
                reason = "후반 세 시점에서 완주가 연속 증가했고 독립 평가에서도 60만 완주가 대표 수준을 유지했다. 연장 학습 후보로 유지한다."
            else:
                reason = "후반 중간 평가와 독립 검증을 함께 보면 완주의 지속 개선 근거가 부족해 연장 학습을 보류한다."
            lines.extend(["", reason])
    temp = doc.with_suffix(".md.partial")
    temp.write_text("\n".join(lines)+"\n", encoding="utf8")
    temp.replace(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("check","stage1","stage2","report","auto","s2-extension"):
        group.add_argument("--"+name,action="store_true")
    parser.add_argument("--arms",nargs="+",choices=ARMS)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    if args.report:
        report()
        return
    import msvcrt
    with (ROOT/"manager.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if args.s2_extension:
            assert read(ROOT/"preflight_checks.json")["resume_equivalence"]["equal_optimizer"]
            run_stage([ARMS[1]], (11,22,33), 600000, ROOT/"s2_extension")
            return
        if args.check or args.auto:
            save(ROOT/"protocol.json",protocol())
            check()
        if args.stage1 or args.stage2 or args.auto:
            assert (ROOT/"preflight_checks.json").is_file()
        if args.stage1 or args.auto:
            run_stage(ARMS,(11,22),300000)
            report()
        if args.stage2 or args.auto:
            promoted=read(ROOT/"stage1_summary.json")["promoted"]
            if args.stage2 and args.arms != promoted:
                parser.error("Use the promoted arm from stage1_summary.json")
            run_stage(promoted,(11,22,33),600000)
            report()


if __name__ == "__main__":
    main()
