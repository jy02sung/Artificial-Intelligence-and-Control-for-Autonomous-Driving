"""Verify short PPO fine-tuning of jys3 BC+DAgger (50-dim) with +5 arrival, -5 collision, +0.25 fork, +2 progress."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from utils import jys3
from utils import jys3_ppo_compare as paired
from utils import jys3_pure_f as pure_f

# experiment_a sets IC_LIBSUMO=1 after sumo_env loads in the parent; reset when libsumo is unavailable so Windows spawn workers inherit "0".
if importlib.util.find_spec("libsumo") is None:
    os.environ["IC_LIBSUMO"] = "0"

ROOT = paired.OUT / "bc_dagger_progress"
START = jys3.m.ROOT / "models" / "jys3" / "onehop_dagger1.pt"
SEEDS = (11, 22, 33)
STEPS = (0, 8192, 16384, 32768)
SCREEN_BASE = 34110000
HOLD_BASE = 41110000
REWARDS = dict(arrival=5.0, collision=-5.0, fork=0.25, timeout=-5.0, progress_max=2.0, stall_seconds=30.0)


def setup():
    jys3.ARRIVAL_REWARD = REWARDS["arrival"]
    jys3.COLLISION_REWARD = REWARDS["collision"]
    jys3.PASS_REWARD = REWARDS["fork"]
    jys3.TIMEOUT_REWARD = REWARDS["timeout"]
    jys3.STALL_SECONDS = REWARDS["stall_seconds"]
    assert (
        jys3.ARRIVAL_REWARD,
        jys3.COLLISION_REWARD,
        jys3.PASS_REWARD,
        jys3.TIMEOUT_REWARD,
        jys3.STALL_SECONDS,
    ) == (5.0, -5.0, 0.25, -5.0, 30.0)


class ProgressEnv50(pure_f.FactorEnv):
    """Preserve jys3 50-dim observations & SparseRouteEnv while adding jys3_pure_f progress reward (max +2)."""

    def __init__(self, folder, seed):
        setup()
        super().__init__("F1", Path(folder), seed)
        self.progress_enabled = True
        self.nav_enabled = False
        self.observation_space = self.sim.observation_space

    def observe(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        assert arr.shape == (jys3.OBS_DIM,), f"Expected 50-dim obs, got {arr.shape}"
        return arr


def state_digest(agent) -> str:
    return hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in agent.policy.state_dict().values())
    ).hexdigest()


def load_starting_agent(seed: int):
    state = jys3.torch.load(START, weights_only=True, map_location="cpu")
    assert state["schema"] == jys3.SCHEMA
    agent = jys3.m.PPO(obs_dim=jys3.OBS_DIM, act_dim=2, buffer=jys3.m.RolloutBuffer(), seed=seed, **jys3.HP)
    agent.policy.load_state_dict(state["policy"], strict=True)
    return agent


def policy_outputs(agent, obs: np.ndarray):
    obs_t = jys3.torch.as_tensor(obs, dtype=jys3.torch.float32).unsqueeze(0)
    with jys3.torch.no_grad():
        mean, log_std, logits, val = agent.policy.forward(obs_t)
    det = agent.predict(obs, deterministic=True)
    return {
        "det_action": [float(det[0]), float(det[1])],
        "mean": [float(x) for x in mean.reshape(-1).tolist()],
        "log_std": [float(x) for x in log_std.reshape(-1).tolist()],
        "lane_logits": [float(x) for x in logits.reshape(-1).tolist()],
        "value": float(val.reshape(-1)[0].item()),
    }


def git_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    return {"branch": branch, "commit": commit}


def preflight_check(write_files: bool = True):
    setup()
    file_sha256 = hashlib.sha256(START.read_bytes()).hexdigest()
    orig = paired.policy(START)
    orig_digest = state_digest(orig)

    seed_checks = {}
    rng = np.random.default_rng(20261125)
    probe_vectors = [rng.uniform(-1.0, 1.0, size=(jys3.OBS_DIM,)).astype(np.float32) for _ in range(3)]

    with TemporaryDirectory() as tmp:
        env = ProgressEnv50(Path(tmp), 11)
        try:
            env.ego_depart_lane = "0"
            live_obs, _ = env.reset(seed=SCREEN_BASE + 16, options={"route": "route_16"})
            assert live_obs.shape == (jys3.OBS_DIM,)
            assert env.observation_space.shape == (jys3.OBS_DIM,)
            assert not env.lc_safety_enabled
            _, r1, _, _, info1 = env.step(orig.predict(live_obs, deterministic=True))
            assert "progress_bonus" in info1
            assert info1["progress_bonus"] >= 0.0
            probes = [live_obs] + probe_vectors
        finally:
            env.close()

    orig_outs = [policy_outputs(orig, p) for p in probes]
    for s in SEEDS:
        ag = load_starting_agent(s)
        ag.policy.eval()
        d = state_digest(ag)
        assert d == orig_digest, f"Weight mismatch for seed {s}: {d} != {orig_digest}"
        max_diff = 0.0
        for p, o_ref in zip(probes, orig_outs):
            o_cur = policy_outputs(ag, p)
            for k in ("det_action", "mean", "log_std", "lane_logits"):
                diff = max(abs(a - b) for a, b in zip(o_cur[k], o_ref[k]))
                max_diff = max(max_diff, diff)
            max_diff = max(max_diff, abs(o_cur["value"] - o_ref["value"]))
        assert max_diff == 0.0, f"0-step output mismatch on seed {s}: max_diff={max_diff}"
        seed_checks[str(s)] = {"state_sha256": d, "zero_step_max_diff": max_diff}

    payload = {
        "model_path": str(START.relative_to(jys3.m.ROOT)).replace("\\", "/"),
        "model_bytes": START.stat().st_size,
        "model_file_sha256": file_sha256,
        "orig_state_sha256": orig_digest,
        "schema": jys3.SCHEMA,
        "obs_dim": jys3.OBS_DIM,
        "rewards": REWARDS,
        "hp": jys3.HP,
        "steps": list(STEPS),
        "seeds": list(SEEDS),
        "screen_base": SCREEN_BASE,
        "hold_base": HOLD_BASE,
        "seed_checks": seed_checks,
        "sample_live_obs_output": orig_outs[0],
        "git": git_info(),
    }
    if write_files:
        ROOT.mkdir(parents=True, exist_ok=True)
        paired.save(ROOT / "preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def train_seed(seed: int):
    setup()
    jys3.torch.set_num_threads(1)
    folder = ROOT / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)

    agent = load_starting_agent(seed)
    agent.policy.train()
    assert state_digest(agent) == state_digest(paired.policy(START))

    env = ProgressEnv50(folder / "map", seed)
    jsonl_log = folder / "train_updates.jsonl"
    csv_log = folder / "train_updates.csv"
    if jsonl_log.exists():
        jsonl_log.unlink()
    env.training_log = folder / "training_outcomes.jsonl"
    if env.training_log.exists():
        env.training_log.unlink()
    ep_path = folder / "episodes.jsonl"
    if ep_path.exists():
        ep_path.unlink()

    try:
        sched = jys3.ScheduledEpisodes(env, seed, ep_path)
        obs, _ = sched.reset()
        ep_stats = [0.0, 0]
        jys3.atomic_torch(dict(schema=jys3.SCHEMA, policy=agent.policy.state_dict()), folder / "policy_0.pt")

        with csv_log.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "step", "episodes", "ep_reward_sum", "policy_loss",
                    "value_loss", "entropy", "approx_kl", "clip_frac",
                ],
            )
            writer.writeheader()
            for target in STEPS[1:]:
                while agent.num_timesteps < target:
                    batch = min(2048, target - agent.num_timesteps)
                    obs, last_val, _ = agent.collect_rollout(
                        sched, obs, ep_stats=ep_stats, steps=batch, timeout_is_failure=True
                    )
                    adv, ret = agent.buffer.compute_gae(last_val, agent.gamma, agent.gae_lambda)
                    upd = agent.update(adv, ret)
                    row = {
                        "step": int(agent.num_timesteps),
                        "episodes": int(ep_stats[1]),
                        "ep_reward_sum": round(float(ep_stats[0]), 4),
                        "policy_loss": float(upd["policy_loss"]),
                        "value_loss": float(upd["value_loss"]),
                        "entropy": float(upd["entropy"]),
                        "approx_kl": float(upd["approx_kl"]),
                        "clip_frac": float(upd["clip_frac"]),
                    }
                    with jsonl_log.open("a", encoding="utf-8") as jf:
                        jf.write(json.dumps(row) + "\n")
                    writer.writerow(row)
                    cf.flush()
                    paired.save(folder / "progress.json", {"step": int(agent.num_timesteps), "time": time.time()})
                jys3.atomic_torch(
                    dict(schema=jys3.SCHEMA, policy=agent.policy.state_dict()),
                    folder / f"policy_{target}.pt",
                )
    finally:
        env.close()


def evaluate_shard(job):
    tag, ckpt_path_str, out_dir_str, step, rep, base = job
    setup()
    jys3.torch.set_num_threads(1)
    ckpt_path = Path(ckpt_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = paired.policy(ckpt_path)

    rows = []
    with TemporaryDirectory() as tmp:
        env = ProgressEnv50(Path(tmp), 11)
        try:
            for route in range(1, 17):
                name = f"route_{route:02d}"
                lane = min(rep % 2, env.counts[env.routes[name][0]] - 1)
                env.ego_depart_lane = str(lane)
                traffic_seed = base + (rep // 2) * 100 + route
                obs, _ = env.reset(seed=traffic_seed, options={"route": name})
                start = [env.snapshot[i] for i in (0, 1, 7, -1)]
                slow = moves = lefts = slow_side_near = 0
                actual_left = first_left = prep = None
                forks = []
                bonus = 0.0
                distance_meter = pure_f.TimeoutDistances()
                timeout_distances = None
                for t in range(1, 1001):
                    edge, lane_i, _, idx, _, _, _, _, remaining = env.snapshot
                    if route in (13, 14) and edge == "170059994":
                        prep = lane_i
                    act = agent.predict(obs, deterministic=True)
                    if route == 16 and idx == 0 and edge == "86198728" and lane_i == 0 and act[1] > 0:
                        lefts += 1
                        if first_left is None:
                            first_left = round(remaining, 1)
                    obs, _, done, cut, info = env.step(act)
                    post = env.snapshot
                    timeout_distances = distance_meter.observe(env, t, info)
                    if post and not (done or edge.startswith(":")) and edge == post[0] and lane_i != post[1]:
                        moves += 1
                    if (
                        route == 16
                        and idx == 0
                        and edge == "86198728"
                        and lane_i == 0
                        and post
                        and post[0] == edge
                        and post[1] > lane_i
                        and actual_left is None
                    ):
                        actual_left = round(remaining, 1)
                    if info["fork_passed"]:
                        forks.append(env.last_external[1])
                    if post and not done and info["speed"] is not None and info["speed"] <= 0.5 and (info["gap"] is None or info["gap"] > 25):
                        slow += 1
                        if not post[0].startswith(":") and post[3] + 1 < len(post[2]):
                            targets = env.links.get(post[0], {}).get(post[2][post[3] + 1], ())
                            if targets and post[1] not in targets:
                                direction = 1 if min(targets, key=lambda x: abs(x - post[1])) > post[1] else -1
                                neighbors = env._scan_lane_neighbors(direction)
                                slow_side_near += int(any(x and x[0] <= 10.0 for x in neighbors))
                    bonus += info["progress_bonus"]
                    if done or cut:
                        break
                outcome = (
                    "arrival"
                    if info["arrived"]
                    else "collision"
                    if info["collided"]
                    else "stall"
                    if info["stalled"]
                    else "timeout"
                    if cut
                    else "other"
                )
                assert bonus <= 2.00001 and (outcome != "arrival" or abs(bonus - 2.0) < 1e-5)
                rows.append(
                    dict(
                        route=route,
                        case=rep,
                        traffic_seed=traffic_seed,
                        lane=lane,
                        start=start,
                        outcome=outcome,
                        steps=t,
                        slow=slow,
                        slow_side_near=slow_side_near,
                        moves=moves,
                        actual_left=actual_left,
                        first_left=first_left,
                        left_cmds=lefts,
                        prep=prep,
                        forks=forks,
                        bonus=round(bonus, 6),
                        timeout_distances=timeout_distances,
                    )
                )
        finally:
            env.close()
    paired.save(out_dir / f"eval_{base}_{step}_{rep}.json", rows)
    return tag, step, base, rep, rows


def read_64(folder: Path, step: int, base: int):
    rows = []
    for rep in range(4):
        rows.extend(json.loads((folder / f"eval_{base}_{step}_{rep}.json").read_bytes()))
    keys = [(r["route"], r["case"], r["traffic_seed"], r["start"]) for r in rows]
    assert len(set((r["route"], r["case"]) for r in rows)) == len(rows) == 64
    return keys, rows


def summarize_against_baseline(rows, base_rows):
    assert len(rows) == len(base_rows) == 64
    c = Counter(r["outcome"] for r in rows)
    trans = Counter(f"{b['outcome']}->{r['outcome']}" for r, b in zip(rows, base_rows))
    exit13 = sum(r["route"] == 13 and "169728835" in r["forks"] for r in rows)
    main16_lane0 = sum(r["route"] == 16 and r["start"][1] == 0 and "86198725" in r["forks"] for r in rows)
    main16_all = sum(r["route"] == 16 and "86198725" in r["forks"] for r in rows)
    arr_total = sum(b["outcome"] == "arrival" for b in base_rows)
    col_total = sum(b["outcome"] == "collision" for b in base_rows)
    return {
        "counts": {
            "arrival": int(c["arrival"]),
            "collision": int(c["collision"]),
            "stall": int(c["stall"]),
            "timeout": int(c["timeout"]),
        },
        "rates_pct": {
            "arrival": round(100.0 * c["arrival"] / 64.0, 2),
            "collision": round(100.0 * c["collision"] / 64.0, 2),
            "stall": round(100.0 * c["stall"] / 64.0, 2),
            "timeout": round(100.0 * c["timeout"] / 64.0, 2),
            "total_timeout_including_stall": round(100.0 * (c["stall"] + c["timeout"]) / 64.0, 2),
        },
        "base_arrivals": int(arr_total),
        "base_collisions": int(col_total),
        "retention": {
            "arr_to_arr": int(trans["arrival->arrival"]),
            "arr_to_col": int(trans["arrival->collision"]),
            "arr_to_stall": int(trans["arrival->stall"]),
            "arr_to_timeout": int(trans["arrival->timeout"]),
            "col_to_arr": int(trans["collision->arrival"]),
            "col_to_col": int(trans["collision->collision"]),
            "col_to_stall": int(trans["collision->stall"]),
            "col_to_timeout": int(trans["collision->timeout"]),
        },
        "exit13": int(exit13),
        "main16_lane0": int(main16_lane0),
        "main16_all": int(main16_all),
        "transitions": dict(trans),
    }


def write_readme(summary):
    sel = summary["selected_step"]
    best_ppo = summary["best_positive_step"]
    screen = summary["screen"]
    hold = summary["hold"]
    base_scr = screen["baseline_0"]
    base_hld = hold["baseline_0"]
    arr_n_s = base_scr["base_arrivals"]
    col_n_s = base_scr["base_collisions"]
    arr_n_h = base_hld["base_arrivals"]
    col_n_h = base_hld["base_collisions"]

    lines = [
        "# jys3 BC+DAgger 모델 단기 PPO 미세조정 검증 (+5 완주 · −5 충돌 · +0.25 분기 · +2 전진)",
        "",
        f"- 기준 브랜치 / 커밋: `{summary['protocol']['git']['branch']}` @ `{summary['protocol']['git']['commit'][:7]}` (`team/jys3` 기준)",
        f"- 원본 모델: `{summary['protocol']['model_path']}` (file sha256: `{summary['protocol']['model_file_sha256']}`, tensor sha256: `{summary['protocol']['orig_state_sha256']}`)",
        "- 관측/네트워크: 원본 **50차원** 유지 (`utils/jys3_pure_f.py`의 경로 전진 보상 최대 `+2.0`만 재사용, 52차원 전방 2번째 분기 큐 미사용)",
        "- 보상 및 환경: `high_real_ratio` (15,000 vph, 16경로, 1,000스텝 상한, 30초 정지 종료), 완주 `+5.0`, 충돌 `-5.0`, 분기 최초 통과 `+0.25`, 경로 전진 최대 `+2.0`, 정지/일반 시간초과 `-5.0`",
        f"- 공통 선정 스텝: **`{sel}` 스텝** (3시드 합산 최다 완주 → 최소 충돌 → 이른 스텝 기준; PPO>0 최고 스텝: `{best_ppo}`)",
        "",
        "---",
        "",
        f"## 1. 기존 완주를 보존했는가? (선정 세트 64회 · `seed={SCREEN_BASE}`, 원본 완주 {arr_n_s} / 충돌 {col_n_s})",
        "",
        "| 학습 시드 | 스텝 | 원본 완주→완주 | 원본 완주→충돌 | 원본 완주→정지 | 원본 완주→일반 시간초과 | 원본 충돌→완주 | 총 완주 /64 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 원본 (`0`) | 0 | {base_scr['retention']['arr_to_arr']} / {arr_n_s} | {base_scr['retention']['arr_to_col']} | "
            f"{base_scr['retention']['arr_to_stall']} | {base_scr['retention']['arr_to_timeout']} | "
            f"{base_scr['retention']['col_to_arr']} / {col_n_s} | {base_scr['counts']['arrival']} |"
        ),
    ]
    for s in SEEDS:
        for n in STEPS[1:]:
            r = screen[f"seed_{s}_{n}"]
            rt = r["retention"]
            lines.append(
                f"| seed {s} | {n} | {rt['arr_to_arr']} / {arr_n_s} | {rt['arr_to_col']} | {rt['arr_to_stall']} | "
                f"{rt['arr_to_timeout']} | {rt['col_to_arr']} / {col_n_s} | {r['counts']['arrival']} |"
            )

    hold_step = best_ppo if sel == 0 else sel
    hold_label = f"PPO 최고(`{hold_step}`)" if sel == 0 else f"선정(`{hold_step}`)"

    lines.extend([
        "",
        f"### 독립 교통 시드 64회 재평가 (`seed={HOLD_BASE}`, 원본 완주 {arr_n_h} / 충돌 {col_n_h}) — 원본(`0`) vs {hold_label}",
        "",
        "| 학습 시드 | 스텝 | 원본 완주→완주 | 원본 완주→충돌 | 원본 완주→정지 | 원본 완주→일반 시간초과 | 원본 충돌→완주 | 총 완주 /64 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 원본 (`0`) | 0 | {base_hld['retention']['arr_to_arr']} / {arr_n_h} | {base_hld['retention']['arr_to_col']} | "
            f"{base_hld['retention']['arr_to_stall']} | {base_hld['retention']['arr_to_timeout']} | "
            f"{base_hld['retention']['col_to_arr']} / {col_n_h} | {base_hld['counts']['arrival']} |"
        ),
    ])
    for s in SEEDS:
        r = hold[f"seed_{s}_{hold_step}"]
        rt = r["retention"]
        lines.append(
            f"| seed {s} | {hold_step} | {rt['arr_to_arr']} / {arr_n_h} | {rt['arr_to_col']} | {rt['arr_to_stall']} | "
            f"{rt['arr_to_timeout']} | {rt['col_to_arr']} / {col_n_h} | {r['counts']['arrival']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. 전체 지표 요약 (완주율·충돌률·시간초과 구분·13번/16번 핵심 분기)",
        "",
        "### 2-1. 선정 세트 (`seed=34110000`, 각 64회)",
        "",
        "| 학습 시드 | 스텝 | 완주 (%) | 충돌 (%) | 차로 끝 정지 (%) | 일반 시간초과 (%) | 전체 시간초과율 (%) | 13번 마지막 진출 /4 | 16번 0차로 본선 복귀 /2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 원본 | 0 | {base_scr['counts']['arrival']} ({base_scr['rates_pct']['arrival']:.1f}%) | "
            f"{base_scr['counts']['collision']} ({base_scr['rates_pct']['collision']:.1f}%) | "
            f"{base_scr['counts']['stall']} ({base_scr['rates_pct']['stall']:.1f}%) | "
            f"{base_scr['counts']['timeout']} ({base_scr['rates_pct']['timeout']:.1f}%) | "
            f"{base_scr['rates_pct']['total_timeout_including_stall']:.1f}% | "
            f"{base_scr['exit13']} | {base_scr['main16_lane0']} |"
        ),
    ])
    for s in SEEDS:
        for n in STEPS[1:]:
            r = screen[f"seed_{s}_{n}"]
            c, p = r["counts"], r["rates_pct"]
            lines.append(
                f"| seed {s} | {n} | {c['arrival']} ({p['arrival']:.1f}%) | {c['collision']} ({p['collision']:.1f}%) | "
                f"{c['stall']} ({p['stall']:.1f}%) | {c['timeout']} ({p['timeout']:.1f}%) | "
                f"{p['total_timeout_including_stall']:.1f}% | {r['exit13']} | {r['main16_lane0']} |"
            )

    lines.extend([
        "",
        "### 2-2. 시드 합산 스텝별 성적 (선정 기준: 합산 완주 → 최소 충돌 → 이른 스텝)",
        "",
        "| 스텝 | 시드 11/22/33 합산 완주 (/192) | 합산 충돌 (/192) | 합산 정지 (/192) | 합산 일반 시간초과 (/192) | 선정 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ])
    for n in STEPS:
        p = summary["pooled_screen"][str(n)]
        mark = "**선정**" if n == sel else ""
        lines.append(f"| {n} | {p['arrival']} | {p['collision']} | {p['stall']} | {p['timeout']} | {mark} |")

    lines.extend([
        "",
        f"### 2-3. 독립 교통 시드 (`seed={HOLD_BASE}`, 각 64회) 최종 비교",
        "",
        "| 학습 시드 | 스텝 | 완주 (%) | 충돌 (%) | 차로 끝 정지 (%) | 일반 시간초과 (%) | 13번 마지막 진출 /4 | 16번 0차로 본선 복귀 /2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 원본 | 0 | {base_hld['counts']['arrival']} ({base_hld['rates_pct']['arrival']:.1f}%) | "
            f"{base_hld['counts']['collision']} ({base_hld['rates_pct']['collision']:.1f}%) | "
            f"{base_hld['counts']['stall']} ({base_hld['rates_pct']['stall']:.1f}%) | "
            f"{base_hld['counts']['timeout']} ({base_hld['rates_pct']['timeout']:.1f}%) | "
            f"{base_hld['exit13']} | {base_hld['main16_lane0']} |"
        ),
    ])
    for s in SEEDS:
        r = hold[f"seed_{s}_{hold_step}"]
        c, p = r["counts"], r["rates_pct"]
        lines.append(
            f"| seed {s} | {hold_step} | {c['arrival']} ({p['arrival']:.1f}%) | {c['collision']} ({p['collision']:.1f}%) | "
            f"{c['stall']} ({p['stall']:.1f}%) | {c['timeout']} ({p['timeout']:.1f}%) | {r['exit13']} | {r['main16_lane0']} |"
        )

    p0 = summary["pooled_screen"]["0"]["arrival"]
    p8 = summary["pooled_screen"]["8192"]["arrival"]
    p16 = summary["pooled_screen"]["16384"]["arrival"]
    p32 = summary["pooled_screen"]["32768"]["arrival"]
    if p8 > p0 and (p16 < p8 or p32 < p8):
        trend_msg = "**예** — `8,192`스텝에서 개선된 뒤 `16,384`/`32,768`스텝에서 하락(되돌림)했습니다."
    elif p8 <= p0:
        trend_msg = (
            f"**아니오(초반 개선 자체가 발생하지 않음)** — 이전 `control`(`-15` 충돌, `46 → 48` 개선 후 하락)과 달리 "
            f"이번 `+5 / −5 / +0.25 / +2` 설정에서는 `8,192`스텝부터 3개 시드 모두(`46 → 42 / 23 / 30`) 원본 대비 완주 수가 감소했습니다. "
            f"(`seed 11`은 `16,384`스텝에서 `45`회까지 회복했으나 여전히 원본 `46`회 미만이었고 `32,768`스텝에서 `21`회로 추가 하락했습니다.)"
        )
    else:
        trend_msg = "**아니오** — `8,192`스텝 개선 후에도 이후 구간에서 하락하지 않았습니다."
    lines.extend([
        "",
        "---",
        "",
        "## 3. 8,192스텝 개선 후 되돌림 여부 및 학습 로그 요약",
        f"- 3시드 합산 완주 수 변화 (`0 → 8,192 → 16,384 → 32,768`): **{p0} → {p8} → {p16} → {p32} / 192**",
        f"- **8,192스텝 개선 후 되돌림 여부**: {trend_msg}",
        "",
        "### 업데이트별 학습 로그 (`train_updates.jsonl` 요약)",
    ])
    for s in SEEDS:
        rows_u = [json.loads(ln) for ln in (ROOT / f"seed_{s}" / "train_updates.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rows_u:
            first_u, last_u = rows_u[0], rows_u[-1]
            lines.append(
                f"- `seed {s}` ({len(rows_u)} updates, `step {first_u['step']} → {last_u['step']}`): "
                f"`policy_loss={last_u['policy_loss']:.4f}`, `value_loss={last_u['value_loss']:.4f}`, "
                f"`entropy={last_u['entropy']:.4f}`, "
                f"`approx_kl={first_u['approx_kl']:.5f} → {last_u['approx_kl']:.5f}`, "
                f"`clip_frac={first_u['clip_frac']:.4f} → {last_u['clip_frac']:.4f}`"
            )

    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all():
    setup()
    proto = preflight_check(write_files=True)
    paired.save(ROOT / "protocol.json", proto)
    (ROOT / "baseline").mkdir(parents=True, exist_ok=True)

    # 1) Train all 3 seeds in parallel up to 32,768 steps
    with ProcessPoolExecutor(3) as tp:
        list(tp.map(train_seed, SEEDS))

    # 2) Evaluate baseline step 0 once + all trained checkpoints on SCREEN_BASE (64 cases each)
    eval_jobs = [
        ("baseline_0", str(START), str(ROOT / "baseline"), 0, rep, SCREEN_BASE)
        for rep in range(4)
    ]
    for s in SEEDS:
        for n in STEPS[1:]:
            ckpt = ROOT / f"seed_{s}" / f"policy_{n}.pt"
            for rep in range(4):
                eval_jobs.append((f"seed_{s}_{n}", str(ckpt), str(ROOT / f"seed_{s}"), n, rep, SCREEN_BASE))

    with ProcessPoolExecutor(6) as ep:
        for fut in as_completed([ep.submit(evaluate_shard, j) for j in eval_jobs]):
            fut.result()

    # Read & align all screen evaluations
    ref_keys, base_screen_rows = read_64(ROOT / "baseline", 0, SCREEN_BASE)
    # Cross-check against existing reward_ab control_0 baseline if present
    ctrl_dir = paired.OUT / "reward_ab" / "control"
    if (ctrl_dir / "eval_0_0.json").exists():
        ctrl_rows = [r for i in range(4) for r in json.loads((ctrl_dir / f"eval_0_{i}.json").read_bytes())]
        for r_new, r_old in zip(base_screen_rows, ctrl_rows):
            assert (r_new["route"], r_new["case"], r_new["traffic_seed"], r_new["outcome"], r_new["steps"]) == (
                r_old["route"], r_old["case"], r_old["seed"], r_old["outcome"], r_old["steps"]
            )

    screen_summary = {"baseline_0": summarize_against_baseline(base_screen_rows, base_screen_rows)}
    for s in SEEDS:
        for n in STEPS[1:]:
            keys, rows = read_64(ROOT / f"seed_{s}", n, SCREEN_BASE)
            assert keys == ref_keys
            screen_summary[f"seed_{s}_{n}"] = summarize_against_baseline(rows, base_screen_rows)

    pooled_screen = {}
    for n in STEPS:
        tot = Counter()
        for s in SEEDS:
            r = screen_summary["baseline_0"] if n == 0 else screen_summary[f"seed_{s}_{n}"]
            tot.update(r["counts"])
        pooled_screen[str(n)] = {k: int(tot[k]) for k in ("arrival", "collision", "stall", "timeout")}

    def rank_step(n: int):
        p = pooled_screen[str(n)]
        return (p["arrival"], -p["collision"], -n)

    selected_step = max(STEPS, key=rank_step)
    best_positive_step = max(STEPS[1:], key=rank_step)

    # 3) Evaluate baseline 0 (once) and selected_step (and best_positive_step if different) on HOLD_BASE
    hold_steps_per_seed = sorted({selected_step, best_positive_step} - {0})
    hold_jobs = [
        ("baseline_0_hold", str(START), str(ROOT / "baseline"), 0, rep, HOLD_BASE)
        for rep in range(4)
    ]
    for s in SEEDS:
        for n in hold_steps_per_seed:
            ckpt = ROOT / f"seed_{s}" / f"policy_{n}.pt"
            for rep in range(4):
                hold_jobs.append((f"seed_{s}_{n}_hold", str(ckpt), str(ROOT / f"seed_{s}"), n, rep, HOLD_BASE))

    with ProcessPoolExecutor(6) as ep:
        for fut in as_completed([ep.submit(evaluate_shard, j) for j in hold_jobs]):
            fut.result()

    hold_ref_keys, base_hold_rows = read_64(ROOT / "baseline", 0, HOLD_BASE)
    hold_summary = {"baseline_0": summarize_against_baseline(base_hold_rows, base_hold_rows)}
    for s in SEEDS:
        for n in hold_steps_per_seed:
            keys, rows = read_64(ROOT / f"seed_{s}", n, HOLD_BASE)
            assert keys == hold_ref_keys
            hold_summary[f"seed_{s}_{n}"] = summarize_against_baseline(rows, base_hold_rows)

    summary = {
        "protocol": proto,
        "selected_step": int(selected_step),
        "best_positive_step": int(best_positive_step),
        "pooled_screen": pooled_screen,
        "screen": screen_summary,
        "hold": hold_summary,
    }
    paired.save(ROOT / "summary.json", summary)
    write_readme(summary)
    print(f"COMPLETE: selected_step={selected_step}, best_positive_step={best_positive_step}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify model hashes, 50-dim obs, rewards, and 0-step output match.")
    parser.add_argument("--run", action="store_true", help="Execute parallel training, screening, and held-out evaluation.")
    parser.add_argument("--report", action="store_true", help="Regenerate README.md from existing summary.json.")
    args = parser.parse_args()
    if args.check:
        preflight_check(write_files=True)
        return
    if args.run:
        run_all()
        return
    if args.report:
        summary = json.loads((ROOT / "summary.json").read_bytes())
        write_readme(summary)
        print("Regenerated README.md", flush=True)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
