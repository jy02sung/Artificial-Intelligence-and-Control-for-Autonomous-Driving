"""Pure PPO S2 warm start and S1 matched from scratch. Run as a module to resume."""
import hashlib
import math
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from utils import jys3_pure_ppo_q_sweep as q

p, old, final, read, save = q.p, q.old, q.final, q.read, q.save
torch, np = p.torch, p.np
ROOT = q.ROOT.parent / "jys3_pure_ppo_antispam"
REPORT = q.REPORT.parent / "jys3-pure-ppo-antispam.md"
SEEDS = (22, 33)
PEAK = {22: 200000, 33: 250000}
MARKS = {"S2": tuple(range(25000, 200001, 25000)), "S1": tuple(range(50000, 350001, 50000))}
OUT = ("arrival", "collision", "off_route", "stall", "timeout")


def folder(track, seed):
    return ROOT / f"{track}_s{seed}"


def checkpoint(seed):
    return q.folder("Q1", seed) / f"policy_{PEAK[seed]}.pt"


def protected():
    files = sorted(f for f in q.ROOT.rglob("*") if f.is_file()) + [old.ORIG["R5"]]
    return {f.relative_to(q.ROOT.parent).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def penalties(obs, cmd):
    assert cmd in (-1, 0, 1)
    overlap = (cmd == 1 and obs[37] > .5) or (cmd == -1 and obs[36] > .5)
    return -.02 * bool(cmd), -.30 * bool(overlap)


class Episodes(p.Episodes):
    def __init__(self, env, arm, directory, *, training_seed, warm=False):
        super().__init__(env, arm, directory, training_seed=training_seed)
        self.transform = p.lateral_transform(env)
        old.driving.COLLISION_REWARD = -16.
        self.count = (200000 if warm else 100000) if arm == "S2" else 0
        self.stats = Counter()

    def drive(self, action):
        cmd = self.env.sim._quantize_lane_change(action[1])
        command_cost, overlap_cost = penalties(self.obs, cmd)
        nxt, reward, done, cut, info = super().drive(action)
        return nxt, reward + command_cost + overlap_cost, done, cut, info

    def step(self, action):
        cmd = self.env.sim._quantize_lane_change(action[1])
        command_cost, overlap_cost = penalties(self.obs, cmd)
        self.stats["commands"] += int(cmd != 0)
        self.stats["overlap_commands"] += int(overlap_cost < 0)
        self.stats["command_cost"] += command_cost
        self.stats["overlap_cost"] += overlap_cost
        return super().step(action)


def load_peak(seed):
    agent = p.agent(seed)
    state = torch.load(checkpoint(seed), map_location="cpu", weights_only=True)
    assert state["step"] == PEAK[seed]
    weights = dict(state["policy"])
    weights.update({"value_trunk." + key: val for key, val in state["value_trunk"].items()})
    agent.policy.load_state_dict(weights, strict=True)
    return agent


def warmup(seed):
    torch.set_num_threads(1)
    dest = folder("S2", seed)
    dest.mkdir(parents=True, exist_ok=True)
    agent = load_peak(seed)
    policy = agent.policy
    is_value = p.prev.is_value
    original = {k: v.clone() for k, v in policy.state_dict().items()}
    for name, param in policy.named_parameters():
        param.requires_grad_(is_value(name))
    critic = [param for param in policy.parameters() if param.requires_grad]
    opt = torch.optim.Adam(critic, lr=2e-5)
    warm_dir = dest / "warm"
    warm_dir.mkdir(exist_ok=True)
    (warm_dir / "episodes.jsonl").write_text("")
    history = []
    with final.p.roll.baseline.simulator() as env:
        episodes = Episodes(env, "S2", warm_dir, training_seed=seed, warm=True)
        obs, _ = episodes.reset()
        episode_stats = [0., 0]
        while agent.num_timesteps < 10000:
            size = min(agent.n_steps, 10000-agent.num_timesteps)
            obs, last, _ = agent.collect_rollout(episodes, obs, ep_stats=episode_stats, steps=size, timeout_is_failure=True)
            _, returns = agent.buffer.compute_gae(last, agent.gamma, agent.gae_lambda)
            states = agent.buffer.to_tensors(agent.device)[0]
            targets = torch.as_tensor(returns)
            for _ in range(10):
                for ids in torch.randperm(size).split(64):
                    values = policy.v_head(policy.value_trunk(states[ids])).flatten()
                    loss = torch.nn.functional.mse_loss(values, targets[ids])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic, .5)
                    opt.step()
            history.append(dict(step=agent.num_timesteps, last_mse=loss.item(), **episodes.stats))
            episodes.stats.clear()
    for name, value in policy.state_dict().items():
        if not is_value(name):
            assert torch.equal(value.flatten().view(torch.uint8), original[name].flatten().view(torch.uint8)), name
    for prefix in ("value_trunk.", "v_head."):
        assert any(not torch.equal(v, original[k]) for k, v in policy.state_dict().items() if k.startswith(prefix))
    save(dest/"warm_check.json", dict(actor_bits_identical=True, critic_trunk_and_head_changed=True,
                                      source_step=PEAK[seed], updates=history, sha256=p.digest(policy)))
    final.atomic_torch(policy.state_dict(), dest/"warm.pt")


def start_agent(track, seed):
    agent = p.agent(seed)
    if track == "S2":
        agent.policy.load_state_dict(torch.load(folder(track, seed)/"warm.pt", weights_only=True))
    return agent


def update(agent, ep, advantages, returns):
    n = agent.num_timesteps
    if ep.arm == "S1":
        lr = .0001 - .00008 * max(0, n-200000) / 150000
        entropy = .002 if n <= 200000 else .0005
    else:
        progress = max(0, n-agent.n_steps)/(200000-agent.n_steps)
        lr = .00003 - .00002 * progress
        entropy = .0005
    for group in agent.optimizer.param_groups:
        group["lr"] = lr
    agent.lane_entropy_coef = entropy
    # Adam's counter on an actor weight counts real optimizer steps.
    param = next(agent.policy.trunk.parameters())
    def count():
        return int(agent.optimizer.state.get(param, {}).get("step", 0))
    before = count()
    metrics = agent.update(advantages, returns)
    applied = count()-before
    assert applied == agent.n_epochs * math.ceil(len(returns)/agent.minibatch_size)
    metrics.update(minibatches=applied, lr=lr, lane_entropy_coef=entropy, **ep.stats)
    ep.stats.clear()
    return metrics


def train(track, seed, n):
    path = folder(track, seed)
    if track == "S2" and not (path/"warm.pt").exists():
        warmup(seed)
    digest = (read(path/"warm_check.json")["sha256"] if track=="S2" else
              read(q.folder("Q1",seed)/"initial.json")["sha256"])
    p.train(track,n,directory=path,make_agent=partial(start_agent,track,seed),
            make_episodes=Episodes,expected_hash=digest,training_seed=seed,update=update)


def eval_shard(track, seed, n, panel, rep):
    """Deterministic policy only; directly observe actions and actual lane changes."""
    torch.set_num_threads(1)
    dest = folder(track, seed)
    part = dest/"parts"/f"{n}_{panel}_{rep}.json"
    if part.exists():
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    source = checkpoint(seed) if track == "Q1" else dest/f"policy_{n}.pt"
    model = final.network(source)
    base, _ = old.PANELS[panel]
    rows = []
    with final.p.roll.baseline.simulator() as env, torch.inference_mode():
        transform = p.lateral_transform(env)
        assert env.physical_forks and env.occlusion and not env.lc_safety_enabled
        for route in range(1, 17):
            path = f"route_{route:02d}"
            lane = min(rep%2, env.counts[env.routes[path][0]]-1)
            env.ego_depart_lane = str(lane)
            traffic_seed = base + rep*100 + route
            obs, _ = env.reset(seed=traffic_seed, options={"route":path})
            initial = hashlib.sha256(obs.tobytes()).hexdigest()
            left = right = overlap = moves = 0
            for step in range(1, 1001):
                assert obs.shape == (52,) and not obs[50:].any()
                state = transform(obs[:50])
                action = model.det_action(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                cmd = env.sim._quantize_lane_change(action[1])
                left += int(cmd == 1)
                right += int(cmd == -1)
                overlap += int(penalties(state, cmd)[1] < 0)
                prev_edge, prev_lane = env.snapshot[:2]
                obs, _, done, cut, info = env.step(action)
                if env.snapshot and prev_edge == env.snapshot[0] and prev_lane != env.snapshot[1]:
                    moves += 1
                if done or cut:
                    break
            assert (done or cut) and info["outcome"] in OUT
            assert info["outcome"] != "arrival" or info["route_intact"]
            rows.append(dict(seed=traffic_seed,rep=rep,route=route,lane=lane,steps=step,initial_sha256=initial,
                             outcome=info["outcome"],left_commands=left,right_commands=right,
                             overlap_commands=overlap,within_edge_moves=moves))
    save(part,rows)


def evaluate(pool, jobs):
    requests = {}
    for track, seed, n, panel in jobs:
        dest = folder(track, seed)
        if (dest/f"eval_{n}_{panel}.json").exists():
            continue
        reps = old.PANELS[panel][1]
        requests[track,seed,n,panel] = [pool.submit(eval_shard,track,seed,n,panel,i) for i in range(reps)]
    for (track,seed,n,panel), futures in requests.items():
        for f in as_completed(futures):
            f.result()
        dest = folder(track, seed)
        rows = [r for i in range(len(futures)) for r in read(dest/"parts"/f"{n}_{panel}_{i}.json")]
        reference = read(q.panel("Q1",seed,PEAK[seed],panel))["rows"]
        old.paired(reference,rows)
        if track == "Q1":
            for a,b in zip(reference,rows):
                assert all(a[k]==b[k] for k in ("outcome","steps","left_commands","right_commands"))
        save(dest/f"eval_{n}_{panel}.json", dict(rows=rows,metrics=measure(rows)))
        print(track,seed,n,panel,measure(rows),flush=True)


def measure(rows):
    counts = Counter(r["outcome"] for r in rows)
    needed = [r for r in rows if p.groups()[r["route"],r["lane"]]=="needs_change"]
    return dict(n=len(rows),**{k:counts[k] for k in OUT},
                left=sum(r["left_commands"] for r in rows),right=sum(r["right_commands"] for r in rows),
                required=len(needed),required_arrived=sum(r["outcome"]=="arrival" for r in needed))


def select(track,seed):
    def rank(n):
        m=read(folder(track,seed)/f"eval_{n}_screen.json")["metrics"]
        return m["arrival"],-m["collision"],-m["stall"]-m["timeout"],-n
    return max(MARKS[track],key=rank)


def preflight():
    original = protected()
    assert original[old.ORIG["R5"].relative_to(q.ROOT.parent).as_posix()]==read(q.ROOT/"protocol.json")["protected_sha256"]["R5"]
    from utils import jys3, jys3_collect_bc
    import inspect
    assert "obs[37 if direction==1 else 36]" in inspect.getsource(jys3_collect_bc.expert)
    assert "for direction in (-1, 1)" in inspect.getsource(jys3.SparseRouteEnv._get_obs)
    with final.p.roll.baseline.simulator() as env:
        assert not env.lc_safety_enabled and env.physical_forks and env.occlusion
        assert [env.sim._quantize_lane_change(x) for x in (-1.,0.,1.)]==[-1,0,1]
        for cmd in (-1,0,1):
            for side in (36,37):
                obs=np.zeros(50)
                obs[side]=1.
                assert penalties(obs,cmd)==(-.02*bool(cmd),-.30*((cmd,side) in ((1,37),(-1,36))))
    for seed in SEEDS:
        assert p.digest(p.agent(seed).policy)==read(q.folder("Q1",seed)/"initial.json")["sha256"]
        assert read(q.panel("Q1",seed,PEAK[seed],"screen"))["metrics"]["arrival"]=={22:104,33:103}[seed]
    save(ROOT/"protected_start.json", original)
    save(ROOT/"protocol.json",dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         source_step=PEAK,workers=4,marks=MARKS,command_cost=-.02,overlap_cost=-.30,collision=-16.,
         warm_steps=10000,warm_lr=.00002,fresh_adam=True,ppo_epochs=10,source_folder=str(q.ROOT)))


def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    if not (ROOT/"protocol.json").exists():
        preflight()
    assert protected()==read(ROOT/"protected_start.json")
    with ProcessPoolExecutor(4) as pool:
        for track in ("S2","S1"):
            for n in MARKS[track]:
                jobs=[pool.submit(train,track,seed,n) for seed in SEEDS
                      if not (folder(track,seed)/f"policy_{n}.pt").exists()]
                for job in as_completed(jobs):
                    job.result()
                evaluate(pool,[(track,s,n,"screen") for s in SEEDS])
                report()
            holdouts = [(track,s,select(track,s),panel) for s in SEEDS for panel in q.HOLDS]
            if track == "S2":
                holdouts.extend(("Q1",s,PEAK[s],panel) for s in SEEDS for panel in q.HOLDS)
            evaluate(pool,holdouts)
            report()
    end=protected()
    assert end==read(ROOT/"protected_start.json")
    save(ROOT/"protected_end.json",end)
    save(ROOT/"finished.json",dict(protected_files=len(end),selection={f"{a}/{s}":select(a,s) for a in MARKS for s in SEEDS},
                                   screen_episodes=3840,holdout_episodes=2304))


def report():
    from utils.jys3_pure_ppo_antispam_report import generate
    generate()


if __name__=="__main__":
    main()
