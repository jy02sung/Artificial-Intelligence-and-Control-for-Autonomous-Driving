from pathlib import Path
from collections import Counter
from utils import ppo_initial as q
from utils.storage import save
import utils.imitation as final
import utils.ppo_rollout as p
import utils.kl_ppo as old
import torch
ROOT = Path(__file__).resolve().parents[1] / "results/jys3_pure_ppo_antispam"
PEAK = {22: 200000, 33: 250000}


def folder(track, seed):
    return ROOT / f"{track}_s{seed}"


def checkpoint(seed):
    return q.folder("Q1", seed) / f"policy_{PEAK[seed]}.pt"


def load_peak(seed):
    agent = p.agent(seed)
    state = torch.load(checkpoint(seed), map_location="cpu", weights_only=True)
    assert state["step"] == PEAK[seed]
    weights = dict(state["policy"])
    weights.update({"value_trunk." + key: val for key, val in state["value_trunk"].items()})
    agent.policy.load_state_dict(weights, strict=True)
    return agent


def warmup(seed, *, arm, dest, episode_type):
    torch.set_num_threads(1)
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
    with final.simulator() as env:
        episodes = episode_type(env, arm, warm_dir, training_seed=seed, warm=True)
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
