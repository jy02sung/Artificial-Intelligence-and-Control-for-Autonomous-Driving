import numpy as np
import torch


class RolloutBuffer:
    """한 번의 rollout(n_steps) 동안의 경험을 담아두는 버퍼.

    PPO는 on-policy 알고리즘이라, 업데이트 후에는 이 데이터를 버리고
    새 정책으로 다시 수집한다 (ReplayBuffer처럼 오래 쌓아두지 않는다).

    저장하는 것: (obs, action, log_prob, reward, value, done)
        log_prob : 그 행동을 뽑을 당시의 log π_old(a|s).
                   PPO의 확률비 r(θ) 계산에 필요해서 함께 저장한다.
        value    : 그 상태의 V(s) 추정치. GAE 계산에 필요.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(float(done))

    def __len__(self):
        return len(self.rewards)

    # ------------------------------------------------------------------
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE(λ)로 advantage와 return(critic 타깃)을 계산한다.

            δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
            A_t = δ_t + (γλ)·(1-done)·A_{t+1}     ← 뒤에서 앞으로 재귀
            R_t = A_t + V(s_t)

        Args:
            last_value: rollout 마지막 상태의 V(s). 에피소드가 중간에
                끊긴 경우 그 이후의 가치를 부트스트랩하는 데 쓴다.
        Returns:
            (advantages, returns) — 둘 다 np.ndarray, 길이 T
        """
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]     # 종료 상태면 미래가치 차단
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def to_tensors(self, device):
        """(obs, actions, old_log_probs)를 학습용 텐서로 변환."""
        return (
            torch.as_tensor(np.asarray(self.obs, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=device),
        )
