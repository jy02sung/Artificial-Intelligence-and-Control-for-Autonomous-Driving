"""Isolated PPO rollout boundary fix; leaves currently running legacy trials intact."""
class TimeoutBoundaryMixin:
    def collect_rollout(self, env, state, logger=None, ep_stats=None, steps=None):
        obs = state
        finished_returns = []
        self.buffer.clear()
        for _ in range(self.n_steps if steps is None else steps):
            action, log_prob, value = self.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            self.num_timesteps += 1
            ep_stats[0] += reward
            ep_stats[1] += 1
            stored_reward = reward
            if truncated and not terminated:
                stored_reward += self.gamma * self.get_value(next_obs)
            # Bootstrap the timeout observation above, then cut both GAE and
            # next-value links before resetting into an unrelated episode.
            self.buffer.add(obs, action, log_prob, stored_reward, value,
                            terminated or truncated)
            if terminated or truncated:
                finished_returns.append(ep_stats[0])
                if logger is not None:
                    logger.log_episode(self.num_timesteps, ep_stats[0], int(ep_stats[1]))
                ep_stats[0], ep_stats[1] = 0.0, 0
                obs, _ = env.reset()
            else:
                obs = next_obs
        return obs, self.get_value(obs), finished_returns
