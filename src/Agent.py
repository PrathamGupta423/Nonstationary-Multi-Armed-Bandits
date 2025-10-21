from abc import ABC, abstractmethod
import numpy as np
from Environment import Bandit_Environment
from typing import Dict, Any

class BanditAlgorithm(ABC):
    """Abstract base class for any bandit algorithm."""

    def __init__(self, n_arms: int, horizon: int):
        self.n_arms = n_arms
        self.horizon = horizon
        self.reset()

    @abstractmethod
    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        """Main loop: interact with env for horizon steps."""
        pass


class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling for Bernoulli rewards with Beta(1,1) priors."""

    def reset(self):
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon)
        actions = np.zeros(self.horizon, dtype=int)

        env.reset_env()
        self.reset()

        for t in range(self.horizon):
            arm = self.select_arm()
            reward = env.pull(arm)
            self.update(arm, reward)

            actions[t] = arm
            rewards[t] = reward

        return {
            "actions": actions,
            "rewards": rewards,
            "cumulative_reward": rewards.sum(),
            "mean_reward": rewards.mean(),
            "alpha": self.alpha,
            "beta": self.beta,
        }
    

def create_thompson_sampling_agent(n_arms: int, horizon: int) -> ThompsonSampling:
    return ThompsonSampling(n_arms=n_arms, horizon=horizon)

