from abc import ABC, abstractmethod
import numpy as np


class Bandit_Environment(ABC):
    
    def __init__(self, n_arms: int , seed: int = 23626):
        self.n_arms = n_arms
        self.seed = seed
        self.time = 0
        np.random.seed(self.seed)

    @abstractmethod
    def pull(self, arm: int) -> float:
        pass

    @abstractmethod
    def ideal_reward(self, Horizon: int) -> float:
        pass

    # @abstractmethod
    def reset_env(self):
        self.time = 0

    # @abstractmethod
    def step(self):
        self.t += 1


class StaticBernoulliBandit(Bandit_Environment):
    """
    Stationary Bernoulli bandit environment.
    Each arm i has a fixed success probability p_i in [0, 1].
    """

    def __init__(self, probs , seed: int = 23626):
        super().__init__(n_arms=len(probs), seed=seed)
        self.probs = np.array(probs, dtype=float)

    def pull(self, arm: int) -> float:
        reward = np.random.rand() < self.probs[arm]
        self.step()
        return float(reward)
    
    def ideal_reward(self, Horizon: int) -> float:
        best_prob = np.max(self.probs)
        return best_prob * Horizon

    def reset(self):
        super().reset()

    def step(self):
        # Static environment: nothing changes over time.
        self.time += 1


def create_static_env_paper(seed: int = 23626) -> StaticBernoulliBandit:
    probs = [(100-i)/100 for i in range(100)]
    return StaticBernoulliBandit(probs=probs, seed=seed)


class GradualBernoulliBandit(Bandit_Environment):


    def __init__(self, probs, time_horizon: int, seed: int = 23626):
        super().__init__(n_arms=len(probs), seed=seed)
        self.probs = np.array(probs, dtype=float)
        self.time_horizon = time_horizon

    def pull(self, arm: int) -> float:
        mu_a_t = self.probs[arm]*(self.time_horizon - self.time)/self.time_horizon + (1-self.probs[arm])*(self.time)/self.time_horizon
        reward = np.random.rand() < mu_a_t
        self.step()
        return float(reward)
    
    def ideal_reward(self, Horizon: int) -> float:
        ideal_reward = 0.0
        for t in range(Horizon):
            best_prob = np.max([self.probs[a]*(self.time_horizon - t)/self.time_horizon + (1-self.probs[a])*(t)/self.time_horizon for a in range(self.n_arms)])
            ideal_reward += best_prob
        return ideal_reward
    
    def reset(self):
        super().reset()

    def step(self):
        self.time += 1


def create_gradual_env_paper(seed, time_horizon: int) -> GradualBernoulliBandit:
    probs = [(100-i)/100 for i in range(100)]
    return GradualBernoulliBandit(probs=probs, time_horizon=time_horizon, seed=seed)


class AbruptBernoulliBandit(Bandit_Environment):
    
    def __init__(self, probs_list_0 , probs_list_1, time_horizon, seed: int = 23626):
        n_arms = len(probs_list_0)
        super().__init__(n_arms=n_arms, seed=seed)
        self.probs_list_0 = [np.array(probs, dtype=float) for probs in probs_list_0]
        self.probs_list_1 = [np.array(probs, dtype=float) for probs in probs_list_1]
        self.time_horizon = time_horizon

    def pull(self, arm: int) -> float:
        if self.time <= self.time_horizon /3 or self.time > 2*self.time_horizon /3:
            reward = np.random.rand() < self.probs_list_0[arm]
        else:
            reward = np.random.rand() < self.probs_list_1[arm]
        self.step()
        return float(reward)
    
    def ideal_reward(self, Horizon: int) -> float:
        best_prob_0 = np.max([self.probs_list_0[a] for a in range(self.n_arms)])
        best_prob_1 = np.max([self.probs_list_1[a] for a in range(self.n_arms)])
        ideal_reward = best_prob_0*(Horizon/3) + best_prob_1*(Horizon/3) + best_prob_0*(Horizon/3)
        return ideal_reward

    def reset(self):
        super().reset()

    def step(self):
        self.time += 1

def create_abrupt_env_paper(seed, time_horizon) -> AbruptBernoulliBandit:
    probs_list_0 = [(100-i)/100 for i in range(100)]
    probs_list_1 = [1- (100-i)/100 for i in range(100)]
    return AbruptBernoulliBandit(probs_list_0=probs_list_0, probs_list_1=probs_list_1, time_horizon=time_horizon, seed=seed)


def create_locally_abrupt_env_paper(seed: int = 23626, time_horizon: int = 10000) -> AbruptBernoulliBandit:
    probs_list_0 = [(100-i)/100 for i in range(100)]
    probs_list_1 = [0.5 for i in range(10)] + [(100-i)/100 for i in range(10,100)]
    return AbruptBernoulliBandit(probs_list_0=probs_list_0, probs_list_1=probs_list_1, time_horizon=time_horizon, seed=seed)