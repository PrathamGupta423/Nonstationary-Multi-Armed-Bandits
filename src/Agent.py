from abc import ABC, abstractmethod
import numpy as np
from Environment import Bandit_Environment
from typing import Dict, Any
from scipy.optimize import brentq

class BanditAlgorithm(ABC):
    """Abstract base class for any bandit algorithm."""

    def __init__(self, n_arms: int, horizon: int, **kwargs):
        self.n_arms = n_arms
        self.horizon = horizon
        self.kwargs = kwargs  # Store extra kwargs for derived classes
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



# since we are only dealing with bernouli distribution
def KL_bernouli(p,q):
    if p == 0:
        return np.log(1/(1-q)) if q<1 else np.inf
    if p == 1:
        return np.log(1/q) if q>0 else np.inf
    return p * np.log(p/q) + (1 - p) * np.log((1 - p)/(1 - q))

# to detect change, we need adwin
class ADWIN:
    def __init__(self, delta=1e-3):
        self.delta= delta
        self.window = []
        self.width = 0 
        self.sum =0 
    def add_element(self, value):
        self.window.append(value)
        self.width += 1
        self.sum += value
        if self.detect_change():
            return True 
        return False
    def detect_change(self):
        if self.width < 2:
            return False
        for cut in range(1, self.width):
            n0 = cut
            n1 = self.width - cut
            mean0 = sum(self.window[:cut]) / n0 if n0 > 0 else 0
            mean1 = sum(self.window[cut:]) / n1 if n1 > 0 else 0
            eps_cut = np.sqrt(1 / (2 * n0)* np.log(1 / self.delta)) + np.sqrt(1 / (2 * n1)* np.log(1 / self.delta))
            if abs(mean0 - mean1) > eps_cut:
                self.window = self.window[cut:]
                self.width = len(self.window)
                self.sum = sum(self.window)
                return True
        return False 

    def get_mean(self):
        if self.width == 0:
            return 0
        return self.sum / self.width

    def get_samples(self):
        return self.window.copy()
class UCB(BanditAlgorithm):
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.delta = 1e-2
        self.t = 0
    
    def select_arm(self, t: int) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        ucb_values = self.values + np.sqrt((2 * np.log(self.t)) / self.counts)
        return int(np.argmax(ucb_values))
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm] = new_value
    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon)
        actions = np.zeros(self.horizon, dtype=int)

        env.reset_env()
        self.reset()

        for t in range(self.horizon):
            arm = self.select_arm(t)
            reward = env.pull(arm)
            self.update(arm, reward)

            actions[t] = arm
            rewards[t] = reward

        return {
            "actions": actions,
            "rewards": rewards,
            "cumulative_reward": rewards.sum(),
            "mean_reward": rewards.mean(),
            "counts": self.counts,
            "values": self.values,
        }
    

def create_ucb_agent(n_arms: int, horizon: int) -> UCB:
    return UCB(n_arms=n_arms, horizon=horizon)

class KL_UCB(BanditAlgorithm):
    """ KL-UCB for bernouli rewards"""
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0
        self.c = self.kwargs.get('c', 2)

    def kl_confidence(self, t, emp_mean, num_pulls):
        lower_bound = emp_mean
        upper_bound = 1.0
        precision = 1e-5
        max_iter = 50
        n = 0
        while n < max_iter and upper_bound - lower_bound > precision:
            q = (lower_bound + upper_bound) / 2
            if KL_bernouli(emp_mean, q) > (np.log(1+t * (np.log(t))**2) / num_pulls):
                upper_bound = q
            else:
                lower_bound = q
            n += 1
        return (lower_bound + upper_bound) / 2

    def select_arm(self, t: int) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self.kl_confidence(t, self.values[arm], self.counts[arm])
        return int(np.argmax(kl_ucb_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm] = new_value
    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon)
        actions = np.zeros(self.horizon, dtype=int)

        env.reset_env()
        self.reset()

        for t in range(self.horizon):
            arm = self.select_arm(t)
            reward = env.pull(arm)
            self.update(arm, reward)

            actions[t] = arm
            rewards[t] = reward

        return {
            "actions": actions,
            "rewards": rewards,
            "cumulative_reward": rewards.sum(),
            "mean_reward": rewards.mean(),
            "counts": self.counts,
            "values": self.values,
        }
def create_kl_ucb_agent(n_arms: int, horizon: int, c: float = 2) -> KL_UCB:
    return KL_UCB(n_arms=n_arms, horizon=horizon, c=c)

class RExp3(BanditAlgorithm):
    def reset(self):
        self.gamma = self.kwargs.get('gamma', 0.1)
        self.restart_interval = self.kwargs.get('restart_interval', 100)
        self.weights = np.ones(self.n_arms)
        self.t_internal = 0
        self.t = 0

    def select_arm(self):
        self.t +=1 
        if self.t_internal % self.restart_interval ==0:
            self.weights = np.ones(self.n_arms)
            self.t_internal =0
        probs = (1 - self.gamma) * (self.weights / self.weights.sum()) + (self.gamma / self.n_arms)
        self.t_internal += 1
        return int(np.random.choice(self.n_arms, p=probs))
    def update(self, arm: int, reward: float):
        probs = (1 - self.gamma) * (self.weights / self.weights.sum()) + (self.gamma / self.n_arms)
        x_hat = reward / probs[arm]
        growth_factor = np.exp((self.gamma * x_hat) / self.n_arms)
        self.weights[arm] *= growth_factor

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
            "weights": self.weights,
        }

def create_rexp3_agent(n_arms: int, horizon: int, gamma: float = 0.1, restart_interval: int = 100) -> RExp3:
    return RExp3(n_arms=n_arms, horizon=horizon, gamma=gamma, restart_interval=restart_interval)

class Discounted_UCB(BanditAlgorithm):
    def reset(self):
        self.gamma = self.kwargs.get('gamma', 0.9)
        self.c = self.kwargs.get('c', 2)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0 

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts )
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.counts *= self.gamma
        self.values *= self.gamma
        self.counts[arm] += 1
        self.values[arm] += reward
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
            "counts": self.counts,
            "values": self.values,
        }
def create_discounted_ucb_agent(n_arms: int, horizon: int, gamma: float = 0.9, c: float = 2) -> Discounted_UCB:
    return Discounted_UCB(n_arms=n_arms, horizon=horizon, gamma=gamma, c=c)


class Sliding_Window_Thompson_sampling(BanditAlgorithm):
    def reset(self):
        self.window_size = self.kwargs.get('window_size', 100)
        self.rewards = [[] for _ in range(self.n_arms)]
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self._update_posteriors()

    def _update_posteriors(self):
        for arm in range(self.n_arms):
            window_rewards = self.rewards[arm][-self.window_size:]
            self.alphas[arm] = 1 + sum(window_rewards)
            self.betas[arm] = 1 + len(window_rewards) - sum(window_rewards)
    def select_arm(self) -> int:
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))
    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        self._update_posteriors()
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
            "alphas": self.alphas,
            "betas": self.betas,
        }
def create_sliding_window_thompson_sampling_agent(n_arms: int, horizon: int, window_size: int = 100) -> Sliding_Window_Thompson_sampling:
    return Sliding_Window_Thompson_sampling(n_arms=n_arms, horizon=horizon, window_size=window_size)

class Sliding_Window_UCB(BanditAlgorithm):
    def reset(self):
        self.window_size = self.kwargs.get('window_size', 100)
        self.rewards = [[] for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0
        self.c = self.kwargs.get('c', 2)
    def _update_estimates(self):
        for i in range(self.n_arms):
            window_rewards = self.rewards[i][-self.window_size:]
            self.counts[i] = len(window_rewards)
            self.values[i] = np.mean(window_rewards) if window_rewards else 0

    def select_arm(self) -> int:
        self.t += 1
        self._update_estimates()
        if self.t <= self.n_arms:
            return self.t - 1
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / (self.counts + 1e-5))
        return int(np.argmax(ucb_values))
    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        self._update_estimates()
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
            "counts": self.counts,
            "values": self.values,
        }
def create_sliding_window_ucb_agent(n_arms: int, horizon: int, window_size: int = 100, c: float = 2) -> Sliding_Window_UCB:
    return Sliding_Window_UCB(n_arms=n_arms, horizon=horizon, window_size=window_size, c=c)

class GLR_KL_UCB(BanditAlgorithm):
    def reset(self):
        self.deltta = self.kwargs.get('delta', 1e-2)
        self.c = self.kwargs.get('c', 2)
        self.alpha_seq = self.kwargs.get('alpha_seq', lambda k: np.sqrt(k * self.n_arms * np.log(self.horizon) / self.horizon))  # Agnostic
        self.global_restart = self.kwargs.get('global_restart', False)
        self.klucb = KL_UCB(self.n_arms, self.horizon, c=self.c)  # Use internal KL_UCB
        self.rewards = [[] for _ in range(self.n_arms)]  # History per arm
        self.last_restart = np.zeros(self.n_arms, dtype=int)
        self.episode = 1
        self.t = 0
        self.forced_exploration_counter = 0

    def glr_test(self, arm):
        rewards = self.rewards[arm]
        n = len(rewards)
        if n<2:
            return False
        # Compute GLR Statistic
        mu_hat = np.mean(rewards)
        max_stat = 0
        for s in range(1, n):
            mu1 = np.mean(rewards[:s])
            mu2 = np.mean(rewards[s:])
            stat = s * KL_bernouli(mu1, mu_hat) + (n - s) * KL_bernouli(mu2, mu_hat)
            if stat > max_stat:
                max_stat = stat
        beta = np.log(n ** 1.5 / self.deltta)
        return max_stat > beta

    def is_forced_exploration(self):
        alpha_k = self.alpha_seq(self.episode)
        block_size = int(self.n_arms / alpha_k)
        if self.forced_exploration_counter < self.n_arms:
            return True
        if self.t % block_size == 0:
            self.forced_exploration_counter = 0
        self.forced_exploration_counter += 1
        return self.forced_exploration_counter <= self.n_arms
    
    def select_arm(self):
        self.t += 1
        if self.is_forced_exploration():
            arm = self.forced_exploration_counter - 1
            return arm
        return self.klucb.select_arm(self.t)

    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        if self.glr_test(arm):
            self.last_restart[arm] = self.t
            self.rewards[arm] = []
            if self.global_restart:
                for a in range(self.n_arms):
                    self.last_restart[a] = self.t
                    self.rewards[a] = []
            self.episode += 1
        self.klucb.update(arm, reward)
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
            "klucb_counts": self.klucb.counts,
            "klucb_values": self.klucb.values,
        }

def create_glr_kl_ucb_agent(n_arms: int, horizon: int, delta: float = 1e-2, c: float = 2, alpha_seq=None, global_restart: bool = False) -> GLR_KL_UCB:
    return GLR_KL_UCB(n_arms=n_arms, horizon=horizon, delta=delta, c=c, alpha_seq=alpha_seq, global_restart=global_restart)
    

class M_UCB(BanditAlgorithm):
    def reset(self):
        self.window_size = self.kwargs.get('window_size', 100)  
        self.b_threshold_factor = self.kwargs.get('b_factor', 1.0)  
        self.delta = self.kwargs.get('delta', 0.05)
        self.rewards = [[] for _ in range(self.n_arms)]  
        self.ucb = UCB(self.n_arms, self.horizon)  
        self.t = 0

    def change_detection(self, arm):
        recent = self.rewards[arm][-self.window_size:]
        w = len(recent)
        if w< self.window_size or w%2 !=0:
            return False
        mid = w // 2
        mean1 = np.mean(recent[:mid])
        mean2 = np.mean(recent[mid:])
        b = self.b_threshold_factor * np.sqrt((2 * np.log(2 / self.delta)) / mid)
        return abs(mean1 - mean2) > b
    def select_arm(self):
        self.t += 1
        return self.ucb.select_arm(self.t)
    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        if self.change_detection(arm):
            self.rewards[arm] = []
            self.ucb.counts[arm] = 0
            self.ucb.values[arm] = 0
        self.ucb.update(arm, reward)
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
            "ucb_counts": self.ucb.counts,
            "ucb_values": self.ucb.values,
        }
def create_m_ucb_agent(n_arms: int, horizon: int, window_size: int = 100, b_factor: float = 1.0, delta: float = 0.05) -> M_UCB:
    return M_UCB(n_arms=n_arms, horizon=horizon, window_size=window_size, b_factor=b_factor, delta=delta)

class UCBL_CPD(BanditAlgorithm):
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.rewards = [[] for _ in range(self.n_arms)]  
        self.cumsums = [[] for _ in range(self.n_arms)]  
        self.counts = np.zeros(self.n_arms, dtype=int) 
        self.values = np.zeros(self.n_arms, dtype=float)  
        self.internal_t = 0  

    def S(self,n):
        if n==0:
            return np.inf
        return np.sqrt((1 + 1/n) * np.log(np.sqrt(n + 1) / self.delta) / (2 * n))

    def select_arm(self):
        self.internal_t += 1
        if self.internal_t <= self.n_arms:
            return self.internal_t - 1
        ucb_values = self.values + [self.S(self.counts[arm]) for arm in range(self.n_arms)]
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        if self.cumsums[arm]:
            new_sum = self.cumsums[arm][-1] + reward
        else:
            new_sum = reward
        self.cumsums[arm].append(new_sum)
        self.counts[arm] += 1
        self.values[arm] = new_sum / self.counts[arm]

        # CPD check
        if self.cpd():
            # Global reset
            self.rewards = [[] for _ in range(self.n_arms)]
            self.cumsums = [[] for _ in range(self.n_arms)]
            self.counts = np.zeros(self.n_arms, dtype=int)
            self.values = np.zeros(self.n_arms, dtype=float)
            self.internal_t = 0  # Trigger initial exploration in next steps

    def cpd(self):
        for i in range(self.n_arms):
            n = self.counts[i]
            if n < 2:
                continue
            cum = self.cumsums[i]
            for k in range(1, n):
                mu1 = cum[k-1] / k
                S1 = self.S(k)
                mu2 = (cum[n-1] - cum[k-1]) / (n - k)
                S2 = self.S(n - k)
                if mu1 + S1 < mu2 - S2 or mu1 - S1 > mu2 + S2:
                    return True
        return False
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
            "counts": self.counts,
            "values": self.values,
        }
def create_ucbl_cpd_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> UCBL_CPD:
    return UCBL_CPD(n_arms=n_arms, horizon=horizon, delta=delta)



class ImpCPD(BanditAlgorithm):
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.gamma = self.kwargs.get('gamma', 0.5)
        self.alpha = self.kwargs.get('alpha', 1.5)
        self.alpha = self.kwargs.get('alpha', 1.5)
        self.psi = self.horizon**2 * self.n_arms**2 * np.log(self.n_arms)
        self.M = int(0.5 * np.log(self.horizon / np.e) / np.log(1 + self.gamma))
        self.rewards = [[] for _ in range(self.n_arms)]
        self.cumsums = [[] for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.internal_t = 0
        self.m = 0
        self.epsilon = 1.0
        self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
        self.L = np.full(self.M + 1, 0)
        self.L[0] = self.n_arms * self.ell
        self.B_size = self.n_arms
        self.phase_ends = []  

    def S(self, n, epsilon):
        if n == 0:
            return np.inf
        return np.sqrt(self.alpha * np.log(self.psi * epsilon**2) / (2 * n))

    def select_arm(self):
        self.internal_t += 1 
        if self.internal_t <= self.n_arms:
            return self.internal_t - 1
        ucb_values = self.values + [self.S(self.counts[arm], self.epsilon) for arm in range(self.n_arms)]
        return int(np.argmax(ucb_values))
    def update(self, arm: int, reward: float):
        self.rewards[arm].append(reward)
        if self.cumsums[arm]:
            new_sum = self.cumsums[arm][-1] + reward
        else:
            new_sum = reward
        self.cumsums[arm].append(new_sum)
        self.counts[arm] += 1
        self.values[arm] = new_sum / self.counts[arm]

        if self.internal_t >= self.L[self.m] and self.m <= self.M:
            if self.cpdi():
                # Reset
                self.rewards = [[] for _ in range(self.n_arms)]
                self.cumsums = [[] for _ in range(self.n_arms)]
                self.counts = np.zeros(self.n_arms, dtype=int)
                self.values = np.zeros(self.n_arms, dtype=float)
                self.internal_t = 0
                self.m = 0
                self.epsilon = 1.0
                self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
                self.L[0] = self.n_arms * self.ell
                self.phase_ends = []
                self.B_size = self.n_arms
            else:
                # Pseudo-elimination
                max_lcb = max(self.values[j] - self.S(self.counts[j], self.epsilon) for j in range(self.n_arms))
                for i in range(self.n_arms):
                    if self.values[i] + self.S(self.counts[i], self.epsilon) < max_lcb:
                        self.B_size -= 1
                # Update for next phase
                self.m += 1
                self.epsilon /= (1 + self.gamma)
                self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
                self.L[self.m] = self.internal_t + self.B_size * self.ell
                self.phase_ends.append(self.L[self.m - 1])

    def cpdi(self):
        for i in range(self.n_arms):
            n = self.counts[i]
            if n < 2:
                continue
            cum = self.cumsums[i]
            for k in range(1, n):
                mu1 = cum[k-1] / k
                S1 = self.S(k, self.epsilon)
                mu2 = (cum[n-1] - cum[k-1]) / (n - k)
                S2 = self.S(n - k, self.epsilon)
                if mu1 + S1 < mu2 - S2 or mu1 - S1 > mu2 + S2:
                    return True
        return False
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
            "counts": self.counts,
            "values": self.values,
        }
def create_impcpd_agent(n_arms: int, horizon: int, delta: float = 1e-2, gamma: float = 0.5, alpha: float = 1.5) -> ImpCPD:
    return ImpCPD(n_arms=n_arms, horizon=horizon, delta=delta, gamma=gamma, alpha=alpha)

class ADS_TS(BanditAlgorithm):
    """Adaptive Shrinking Thompson Sampling with per-arm ADWIN."""
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.adwins = [ADWIN(delta=self.delta) for _ in range(self.n_arms)]
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def _update_posterior(self, arm):
        window = self.adwins[arm].get_samples()
        self.alphas[arm] = 1 + sum(window)
        self.betas[arm] = 1 + len(window) - sum(window)

    def select_arm(self) -> int:
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.adwins[arm].add_element(reward)
        self._update_posterior(arm)

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
            "alpha": self.alphas,
            "beta": self.betas,
        }

def create_ads_ts_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> ADS_TS:
    return ADS_TS(n_arms=n_arms, horizon=horizon, delta=delta)


class ADS_UCB(BanditAlgorithm):
    """Adaptive Shrinking UCB with per-arm ADWIN."""
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.adwins = [ADWIN(delta=self.delta) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
            bonus = np.sqrt((2 * np.log(self.t)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.adwins[arm].add_element(reward)
        window = self.adwins[arm].get_samples()
        self.counts[arm] = len(window)
        self.values[arm] = np.mean(window) if window else 0

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
            "counts": self.counts,
            "values": self.values,
        }

def create_ads_ucb_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> ADS_UCB:
    return ADS_UCB(n_arms=n_arms, horizon=horizon, delta=delta)


class ADS_kl_UCB(BanditAlgorithm):
    """Adaptive Shrinking KL-UCB with per-arm ADWIN."""
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.adwins = [ADWIN(delta=self.delta) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0

    def kl_confidence(self, t, emp_mean, num_pulls):
        if num_pulls == 0:
            return 1.0
        lower_bound = emp_mean
        upper_bound = 1.0
        precision = 1e-5
        max_iter = 50
        n = 0
        log_term = np.log(1 + t * (np.log(t))**2)
        while n < max_iter and upper_bound - lower_bound > precision:
            q = (lower_bound + upper_bound) / 2
            if KL_bernouli(emp_mean, q) > (log_term / num_pulls):
                upper_bound = q
            else:
                lower_bound = q
            n += 1
        return (lower_bound + upper_bound) / 2

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self.kl_confidence(self.t, self.values[arm], self.counts[arm])
        return int(np.argmax(kl_ucb_values))

    def update(self, arm: int, reward: float):
        self.adwins[arm].add_element(reward)
        window = self.adwins[arm].get_samples()
        self.counts[arm] = len(window)
        self.values[arm] = np.mean(window) if window else 0

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
            "counts": self.counts,
            "values": self.values,
        }

def create_ads_kl_ucb_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> ADS_kl_UCB:
    return ADS_kl_UCB(n_arms=n_arms, horizon=horizon, delta=delta)