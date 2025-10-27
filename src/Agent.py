from abc import ABC, abstractmethod
import numpy as np
from Environment import Bandit_Environment
from collections import deque
from typing import Dict, Any, List, Tuple
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
    def __init__(self, n_arms, delta=1e-3, max_window=2000):
        self.n_arms = n_arms
        self.delta = delta
        self.max_window = max_window
        
        self.windows = np.zeros((n_arms, max_window), dtype=np.float32)
        self.window_sizes = np.zeros(n_arms, dtype=np.int32)
        self.cumsums = np.zeros((n_arms, max_window + 1), dtype=np.float32)
        
    def add_element(self, arm, reward):
        size = self.window_sizes[arm]
        
        if size < self.max_window:
            self.windows[arm, size] = reward
            self.cumsums[arm, size + 1] = self.cumsums[arm, size] + reward
            self.window_sizes[arm] += 1
        else:
            # Rare: window full, shift
            self.windows[arm, :-1] = self.windows[arm, 1:]
            self.windows[arm, -1] = reward
            self.cumsums[arm] = np.concatenate([[0], np.cumsum(self.windows[arm])])
        
        return self.detect_change(arm)
    
    def detect_change(self, arm):
        n = self.window_sizes[arm]
        if n < 2:
            return False
        
        cuts = np.arange(1, n)
        n0 = cuts
        n1 = n - cuts
        
        sum0 = self.cumsums[arm, cuts]
        sum1 = self.cumsums[arm, n] - sum0
        
        mean0 = sum0 / n0
        mean1 = sum1 / n1
        
        eps_cut = np.sqrt(0.5 / n0 * np.log(1 / self.delta)) + \
                  np.sqrt(0.5 / n1 * np.log(1 / self.delta))
        
        detected = np.abs(mean0 - mean1) > eps_cut
        
        if np.any(detected):
            cut = cuts[detected][0]
            new_size = n - cut
            
            self.windows[arm, :new_size] = self.windows[arm, cut:n]
            self.windows[arm, new_size:] = 0
            self.window_sizes[arm] = new_size
            
            self.cumsums[arm, 0] = 0
            self.cumsums[arm, 1:new_size+1] = np.cumsum(self.windows[arm, :new_size])
            self.cumsums[arm, new_size+1:] = 0
            
            return True
        
        return False
    
    def get_mean(self, arm):
        n = self.window_sizes[arm]
        return self.cumsums[arm, n] / n if n > 0 else 0
    
    def get_window(self, arm):
        n = self.window_sizes[arm]
        return self.windows[arm, :n]
    
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
    """ KL-UCB for Bernoulli rewards"""
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0
        self.c = self.kwargs.get('c', 2)

    def _kl_residual(self, q, emp_mean, c):
        """Residual function for root finding: KL(emp_mean, q) - c"""
        if q <= 0 or q >= 1:
            return np.inf  # Penalize out-of-bounds q
        return KL_bernouli(emp_mean, q) - c

    def kl_confidence(self, t, emp_mean, num_pulls):
        if num_pulls == 0:
            return 1.0
        c = np.log(1 + t * (np.log(t))**2) / num_pulls
        if emp_mean >= 1 - 1e-10:  # Handle near-boundary case
            return 1.0 - 1e-10
        elif emp_mean <= 1e-10:  # Handle near-zero case
            return 1e-10
        # Bracket the root between emp_mean and 1.0 - epsilon
        q = brentq(self._kl_residual, emp_mean, 1.0 - 1e-10, args=(emp_mean, c), xtol=1e-5)
        return q

    def select_arm(self, t: int) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self.kl_confidence(self.t, self.values[arm], self.counts[arm])
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
        self.weights = np.ones(self.n_arms, dtype=np.float32)
        self.t_internal = 0
        self.t = 0
        # Precompute constants
        self.one_minus_gamma = 1.0 - self.gamma
        self.gamma_over_k = self.gamma / self.n_arms
        self.gamma_over_k_exp = self.gamma / self.n_arms

    def select_arm(self):
        self.t += 1
        
        # Reset weights periodically
        if self.t_internal % self.restart_interval == 0:
            self.weights.fill(1.0)
            self.t_internal = 0
        
        # Compute probabilities (vectorized)
        weight_sum = self.weights.sum()
        probs = self.one_minus_gamma * (self.weights / weight_sum) + self.gamma_over_k
        
        self.t_internal += 1
        self.current_probs = probs  # Cache for update
        
        return int(np.random.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float):
        # Use cached probabilities
        x_hat = reward / self.current_probs[arm]
        growth_factor = np.exp(self.gamma_over_k_exp * x_hat)
        self.weights[arm] *= growth_factor

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "weights": self.weights.copy(),
        }

def create_rexp3_agent(n_arms: int, horizon: int, gamma: float = 0.1, restart_interval: int = 100) -> RExp3:
    return RExp3(n_arms=n_arms, horizon=horizon, gamma=gamma, restart_interval=restart_interval)

class Discounted_UCB(BanditAlgorithm):
    def reset(self):
        self.gamma = self.kwargs.get('gamma', 0.9)
        self.c = self.kwargs.get('c', 2)
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.t = 0
        # Precompute constant
        self.sqrt_c = np.sqrt(self.c)

    def select_arm(self) -> int:
        self.t += 1
        
        # Initial exploration
        if self.t <= self.n_arms:
            return self.t - 1
        
        # Vectorized UCB computation
        log_t = np.log(self.t)
        # Avoid division by zero
        safe_counts = np.maximum(self.counts, 1e-10)
        bonuses = self.c * np.sqrt(log_t / safe_counts)
        ucb_values = self.values + bonuses
        
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        # Vectorized discount
        self.counts *= self.gamma
        self.values *= self.gamma
        # Update selected arm
        self.counts[arm] += 1
        self.values[arm] += reward

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }
def create_discounted_ucb_agent(n_arms: int, horizon: int, gamma: float = 0.9, c: float = 2) -> Discounted_UCB:
    return Discounted_UCB(n_arms=n_arms, horizon=horizon, gamma=gamma, c=c)


class Sliding_Window_Thompson_sampling(BanditAlgorithm):
    def reset(self):
        self.window_size = self.kwargs.get('window_size', 100)
        # Use deque for O(1) append and popleft
        self.reward_deques = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]
        # Cache statistics
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.alphas = np.ones(self.n_arms, dtype=np.float32)
        self.betas = np.ones(self.n_arms, dtype=np.float32)

    def _update_posterior(self, arm):
        """Update posterior for one arm"""
        self.alphas[arm] = 1.0 + self.sums[arm]
        self.betas[arm] = 1.0 + self.counts[arm] - self.sums[arm]

    def select_arm(self) -> int:
        # Sample from all arms at once
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        deq = self.reward_deques[arm]
        
        # Check if we need to remove old reward
        was_full = len(deq) == self.window_size
        old_reward = 0.0
        if was_full:
            old_reward = deq[0]  # Will be removed by append
        
        # Add new reward (deque automatically removes oldest if at maxlen)
        deq.append(reward)
        
        # Update cached statistics (O(1) instead of O(window_size))
        if was_full:
            self.sums[arm] += reward - old_reward
            # counts stays same
        else:
            self.sums[arm] += reward
            self.counts[arm] += 1
        
        # Update posterior
        self._update_posterior(arm)

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "alphas": self.alphas.copy(),
            "betas": self.betas.copy(),
        }


def create_sliding_window_thompson_sampling_agent(n_arms: int, horizon: int, window_size: int = 100) -> Sliding_Window_Thompson_sampling:
    return Sliding_Window_Thompson_sampling(n_arms=n_arms, horizon=horizon, window_size=window_size)


class Sliding_Window_UCB(BanditAlgorithm):
    def reset(self):
        self.window_size = self.kwargs.get('window_size', 100)
        self.c = self.kwargs.get('c', 2)
        # Use deques for O(1) operations
        self.reward_deques = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]
        # Cache statistics
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        
        # Initial exploration
        if self.t <= self.n_arms:
            return self.t - 1
        
        # Vectorized UCB computation
        log_t = np.log(self.t)
        safe_counts = np.maximum(self.counts, 1e-10)
        bonuses = self.c * np.sqrt(log_t / safe_counts)
        ucb_values = self.values + bonuses
        
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        deq = self.reward_deques[arm]
        
        # Check if removing old value
        was_full = len(deq) == self.window_size
        old_reward = 0.0
        if was_full:
            old_reward = deq[0]
        
        # Add new reward
        deq.append(reward)
        
        # Update cached statistics (O(1))
        if was_full:
            self.sums[arm] += reward - old_reward
            # count stays same
        else:
            self.sums[arm] += reward
            self.counts[arm] += 1
        
        # Update mean
        if self.counts[arm] > 0:
            self.values[arm] = self.sums[arm] / self.counts[arm]

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }

def create_sliding_window_ucb_agent(n_arms: int, horizon: int, window_size: int = 100, c: float = 2) -> Sliding_Window_UCB:
    return Sliding_Window_UCB(n_arms=n_arms, horizon=horizon, window_size=window_size, c=c)


class GLR_KL_UCB(BanditAlgorithm):
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.c = self.kwargs.get('c', 2)
        self.global_restart = self.kwargs.get('global_restart', False)
        
        # Use deques for efficient history management
        self.reward_deques = [deque() for _ in range(self.n_arms)]
        
        # Cached statistics
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        
        self.last_restart = np.zeros(self.n_arms, dtype=np.int32)
        self.episode = 1
        self.t = 0
        self.forced_exploration_counter = 0
        
        # Precompute alpha sequence
        self.alpha_k = np.sqrt(self.episode * self.n_arms * np.log(10000) / 10000)

    def glr_test_optimized(self, arm):
        """Optimized GLR test using cached statistics"""
        n = self.counts[arm]
        if n < 2:
            return False
        
        rewards = np.array(self.reward_deques[arm], dtype=np.float32)
        mu_hat = self.values[arm]
        
        # Vectorized GLR computation
        max_stat = 0.0
        for s in range(1, n):
            mu1 = np.mean(rewards[:s])
            mu2 = np.mean(rewards[s:])
            
            # Compute KL divergences
            if 0 < mu1 < 1 and 0 < mu_hat < 1:
                kl1 = s * KL_bernouli(mu1, mu_hat)
            else:
                kl1 = 0
            
            if 0 < mu2 < 1 and 0 < mu_hat < 1:
                kl2 = (n - s) * KL_bernouli(mu2, mu_hat)
            else:
                kl2 = 0
            
            stat = kl1 + kl2
            if stat > max_stat:
                max_stat = stat
        
        beta = np.log(n ** 1.5 / self.delta)
        return max_stat > beta

    def _kl_residual(self, q, emp_mean, c):
        if q <= 0 or q >= 1:
            return np.inf
        return KL_bernouli(emp_mean, q) - c

    def kl_confidence(self, t, emp_mean, num_pulls):
        if num_pulls == 0:
            return 1.0
        c = np.log(1 + t * (np.log(t))**2) / num_pulls
        if emp_mean >= 1 - 1e-10:
            return 1.0 - 1e-10
        elif emp_mean <= 1e-10:
            return 1e-10
        try:
            q = brentq(self._kl_residual, emp_mean, 1.0 - 1e-10, 
                      args=(emp_mean, c), xtol=1e-5)
            return q
        except:
            return 1.0

    def select_arm(self):
        self.t += 1
        
        # Forced exploration
        block_size = max(1, int(self.n_arms / max(self.alpha_k, 0.01)))
        
        if self.forced_exploration_counter < self.n_arms:
            arm = self.forced_exploration_counter
            self.forced_exploration_counter += 1
            return arm
        
        if self.t % block_size == 0:
            self.forced_exploration_counter = 0
            return 0
        
        # KL-UCB selection (vectorized)
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self.kl_confidence(
                self.t, self.values[arm], self.counts[arm]
            )
        
        return int(np.argmax(kl_ucb_values))

    def update(self, arm: int, reward: float):
        # Add to deque
        self.reward_deques[arm].append(reward)
        
        # Update cached statistics
        self.sums[arm] += reward
        self.counts[arm] += 1
        self.values[arm] = self.sums[arm] / self.counts[arm]
        
        # GLR test (only every few updates to save time)
        if self.counts[arm] % 10 == 0 and self.glr_test_optimized(arm):
            self.last_restart[arm] = self.t
            self.reward_deques[arm].clear()
            self.sums[arm] = 0
            self.counts[arm] = 0
            self.values[arm] = 0
            
            if self.global_restart:
                for a in range(self.n_arms):
                    self.last_restart[a] = self.t
                    self.reward_deques[a].clear()
                    self.sums[a] = 0
                    self.counts[a] = 0
                    self.values[a] = 0
            
            self.episode += 1
            self.alpha_k = np.sqrt(self.episode * self.n_arms * np.log(10000) / 10000)

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts.copy(),
            "values": self.values.copy(),
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
        # Use deques
        self.reward_deques = [deque() for _ in range(self.n_arms)]
        self.cumsum_deques = [deque([0.0]) for _ in range(self.n_arms)]
        
        # Cached statistics
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.last_check = np.zeros(self.n_arms, dtype=np.int32)
        self.internal_t = 0
        
        # Cache S function evaluations
        self.sqrt_log_delta = np.sqrt(np.log(1.0 / self.delta))

    def S(self, n):
        """Cached S function"""
        if n == 0:
            return np.inf
        return np.sqrt((1 + 1/n) * np.log(np.sqrt(n + 1) / self.delta) / (2 * n))

    def select_arm(self):
        self.internal_t += 1
        if self.internal_t <= self.n_arms:
            return self.internal_t - 1
        
        # Vectorized UCB computation
        ucb_values = np.array([
            self.values[arm] + self.S(self.counts[arm]) 
            for arm in range(self.n_arms)
        ])
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        # Update deques
        self.reward_deques[arm].append(reward)
        new_sum = self.cumsum_deques[arm][-1] + reward
        self.cumsum_deques[arm].append(new_sum)
        
        # Update cached statistics
        self.sums[arm] = new_sum
        self.counts[arm] += 1
        self.values[arm] = new_sum / self.counts[arm]

        # CPD check (only at doubling points)
        if self.counts[arm] >= 2 * self.last_check[arm] or self.last_check[arm] == 0:
            if self.cpd_vectorized(arm):
                # Global reset
                for a in range(self.n_arms):
                    self.reward_deques[a].clear()
                    self.cumsum_deques[a] = deque([0.0])
                    self.sums[a] = 0
                    self.counts[a] = 0
                    self.values[a] = 0
                    self.last_check[a] = 0
                self.internal_t = 0
            else:
                self.last_check[arm] = self.counts[arm]

    def cpd_vectorized(self, arm):
        """Vectorized CPD check"""
        n = self.counts[arm]
        if n < 2:
            return False
        
        cum = np.array(self.cumsum_deques[arm])
        k = np.arange(1, n)
        
        mu1 = cum[k] / k
        mu2 = (cum[n] - cum[k]) / (n - k)
        
        # Vectorized S computation
        S1 = np.sqrt((1 + 1/k) * np.log(np.sqrt(k + 1) / self.delta) / (2 * k))
        S2 = np.sqrt((1 + 1/(n-k)) * np.log(np.sqrt((n-k) + 1) / self.delta) / (2 * (n-k)))
        
        # Vectorized check
        return np.any((mu1 + S1 < mu2 - S2) | (mu1 - S1 > mu2 + S2))

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }

def create_ucbl_cpd_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> UCBL_CPD:
    return UCBL_CPD(n_arms=n_arms, horizon=horizon, delta=delta)


class ImpCPD(BanditAlgorithm):
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.gamma = self.kwargs.get('gamma', 0.5)
        self.alpha = self.kwargs.get('alpha', 1.5)
        self.psi = self.horizon**2 * self.n_arms**2 * np.log(self.n_arms)
        self.M = int(0.5 * np.log(self.horizon / np.e) / np.log(1 + self.gamma))
        
        # Use deques
        self.reward_deques = [deque() for _ in range(self.n_arms)]
        self.cumsum_deques = [deque([0.0]) for _ in range(self.n_arms)]
        
        # Cached statistics
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.internal_t = 0
        self.m = 0
        self.epsilon = 1.0
        self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
        self.L = np.full(self.M + 1, 0.0)
        self.L[0] = self.n_arms * self.ell
        self.B_size = self.n_arms
        self.phase_ends = []
        self.last_check = np.zeros(self.n_arms, dtype=np.int32)

    def S(self, n, epsilon):
        if n == 0:
            return np.inf
        return np.sqrt(self.alpha * np.log(self.psi * epsilon**2) / (2 * n))

    def select_arm(self):
        self.internal_t += 1
        if self.internal_t <= self.n_arms:
            return self.internal_t - 1
        
        # Vectorized selection
        ucb_values = self.values + np.array([
            self.S(self.counts[arm], self.epsilon) 
            for arm in range(self.n_arms)
        ])
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        # Update deques
        self.reward_deques[arm].append(reward)
        new_sum = self.cumsum_deques[arm][-1] + reward
        self.cumsum_deques[arm].append(new_sum)
        
        # Update statistics
        self.sums[arm] = new_sum
        self.counts[arm] += 1
        self.values[arm] = new_sum / self.counts[arm]
        
        run_cpd = False
        if self.counts[arm] >= 2 * self.last_check[arm] or self.last_check[arm] == 0:
            self.last_check[arm] = self.counts[arm]
            run_cpd = True

        if self.internal_t >= self.L[self.m] and self.m <= self.M and run_cpd:
            if self.cpdi_vectorized(arm):
                # Reset
                self.last_check = np.zeros(self.n_arms, dtype=np.int32)
                for a in range(self.n_arms):
                    self.reward_deques[a].clear()
                    self.cumsum_deques[a] = deque([0.0])
                self.sums.fill(0)
                self.counts.fill(0)
                self.values.fill(0)
                self.internal_t = 0
                self.m = 0
                self.epsilon = 1.0
                self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
                self.L[0] = self.n_arms * self.ell
                self.phase_ends = []
                self.B_size = self.n_arms
            else:
                # Pseudo-elimination (vectorized)
                S_values = np.array([self.S(self.counts[j], self.epsilon) for j in range(self.n_arms)])
                max_lcb = np.max(self.values - S_values)
                self.B_size = np.sum(self.values + S_values >= max_lcb)
                
                # Update for next phase
                self.m += 1
                self.epsilon /= (1 + self.gamma)
                self.ell = np.ceil(np.log(self.psi * self.epsilon**2) / (2 * self.epsilon))
                self.L[self.m] = self.internal_t + self.B_size * self.ell
                self.phase_ends.append(self.L[self.m - 1])

    def cpdi_vectorized(self, arm):
        """Vectorized CPDI check"""
        n = self.counts[arm]
        if n < 2:
            return False
        
        cum = np.array(self.cumsum_deques[arm])
        k = np.arange(1, n)
        
        mu1 = cum[k] / k
        mu2 = (cum[n] - cum[k]) / (n - k)
        
        # Vectorized S computation
        S1 = np.sqrt(self.alpha * np.log(self.psi * self.epsilon**2) / (2 * k))
        S2 = np.sqrt(self.alpha * np.log(self.psi * self.epsilon**2) / (2 * (n - k)))
        
        return np.any((mu1 + S1 < mu2 - S2) | (mu1 - S1 > mu2 + S2))

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }
def create_impcpd_agent(n_arms: int, horizon: int, delta: float = 1e-2, gamma: float = 0.5, alpha: float = 1.5) -> ImpCPD:
    return ImpCPD(n_arms=n_arms, horizon=horizon, delta=delta, gamma=gamma, alpha=alpha)


class ADS_TS(BanditAlgorithm):
    """Adaptive Shrinking Thompson Sampling with per-arm ADWIN."""
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        # Single multi-armed ADWIN instead of K separate instances
        self.ma_adwin = ADWIN(
            n_arms=self.n_arms, 
            delta=self.delta,
            max_window=2000
        )
        self.alphas = np.ones(self.n_arms, dtype=np.float32)
        self.betas = np.ones(self.n_arms, dtype=np.float32)
        # Cached statistics for incremental updates
        self.successes = np.zeros(self.n_arms, dtype=np.float32)
        self.counts = np.zeros(self.n_arms, dtype=np.int32)

    def select_arm(self) -> int:
        # Sample from all Beta distributions at once
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        # Check for change
        changed = self.ma_adwin.add_element(arm, reward)
        
        if changed:
            # Only recompute when change detected
            window = self.ma_adwin.get_window(arm)
            self.successes[arm] = np.sum(window)
            self.counts[arm] = len(window)
        else:
            # Fast incremental update (no recomputation!)
            self.successes[arm] += reward
            self.counts[arm] += 1
        
        # Update posterior
        self.alphas[arm] = 1 + self.successes[arm]
        self.betas[arm] = 1 + self.counts[arm] - self.successes[arm]

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "alpha": self.alphas,
            "beta": self.betas,
        }



def create_ads_ts_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> ADS_TS:
    return ADS_TS(n_arms=n_arms, horizon=horizon, delta=delta)


class ADS_kl_UCB(BanditAlgorithm):
    """Adaptive Shrinking KL-UCB with per-arm ADWIN."""
    def reset(self):
        self.delta = self.kwargs.get('delta', 1e-2)
        self.ma_adwin = ADWIN(
            n_arms=self.n_arms,
            delta=self.delta,
            max_window=2000
        )
        self.counts = np.zeros(self.n_arms, dtype=np.int32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.sums = np.zeros(self.n_arms, dtype=np.float32)
        self.t = 0

    def _kl_residual(self, q, emp_mean, c):
        if q <= 0 or q >= 1:
            return np.inf
        return KL_bernouli(emp_mean, q) - c

    def kl_confidence(self, t, emp_mean, num_pulls):
        if num_pulls == 0:
            return 1.0
        c = np.log(1 + t * (np.log(t))**2) / num_pulls
        if emp_mean >= 1 - 1e-10:
            return 1.0 - 1e-10
        elif emp_mean <= 1e-10:
            return 1e-10
        from scipy.optimize import brentq
        q = brentq(self._kl_residual, emp_mean, 1.0 - 1e-10, 
                   args=(emp_mean, c), xtol=1e-5)
        return q

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        
        # Vectorize KL-UCB computation where possible
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self.kl_confidence(
                self.t, self.values[arm], self.counts[arm]
            )
        return int(np.argmax(kl_ucb_values))

    def update(self, arm: int, reward: float):
        changed = self.ma_adwin.add_element(arm, reward)
        
        if changed:
            window = self.ma_adwin.get_window(arm)
            self.counts[arm] = len(window)
            self.sums[arm] = np.sum(window)
            self.values[arm] = self.sums[arm] / self.counts[arm] if self.counts[arm] > 0 else 0
        else:
            self.sums[arm] += reward
            self.counts[arm] += 1
            self.values[arm] = self.sums[arm] / self.counts[arm]

    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        rewards = np.zeros(self.horizon, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)

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
            "cumulative_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "counts": self.counts,
            "values": self.values,
        }

def create_ads_kl_ucb_agent(n_arms: int, horizon: int, delta: float = 1e-2) -> ADS_kl_UCB:
    return ADS_kl_UCB(n_arms=n_arms, horizon=horizon, delta=delta)



class Bucket:
    """Bucket for ADWIN2 exponential histogram."""
    def __init__(self, capacity: int, content: float):
        self.capacity = capacity  # Number of elements
        self.content = content    # Sum of elements
    
    def __repr__(self):
        return f"Bucket(cap={self.capacity}, content={self.content})"

class ADWIN2:
    """
    ADWIN2: Efficient adaptive windowing using exponential histograms.
    Memory: O(M * log(W/M)) where W is window size, M is a parameter
    Time: O(log W) amortized per element
    """
    def __init__(self, delta=1e-3, M=5):
        self.delta = delta
        self.M = M  # Controls memory/time tradeoff
        self.buckets = []  # List of buckets
        self.total = 0.0  # Sum of all elements
        self.width = 0    # Total number of elements
        self.variance = 0.0  # For variance calculation
        self.sum_sq = 0.0    # Sum of squares
    
    def add_element(self, value: float) -> bool:
        """
        Add a new element and check for change.
        Returns True if change detected (window was shrunk).
        """
        # Create new bucket with single element
        self.buckets.insert(0, Bucket(1, value))
        self.total += value
        self.sum_sq += value * value
        self.width += 1
        
        # Compress buckets (merge buckets of same capacity)
        self._compress_buckets()
        
        # Update variance
        if self.width > 0:
            mean = self.total / self.width
            self.variance = (self.sum_sq / self.width) - (mean * mean)
            self.variance = max(0, self.variance)  # Numerical stability
        
        # Check for change
        change_detected = self._detect_change()
        
        return change_detected
    
    def _compress_buckets(self):
        """Merge buckets to maintain exponential histogram structure."""
        # Repeatedly merge until we have at most M+1 buckets of each capacity
        changed = True
        while changed:
            changed = False
            # Count buckets of each capacity
            capacity_count = {}
            for bucket in self.buckets:
                capacity_count[bucket.capacity] = capacity_count.get(bucket.capacity, 0) + 1
            
            # Find capacity with more than M+1 buckets
            for capacity, count in capacity_count.items():
                if count > self.M + 1:
                    # Find the two oldest (rightmost) buckets with this capacity
                    indices = [i for i in range(len(self.buckets)) if self.buckets[i].capacity == capacity]
                    
                    if len(indices) >= 2:
                        # Merge the two oldest buckets (highest indices)
                        idx1 = indices[-1]
                        idx2 = indices[-2]
                        
                        new_capacity = self.buckets[idx1].capacity + self.buckets[idx2].capacity
                        new_content = self.buckets[idx1].content + self.buckets[idx2].content
                        
                        # Remove the older bucket and update the newer one
                        self.buckets.pop(idx1)
                        self.buckets[idx2] = Bucket(new_capacity, new_content)
                        
                        changed = True
                        break  # Restart the process
    
    def _detect_change(self) -> bool:
        """
        Check all possible cuts for distribution change.
        Returns True if change detected and window was shrunk.
        """
        if self.width < 2:
            return False
        
        # We check cuts at exponentially spaced positions
        # corresponding to bucket boundaries
        n1 = 0
        sum1 = 0.0
        sum1_sq = 0.0
        
        for i in range(len(self.buckets)):
            n1 += self.buckets[i].capacity
            sum1 += self.buckets[i].content
            
            # For sum of squares approximation
            mean_bucket = self.buckets[i].content / self.buckets[i].capacity
            sum1_sq += self.buckets[i].capacity * (mean_bucket * mean_bucket)
            
            if n1 >= self.width - 1:
                break
            
            n0 = self.width - n1
            sum0 = self.total - sum1
            
            # Calculate means
            mean0 = sum0 / n0 if n0 > 0 else 0
            mean1 = sum1 / n1 if n1 > 0 else 0
            
            # Calculate threshold
            if self._check_cut(n0, n1, mean0, mean1):
                # Drop the oldest n0 elements
                self._shrink_to_size(n1)
                return True
        
        return False
    
    def _check_cut(self, n0: int, n1: int, mean0: float, mean1: float) -> bool:
        """
        Check if the difference between two subwindows is significant.
        Uses the more practical formula with variance (Eq 2.1 in paper).
        """
        if n0 <= 0 or n1 <= 0:
            return False
        
        # Harmonic mean
        m = 1.0 / (1.0/n0 + 1.0/n1)
        
        # Adjusted delta for multiple hypothesis testing
        # Using ln(n) instead of n for less conservative bound
        delta_prime = self.delta / np.log(self.width) if self.width > 1 else self.delta
        delta_prime = max(delta_prime, 1e-10)  # Numerical stability
        
        # Calculate threshold using variance (Equation 2.1 from paper)
        sigma_sq = self.variance if self.variance > 0 else 0.25  # Use worst-case if variance is 0
        
        epsilon_cut = np.sqrt(2 * sigma_sq * np.log(2/delta_prime) / m)
        epsilon_cut += (2.0 / (3.0 * m)) * np.log(2/delta_prime)
        
        # Check if difference is significant
        return abs(mean0 - mean1) >= epsilon_cut
    
    def _shrink_to_size(self, new_width: int):
        """Remove oldest elements to reduce window to new_width."""
        removed = 0
        
        while removed < self.width - new_width and len(self.buckets) > 0:
            oldest = self.buckets[-1]
            
            if removed + oldest.capacity <= self.width - new_width:
                # Remove entire bucket
                self.buckets.pop()
                self.total -= oldest.content
                
                # Update sum of squares (approximation)
                mean_bucket = oldest.content / oldest.capacity
                self.sum_sq -= oldest.capacity * (mean_bucket * mean_bucket)
                
                removed += oldest.capacity
            else:
                # Partially remove bucket
                elements_to_remove = self.width - new_width - removed
                mean_bucket = oldest.content / oldest.capacity
                content_to_remove = mean_bucket * elements_to_remove
                
                oldest.capacity -= elements_to_remove
                oldest.content -= content_to_remove
                self.total -= content_to_remove
                self.sum_sq -= elements_to_remove * (mean_bucket * mean_bucket)
                
                removed += elements_to_remove
        
        self.width = new_width
        
        # Update variance
        if self.width > 0:
            mean = self.total / self.width
            self.variance = (self.sum_sq / self.width) - (mean * mean)
            self.variance = max(0, self.variance)
    
    def get_mean(self) -> float:
        """Get current mean estimate."""
        if self.width == 0:
            return 0.0
        return self.total / self.width
    
    def get_width(self) -> int:
        """Get current window size."""
        return self.width
    
    def get_samples(self) -> List[float]:
        """
        Approximate reconstruction of samples (for compatibility).
        Note: ADWIN2 doesn't store individual samples, so this is an approximation.
        """
        samples = []
        for bucket in reversed(self.buckets):
            mean_val = bucket.content / bucket.capacity
            samples.extend([mean_val] * bucket.capacity)
        return samples

class ADR_TS:
    """ADR-bandit with Thompson Sampling as base algorithm, using ADWIN2."""
    
    def __init__(self, n_arms: int, horizon: int, delta=10**(-3), N=None, M=5):
        self.n_arms = n_arms
        self.horizon = horizon
        self.delta = delta
        self.M = M  # ADWIN2 parameter
        # Monitoring parameter N (if not provided, use a default based on horizon)
        self.N = N if N is not None else max(10, int(np.sqrt(horizon / n_arms)))
        self.reset()
    
    def reset(self):
        """Reset the entire algorithm."""
        self.adwins = [ADWIN2(delta=self.delta, M=self.M) for _ in range(self.n_arms)]
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self.t = 0
        self.block = 1
        self.monitoring_arm_current = None
        self.monitoring_arm_previous = None
        self.block_start = 0
        self.subblock_count = 0
        self.pulls_per_arm = np.zeros(self.n_arms)
    
    def _update_posterior(self, arm):
        """Update Beta posterior for a given arm."""
        samples = self.adwins[arm].get_samples()
        self.alphas[arm] = 1 + sum(samples)
        self.betas[arm] = 1 + len(samples) - sum(samples)
    
    def _select_monitoring_arm(self):
        """Select the monitoring arm (most pulled arm in current block)."""
        return int(np.argmax(self.pulls_per_arm))
    
    def _is_monitoring_round(self):
        """Check if current round should pull a monitoring arm."""
        if self.block == 1:
            return False
        
        rounds_in_block = self.t - self.block_start
        
        if rounds_in_block % self.n_arms == 0:
            return True
        elif rounds_in_block % self.n_arms == 1:
            return True
        return False
    
    def _get_monitoring_arm(self):
        """Get which monitoring arm to pull."""
        rounds_in_block = self.t - self.block_start
        if rounds_in_block % self.n_arms == 0:
            return self.monitoring_arm_previous
        else:
            return self.monitoring_arm_current
    
    def select_arm(self):
        """Select arm using Thompson Sampling or monitoring."""
        # Check if we should update monitoring arm
        if self.block == 1 and self.t == self.n_arms * self.N:
            self.monitoring_arm_current = self._select_monitoring_arm()
            self.pulls_per_arm = np.zeros(self.n_arms)
            self.block_start = self.t
            self.block = 2
        elif self.block >= 2:
            block_size = self.n_arms * self.N * (2 ** (self.block - 1))
            if self.t >= self.block_start + block_size:
                self.monitoring_arm_previous = self.monitoring_arm_current
                self.monitoring_arm_current = self._select_monitoring_arm()
                self.pulls_per_arm = np.zeros(self.n_arms)
                self.block_start = self.t
                self.block += 1
            elif self.t == self.block_start + (2 ** (self.block - 2) - 2) * self.n_arms * self.N:
                self.monitoring_arm_current = self._select_monitoring_arm()
        
        # Check if this is a monitoring round
        if self._is_monitoring_round() and self.block >= 2:
            arm = self._get_monitoring_arm()
            if arm is not None:
                return arm
        
        # Otherwise, use Thompson Sampling
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        """Update the algorithm with observed reward."""
        self.pulls_per_arm[arm] += 1
        
        # Add to ADWIN2 and check for change
        change_detected = self.adwins[arm].add_element(reward)
        
        if change_detected:
            # Global reset
            self.reset()
        else:
            # Update posterior for the pulled arm
            self._update_posterior(arm)
        
        self.t += 1
    
    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        """Run the algorithm for the full horizon."""
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

class ADR_KL_UCB:
    """ADR-bandit with KL-UCB as base algorithm, using ADWIN2."""
    
    def __init__(self, n_arms: int, horizon: int, delta=1e-3, N=None, M=5):
        self.n_arms = n_arms
        self.horizon = horizon
        self.delta = delta
        self.M = M
        self.N = N if N is not None else max(10, int(np.sqrt(horizon / n_arms)))
        self.reset()
    
    def reset(self):
        """Reset the entire algorithm."""
        self.adwins = [ADWIN2(delta=self.delta, M=self.M) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.t = 0
        self.block = 1
        self.monitoring_arm_current = None
        self.monitoring_arm_previous = None
        self.block_start = 0
        self.pulls_per_arm = np.zeros(self.n_arms)
    
    def _kl_confidence(self, t, emp_mean, num_pulls):
        """Compute KL-UCB upper confidence bound."""
        if num_pulls == 0:
            return 1.0
        
        lower_bound = emp_mean
        upper_bound = 1.0
        precision = 1e-5
        max_iter = 50
        log_term = np.log(1 + t * (np.log(max(t, 2)))**2)
        
        for _ in range(max_iter):
            if upper_bound - lower_bound <= precision:
                break
            q = (lower_bound + upper_bound) / 2
            if KL_bernouli(emp_mean, q) > (log_term / num_pulls):
                upper_bound = q
            else:
                lower_bound = q
        
        return (lower_bound + upper_bound) / 2
    
    def _select_monitoring_arm(self):
        """Select the monitoring arm (most pulled arm in current block)."""
        return int(np.argmax(self.pulls_per_arm))
    
    def _is_monitoring_round(self):
        """Check if current round should pull a monitoring arm."""
        if self.block == 1:
            return False
        
        rounds_in_block = self.t - self.block_start
        
        if rounds_in_block % self.n_arms == 0:
            return True
        elif rounds_in_block % self.n_arms == 1:
            return True
        return False
    
    def _get_monitoring_arm(self):
        """Get which monitoring arm to pull."""
        rounds_in_block = self.t - self.block_start
        if rounds_in_block % self.n_arms == 0:
            return self.monitoring_arm_previous
        else:
            return self.monitoring_arm_current
    
    def select_arm(self):
        """Select arm using KL-UCB or monitoring."""
        # Check if we should update monitoring arm
        if self.block == 1 and self.t == self.n_arms * self.N:
            self.monitoring_arm_current = self._select_monitoring_arm()
            self.pulls_per_arm = np.zeros(self.n_arms)
            self.block_start = self.t
            self.block = 2
        elif self.block >= 2:
            block_size = self.n_arms * self.N * (2 ** (self.block - 1))
            if self.t >= self.block_start + block_size:
                self.monitoring_arm_previous = self.monitoring_arm_current
                self.monitoring_arm_current = self._select_monitoring_arm()
                self.pulls_per_arm = np.zeros(self.n_arms)
                self.block_start = self.t
                self.block += 1
            elif self.t == self.block_start + (2 ** (self.block - 2) - 2) * self.n_arms * self.N:
                self.monitoring_arm_current = self._select_monitoring_arm()
        
        # Check if this is a monitoring round
        if self._is_monitoring_round() and self.block >= 2:
            arm = self._get_monitoring_arm()
            if arm is not None:
                return arm
        
        # Initial exploration: pull each arm once
        if self.t < self.n_arms:
            return self.t
        
        # Otherwise, use KL-UCB
        kl_ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            kl_ucb_values[arm] = self._kl_confidence(self.t, self.values[arm], self.counts[arm])
        
        return int(np.argmax(kl_ucb_values))
    
    def update(self, arm: int, reward: float):
        """Update the algorithm with observed reward."""
        self.pulls_per_arm[arm] += 1
        
        # Add to ADWIN2 and check for change
        change_detected = self.adwins[arm].add_element(reward)
        
        if change_detected:
            # Global reset
            self.reset()
        else:
            # Update estimates from ADWIN2
            self.counts[arm] = self.adwins[arm].get_width()
            self.values[arm] = self.adwins[arm].get_mean()
        
        self.t += 1
    
    def run(self, env: Bandit_Environment) -> Dict[str, Any]:
        """Run the algorithm for the full horizon."""
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

def create_adr_ts_agent(n_arms: int, horizon: int, delta: float = 1e-4, N: int = 10) -> ADR_TS:
    """Factory function for ADR-TS agent."""
    return ADR_TS(n_arms=n_arms, horizon=horizon, delta=delta, N=N)

def create_adr_kl_ucb_agent(n_arms: int, horizon: int, delta: float = 1e-4, N: int = 10) -> ADR_KL_UCB:
    """Factory function for ADR-KL-UCB agent."""
    return ADR_KL_UCB(n_arms=n_arms, horizon=horizon, delta=delta, N=N)