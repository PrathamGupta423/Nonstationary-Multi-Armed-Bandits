# Comparing Bandit Algorithms in Non-Stationary Environments 
This Repository is partly inspired by experiments done in [https://github.com/edouardfouche/G-NS-MAB](https://github.com/edouardfouche/G-NS-MAB) [Arxiv Version: https://arxiv.org/pdf/2107.11419] and developed as part of project assignment in E1 240 (Theory of Multi-armed Bandits) in AUG-DEC 2025.

## Overview

Multi-armed bandits are a classic reinforcement learning problem where an agent must choose between competing options (arms) over time to maximize its cumulative reward. This project explores how different algorithms adapt (or fail to adapt) when the underlying reward probabilities of the arms change over time.

We compare algorithms designed for stationary settings against those specifically developed for non-stationary environments, including methods based on discounting, sliding windows, and explicit change point detection.

## Components

### Environments (`src/Environment.py`)

A base `Bandit_Environment` class and specific implementations for Bernoulli rewards:

1.  **`StaticBernoulliBandit`**: Arm probabilities are fixed over time.
2.  **`GradualBernoulliBandit`**: Arm probabilities change linearly over the time horizon.
3.  **`AbruptBernoulliBandit`**: Arm probabilities change suddenly at specific points in time. Used for both globally abrupt and locally abrupt scenarios.

Helper functions (`create_*_env_paper`) set up specific environment configurations used in the experiments.

### Algorithms (`src/Agent.py`)

A base `BanditAlgorithm` class and implementations for various strategies:

* **Stationary Focused:**
    * `ThompsonSampling`: Basic Thompson Sampling with Beta priors.
    * `UCB`: Upper Confidence Bound algorithm.
    * `KL_UCB`: KL-Upper Confidence Bound (optimized for Bernoulli).
* **Non-Stationary Focused:**
    * `Discounted_UCB`: UCB variant using a discount factor (`gamma`).
    * `Sliding_Window_Thompson_sampling` (SW-TS): Thompson Sampling using only rewards from a recent window.
    * `Sliding_Window_UCB` (SW-UCB): UCB using only rewards from a recent window.
    * `RExp3`: Exponential-weight algorithm for exploration and exploitation, with restarts.
    * `M_UCB`: UCB combined with a simple change detection test on a sliding window.
    * `UCBL_CPD`: UCB-like algorithm with passive change point detection.
    * `ImpCPD`: Implicit change point detection using phase elimination.
    * `ADS_TS`: Adaptive Shrinking Thompson Sampling using ADWIN for change detection per arm.
    * `ADS_kl_UCB`: Adaptive Shrinking KL-UCB using ADWIN for change detection per arm.
    * `ADR_TS`: Adaptive Resetting Thompson Sampling using ADWIN for change detection per arm.
    * `ADR_kl_UCB`: Adaptive Resetting KL-UCB using ADWIN for change detection per arm.


Helper functions (`create_*_agent`) provide easy instantiation of agents.

### Experiment Runner (`src/runner.py`)

* Runs experiments comparing specified agents across different time horizons for a given environment.
* Calculates average cumulative regret and standard deviation over multiple sample runs (`samples=100`).
* Saves results for each agent-environment pair into JSON files in `src/JSON/`.
* Generates summary plots comparing all tested algorithms for each environment and saves them to the root directory (`*_all_algorithms_results.png`).

### Plotter (`src/plotter.ipynb`)

* A Jupyter Notebook to load results from the JSON files.
* Provides code to generate and customize performance comparison plots similar to those in the `PLOT/` directory.

## Results

* **Plots:** Pre-generated plots comparing the average cumulative regret of various algorithms in different environments can be found in the `PLOT/` directory.
    * `output.png`: Static Environment
    * `output4.png`: Gradual Environment
    * `output1.png`: Abrupt Environment
    * `output3.png`: Locally Abrupt Environment
* **Raw Data:** The raw data (average cumulative regret and standard deviation across runs for different horizons) is stored in `.json` files within the `src/JSON/` directory.

