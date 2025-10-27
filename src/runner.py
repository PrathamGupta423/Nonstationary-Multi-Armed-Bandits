from Environment import (
    Bandit_Environment,
    create_static_env_paper,
    create_gradual_env_paper,
    create_abrupt_env_paper,
    create_locally_abrupt_env_paper
)
from Agent import (
    BanditAlgorithm,
    create_thompson_sampling_agent,
    create_ucb_agent,
    create_kl_ucb_agent,
    create_rexp3_agent,
    create_discounted_ucb_agent,
    create_sliding_window_thompson_sampling_agent,
    create_sliding_window_ucb_agent,
    create_glr_kl_ucb_agent,
    create_m_ucb_agent,
    create_ucbl_cpd_agent,
    create_impcpd_agent,
    create_ads_ts_agent,
    create_ads_kl_ucb_agent
)
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def run_experiment(Environment_generator, Agent_generator) -> dict:
    samples = 100
    time_horizons = [400 * i for i in range(1, 26)]
    results = {
        "time_horizons": time_horizons,
        "average_cumulative_regret": [],
        "std_cumulative_regret": [],
    }
    for H in time_horizons:
        cumulative_regrets = []
        for seed in range(samples):
            env = Environment_generator(seed=seed)
            arms = env.n_arms
            agent = Agent_generator(n_arms=arms, horizon=H)
            output = agent.run(env)
            ideal_reward = env.ideal_reward(Horizon=H)
            cumulative_regret = ideal_reward - output["cumulative_reward"]
            cumulative_regrets.append(cumulative_regret)
        avg_regret = np.mean(cumulative_regrets)
        std_regret = np.std(cumulative_regrets)
        results["average_cumulative_regret"].append(avg_regret)
        results["std_cumulative_regret"].append(std_regret)
        print(f"Horizon: {H}, Avg Regret: {avg_regret}, Std Regret: {std_regret}")
    return results

def run_experiment_grad(Environment_generator, Agent_generator) -> dict:
    samples = 100
    time_horizons = [400 * i for i in range(1, 26)]
    results = {
        "time_horizons": time_horizons,
        "average_cumulative_regret": [],
        "std_cumulative_regret": [],
    }
    for H in time_horizons:
        cumulative_regrets = []
        for seed in tqdm(range(samples)):
            env = Environment_generator(seed=seed, time_horizon=H)
            arms = env.n_arms
            agent = Agent_generator(n_arms=arms, horizon=H)
            output = agent.run(env)
            ideal_reward = env.ideal_reward(Horizon=H)
            cumulative_regret = ideal_reward - output["cumulative_reward"]
            cumulative_regrets.append(cumulative_regret)
        avg_regret = np.mean(cumulative_regrets)
        std_regret = np.std(cumulative_regrets)
        results["average_cumulative_regret"].append(avg_regret)
        results["std_cumulative_regret"].append(std_regret)
        print(f"Horizon: {H}, Avg Regret: {avg_regret}, Std Regret: {std_regret}")
    return results

# Define environments with names and their runner type
environments = [
    # {"name": "static", "generator": create_static_env_paper, "runner": run_experiment},
    {"name": "gradual", "generator": create_gradual_env_paper, "runner": run_experiment_grad},
    # {"name": "abrupt", "generator": create_abrupt_env_paper, "runner": run_experiment_grad},
    # {"name": "locally_abrupt", "generator": create_locally_abrupt_env_paper, "runner": run_experiment_grad},
]

# Define agents with names and generators (using defaults from agent.py)
agents = [
    # {"name": "ThompsonSampling", "generator": lambda n_arms, horizon: create_thompson_sampling_agent(n_arms, horizon)},
    # {"name": "UCB", "generator": lambda n_arms, horizon: create_ucb_agent(n_arms, horizon)},
    {"name": "KL_UCB", "generator": lambda n_arms, horizon: create_kl_ucb_agent(n_arms, horizon, c=2)},
    # {"name": "RExp3", "generator": lambda n_arms, horizon: create_rexp3_agent(n_arms, horizon, gamma=0.1, restart_interval=100)},
    # {"name": "Discounted_UCB", "generator": lambda n_arms, horizon: create_discounted_ucb_agent(n_arms, horizon, gamma=0.9, c=2)},
    # {"name": "Sliding_Window_Thompson_sampling", "generator": lambda n_arms, horizon: create_sliding_window_thompson_sampling_agent(n_arms, horizon, window_size=100)},
    # {"name": "Sliding_Window_UCB", "generator": lambda n_arms, horizon: create_sliding_window_ucb_agent(n_arms, horizon, window_size=100, c=2)},
    # {"name": "M_UCB", "generator": lambda n_arms, horizon: create_m_ucb_agent(n_arms, horizon, window_size=100, b_factor=1.0, delta=0.05)},
    # {"name": "UCBL_CPD", "generator": lambda n_arms, horizon: create_ucbl_cpd_agent(n_arms, horizon, delta=1e-2)},
    # {"name": "ImpCPD", "generator": lambda n_arms, horizon: create_impcpd_agent(n_arms, horizon, delta=1e-2, gamma=0.5, alpha=1.5)},
    # {"name": "ADS_TS", "generator": lambda n_arms, horizon: create_ads_ts_agent(n_arms, horizon, delta=1e-2)},
    # {"name": "ADS_kl_UCB", "generator": lambda n_arms, horizon: create_ads_kl_ucb_agent(n_arms, horizon, delta=1e-2)},
]
if __name__ == "__main__":
    for env in environments:
        env_name = env["name"]
        env_generator = env["generator"]
        runner_func = env["runner"]
        all_results = {}  # Still used for plotting, but not saved as single JSON

        for agent in agents:
            agent_name = agent["name"]
            agent_generator = agent["generator"]
            print(f"Running {agent_name} on {env_name} environment...")
            try:
                results = runner_func(
                    Environment_generator=env_generator,
                    Agent_generator=agent_generator,
                )
                # Save individual JSON for this agent-environment pair
                json_filename = f"{env_name}_{agent_name}_results.json"
                with open(json_filename, "w") as f:
                    json.dump(results, f)
                print(f"Saved results to {json_filename}")
                all_results[agent_name] = results  # Keep for plotting
            except Exception as e:
                print(f"Error in {agent_name} on {env_name}: {e}")
                # Continue to next, results for this agent are saved if completed

        # Plot all algos for this env (using collected results)
        plt.figure(figsize=(12, 8))
        for agent_name, results in all_results.items():
            plt.errorbar(
                results["time_horizons"],
                results["average_cumulative_regret"],
                yerr=results["std_cumulative_regret"],
                fmt='-o',
                label=agent_name
            )
        plt.xlabel("Time Horizon")
        plt.ylabel("Average Cumulative Regret")
        plt.title(f"Algorithm Comparison on {env_name.capitalize()} Bernoulli Bandit")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid()
        plt.tight_layout()
        plot_filename = f"{env_name}_all_algorithms_results.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot to {plot_filename}")