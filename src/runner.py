from Environment import Bandit_Environment
from Agent import BanditAlgorithm
import numpy as np
from tqdm import tqdm

def run_experiment(Environment_generator, Agent_generator) -> dict:

    samples = 100
    time_horizons = [400*i for i in range(1, 26)]
    results = {
        "time_horizons": time_horizons,
        "average_cumulative_regret": [],
        "std_cumulative_regret": [],
    }
    for H in time_horizons:
        cumulative_regrets = []
        for seed in range(samples):
            env = Environment_generator(seed = seed)
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
    time_horizons = [400*i for i in range(1, 26)]
    results = {
        "time_horizons": time_horizons,
        "average_cumulative_regret": [],
        "std_cumulative_regret": [],
    }
    for H in time_horizons:
        cumulative_regrets = []
        for seed in tqdm(range(samples)):
            env = Environment_generator(seed = seed, time_horizon=H)
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



def test_static_ts():
    from Environment import create_static_env_paper
    from Agent import create_thompson_sampling_agent

    results = run_experiment(
        Environment_generator=create_static_env_paper,
        Agent_generator=create_thompson_sampling_agent,
    )

    # save results to a file
    import json
    with open("static_experiment_results.json", "w") as f:
        json.dump(results, f)

    # plot results
    import matplotlib.pyplot as plt
    plt.errorbar(
        results["time_horizons"],
        results["average_cumulative_regret"],
        yerr=results["std_cumulative_regret"],
        fmt='-o'
    )
    plt.xlabel("Time Horizon")
    plt.ylabel("Average Cumulative Regret")
    plt.title("Thompson Sampling on Static Bernoulli Bandit")
    plt.grid()
    plt.savefig("static_experiment_results.png")

def gradual_ts():
    from Environment import create_gradual_env_paper
    from Agent import create_thompson_sampling_agent

    results = run_experiment_grad(
        Environment_generator=create_gradual_env_paper,
        Agent_generator=create_thompson_sampling_agent,
    )
    # save results to a file
    import json
    with open("gradual_experiment_results.json", "w") as f:
        json.dump(results, f)
    # plot results
    import matplotlib.pyplot as plt
    plt.errorbar(
        results["time_horizons"],
        results["average_cumulative_regret"],
        yerr=results["std_cumulative_regret"],
        fmt='-o'
    )
    plt.xlabel("Time Horizon")
    plt.ylabel("Average Cumulative Regret")
    plt.title("Thompson Sampling on Gradual Bernoulli Bandit")
    plt.grid()
    plt.savefig("gradual_experiment_results.png")

def test_abrupt_ts():

    from Environment import create_abrupt_env_paper
    from Agent import create_thompson_sampling_agent

    results = run_experiment_grad(
        Environment_generator=create_abrupt_env_paper,
        Agent_generator=create_thompson_sampling_agent,
    )

    # save results to a file
    import json
    with open("abrupt_experiment_results.json", "w") as f:
        json.dump(results, f)

    # plot results
    import matplotlib.pyplot as plt
    plt.errorbar(
        results["time_horizons"],
        results["average_cumulative_regret"],
        yerr=results["std_cumulative_regret"],
        fmt='-o'
    )
    plt.xlabel("Time Horizon")
    plt.ylabel("Average Cumulative Regret")
    plt.title("Thompson Sampling on Abrupt Bernoulli Bandit")
    plt.grid()
    plt.savefig("abrupt_experiment_results.png")

def test_locally_abrupt_ts():

    from Environment import create_locally_abrupt_env_paper
    from Agent import create_thompson_sampling_agent

    results = run_experiment_grad(
        Environment_generator=create_locally_abrupt_env_paper,
        Agent_generator=create_thompson_sampling_agent,
    )

    # save results to a file
    import json
    with open("locally_abrupt_experiment_results.json", "w") as f:
        json.dump(results, f)

    # plot results
    import matplotlib.pyplot as plt
    plt.errorbar(
        results["time_horizons"],
        results["average_cumulative_regret"],
        yerr=results["std_cumulative_regret"],
        fmt='-o'
    )
    plt.xlabel("Time Horizon")
    plt.ylabel("Average Cumulative Regret")
    plt.title("Thompson Sampling on Locally Abrupt Bernoulli Bandit")
    plt.grid()
    plt.savefig("locally_abrupt_experiment_results.png")


if __name__ == "__main__":
    test_static_ts()
    gradual_ts()
    test_abrupt_ts()
    test_locally_abrupt_ts()
