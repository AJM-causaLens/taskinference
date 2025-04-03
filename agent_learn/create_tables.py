import os
import matplotlib.pyplot as plt
from policy_optimization.suite_helpers import get_event_files_from_dir, get_mean_and_std_performance_from_event_files

# metrics = ["rollout/ep_rew_mean", "test_envs/mean_rewards", "test_envs/max_rewards"]
# metrics = ["rollout/ep_rew_mean"]
metrics = ["test_envs/mean_rewards"]
# metrics = ["rollout/ep_rew_mean", "test_envs/mean_rewards"]
metric_labels = {"rollout/ep_rew_mean": "Training Reward", "test_envs/mean_rewards": "Mean Test Reward", "test_envs/max_rewards": "Max Test Reward"}
env_labels = {"point_mass_easy": "Point Mass Easy", "point_mass_hard": "Point Mass Hard", "HalfCheetahVel_easy": "HalfCheetah Easy", "HalfCheetahVel_hard": "HalfCheetahVel (Sparse)"}
agent_labels = {"naive": "Naive", "oracle": "Oracle", "belief": "Belief Net",
                "tibbe": "Tibbe", "moss": "MoSS", "varibad": "VariBAD", "tibbe_alpha": "Tibbe Alpha", "tibbe_beta": "Tibbe Beta", "ti": "TaskInf"}
confidence_interval = 0.2

exclude = ["oracle", "tibbe_alpha", "tibbe_beta", "tibbe_BBE", "tibbe_alpha_BBE", "tibbe_beta_BBE", "tibbe"]
# exclude = ["oracle", "naive", "belief", "tibbe", "moss"]
# exclude = []


def plot_performance(exclude: list[str] = None, env_name: str = "point_mass_easy"):
    """
    Plot the performance of the agents in the given configuration.
    :param exclude: The agents to exclude.
    :param env_name: The environment name.
    """
    smoothing_factor = 100
    config_dir = f"trained_agents/{env_name}"
    # config_dir = f"debug_agents/{env_name}"
    agents_found = os.listdir(config_dir)
    agent_names = [agent for agent in agents_found if agent not in exclude] if exclude else agents_found
    agents = [agent for agent in agent_names if agent != ".DS_Store"]
    agent_dirs = [os.path.join(config_dir, agent) for agent in agents]
    for metric in metrics:
        for index, agent_dir in enumerate(agent_dirs):
            event_files = get_event_files_from_dir(agent_dir)
            mean, std, all_steps, top_scores, all_scores = get_mean_and_std_performance_from_event_files(event_files, [metric])
            # apply smoothing to mean
            # mean = [sum(mean[max(i-smoothing_factor, 0):i])/smoothing_factor if i > smoothing_factor else mean[i] for i in range(len(mean))]
            steps = max(all_steps, key=len)
            plt.plot(steps, mean, label=agent_labels.get(agent_names[index], agent_names[index]))
            plt.fill_between(steps, mean - confidence_interval * std,
                             mean + confidence_interval * std, alpha=0.2)
        plt.legend()
        plt.grid()
        plt.title(f"{env_labels[env_name]}: {metric_labels[metric]}")
        # plt.savefig(f"{env_labels[env_name]}: {metric_labels[metric]}" + ".png")
        plt.show()

plot_performance(env_name ="point_mass_easy", exclude=exclude)
plot_performance(env_name ="point_mass_hard", exclude=exclude)
plot_performance(env_name ="HalfCheetahVel_easy", exclude=exclude)
plot_performance(env_name ="HalfCheetahVel_hard", exclude=exclude)



# plot_performance(env_name ="HalfCheetahVel_hard", exclude=["oracle", "naive", "belief", "varibad"])
# plot_performance(env_name ="HalfCheetahVel_hard", exclude=["oracle"])
# plot_performance(env_name ="point_mass_hard", exclude=["oracle", "naive", "belief", "varibad"])

# plot_performance(configuration="config_0")