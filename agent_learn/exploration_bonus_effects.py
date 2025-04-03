import matplotlib.pyplot as plt
from policy_optimization.suite_helpers import get_event_files_from_dir, get_mean_and_std_performance_from_event_files

metric_labels = {"rollout/ep_rew_mean": "Training Reward", "test_envs/mean_rewards": "Mean Test Reward", "test_envs/max_rewards": "Max Test Reward"}
env_labels = {"point_mass_easy": "Point Mass Easy", "point_mass_hard": "Point Mass Hard", "HalfCheetahVel_easy": "HalfCheetah Easy", "HalfCheetahVel_hard": "HalfCheetahVel (Sparse)",
              "Swimmer_easy": "Swimmer"}

# metrics = ["rollout/ep_rew_mean", "test_envs/mean_rewards"]
metrics = ["test_envs/mean_rewards"]
environments = ["HalfCheetahVel_easy", "HalfCheetahVel_hard", "point_mass_easy", "point_mass_hard", "Swimmer_easy"]

agent = ["belief", "belief_config_e"]


def eval_exploration_bonus(metrics: list[str], environments: list[str], base_agent: str, motivated_agent):
    names = [base_agent, f"{base_agent}_BBE"]
    for env in environments:
        base_agent_dir = f"trained_agents/{env}/{base_agent}"
        motivated_agent_dir = f"bbe_agents/{env}/{motivated_agent}"
        for metric in metrics:
            for index, agent_dir in enumerate([base_agent_dir, motivated_agent_dir]):
                event_files = get_event_files_from_dir(agent_dir)
                mean, std, all_steps, top_scores, all_scores = get_mean_and_std_performance_from_event_files(event_files, [metric])
                steps = max(all_steps, key=len)
                plt.plot(steps, mean, label=names[index])
                plt.fill_between(steps, mean - 0.2 * std, mean + 0.2 * std, alpha=0.2)
            # plt.legend()
            plt.grid()
            # plt.title(f"{env_labels[env]}: {metric_labels[metric]}")
            plt.savefig(f"bonus_effects/{base_agent}_{env_labels[env]}: {metric_labels[metric]}" + ".png", bbox_inches='tight')
            plt.close()
            # plt.show()

eval_exploration_bonus(metrics, environments, "ti", "ti_config_e")
# eval_exploration_bonus(metrics, environments, "belief", "belief_config_e")
# eval_exploration_bonus(metrics, environments, "moss", "moss_config_e")
# eval_exploration_bonus(metrics, environments, "varibad_local", "varibad_local_config_e")