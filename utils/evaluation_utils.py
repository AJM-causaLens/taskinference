import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from utils.vis_point_mass_easy import vis_point_mass, vis_point_mass_multi

def plot_mean_and_logvar(belief_mean, belief_logvar, rewards, title):
    """
    Plot the mean and logvar of the belief.

    :param belief_mean: Mean of the belief (1D array)
    :param belief_logvar: Log-variance of the belief (1D array)
    :param rewards: Rewards
    :param title: Plot title
    """
    # Squeeze the tensors to remove dimensions of size 1
    belief_mean = belief_mean.squeeze()
    belief_logvar = belief_logvar.squeeze()
    rewards = rewards.squeeze()

    # convert logvar to var
    belief_var = np.exp(belief_logvar)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(belief_mean, label="Mean")
    ax[0].set_title("Belief Mean")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(belief_var, label="Variance")
    ax[1].set_title("Variance")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(rewards, label="Rewards")
    ax[2].set_title("Rewards")
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def plot_trajectory_with_visual_cues(x, y, rewards, task=None, x_min=None, x_max=None, y_min=None, y_max=None, title:str=None):
    """
    Improved visualization for trajectory data over time.

    - Uses a **color gradient** to indicate time evolution.
    - Highlights the **start and end points** clearly.
    - Overlays **goal and reward zones**.
    - Connects points with a **faded line** to show motion flow.

    :param x: X-coordinates of trajectory (1D array)
    :param y: Y-coordinates of trajectory (1D array)
    :param rewards: Reward values per coordinate (1D array)
    :param task: Goal/target location (tuple, optional)
    :param x_min, x_max, y_min, y_max: Plot axis limits (optional)
    :param title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize time steps for colormap (0 = start, 1 = end)
    time_steps = np.linspace(0, 1, len(x))

    # Scatter plot with time-based coloring (start → end gradient)
    sc = ax.scatter(x, y, c=time_steps, cmap='coolwarm', s=50, edgecolor="black", label="Trajectory")

    # Connect trajectory with a fading line
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=plt.cm.coolwarm(time_steps[i]), alpha=0.6, linewidth=2)

    # Mark the start & end points
    ax.scatter(x[0], y[0], color='green', marker='o', s=100, label='Start', edgecolor="black", zorder=3)
    ax.scatter(x[-1], y[-1], color='blue', marker='D', s=100, label='End', edgecolor="black", zorder=3)

    # If task (goal location) is provided, mark it
    if task is not None:
        reward_circle = plt.Circle(task, 0.1, color='red', alpha=0.3, label='Reward Zone')
        ax.add_patch(reward_circle)
        ax.scatter(*task, color='red', marker='x', s=120, linewidth=3, label='Goal')

    # Color bar for time-based transition
    cbar = plt.colorbar(sc, ax=ax, label="Time Progression")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Start", "Mid", "End"])

    # Set axis limits if provided

    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(-0.3, 0.3)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(-0.3, 0.3)

    # Labels and title
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Belief visualization for single trajectory") if title is None else ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    task: Any = None
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    if task is not None:
        all_episode_obs = []
        all_episode_rewards = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()

    if task is not None:
        assert len(env.envs) == 1, "Only one environment should be used for visualization"
        env.envs[0].current_env_params = task

    if task is not None:
        all_episode_obs.append(observations[:2])
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        if task is not None:
            all_episode_obs.append(observations[:2])
            all_episode_rewards.append(rewards)

    # if task is not None:
    #     vis_point_mass(all_episode_obs, all_episode_rewards, task)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward