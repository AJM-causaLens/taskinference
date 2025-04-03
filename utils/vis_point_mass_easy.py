import matplotlib.pyplot as plt
import numpy as np

def vis_point_mass(obs_list, rewards_list, task=None, title=None):
    """
    Visualize the agent's trajectory in the Point Mass environment when observations and rewards
    are stored as lists of NumPy arrays (one per episode). Also plots the goal location as an "X"
    with a shaded reward region around it and a color bar legend for reward values.

    Args:
        obs_list (list of np.ndarray): Each element is an array of shape (T_i, 4) containing observations.
        rewards_list (list of np.ndarray): Each element is an array of shape (T_i, ) containing rewards.
        task (tuple): A tuple (x, y) representing the goal location.
        title (str): The title of the plot.
    """

    # Flatten all observations and rewards into single arrays
    x, y, rewards = [], [], []

    for obs, reward in zip(obs_list, rewards_list):
        obs = np.asarray(obs)  # Ensure it's a NumPy array

        # Check for empty observations
        if obs.ndim == 1:
            print(f"Warning: Received 1D obs, reshaping to (1,4). Original shape: {obs.shape}")
            obs = obs.reshape(1, -1)  # Convert to 2D

        if obs.shape[1] < 2:  # Safety check
            print(f"Error: Expected at least 2D observations, got shape {obs.shape}")
            continue  # Skip this iteration to avoid crashing

        x.extend(obs[:, 0])  # Extract x-coordinates
        y.extend(obs[:, 1])  # Extract y-coordinates
        rewards.extend(reward)  # Collect rewards

    x, y, rewards = np.array(x), np.array(y), np.array(rewards)

    # Define environment bounds
    x_min, x_max = -0.3, 0.3
    y_min, y_max = -0.3, 0.3

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the reward zone (0.1 radius) as a transparent circle
    if task is not None:
        reward_circle = plt.Circle(task, 0.1, color='red', alpha=0.3, label='Reward Zone')
        ax.add_patch(reward_circle)
        # Plot the task (goal location) as an "X"
        ax.scatter(*task, color='red', marker='x', s=100, linewidth=3, label='Goal')

    # Plot trajectory line to show movement order
    ax.plot(x, y, color="gray", alpha=0.5, linestyle="-", linewidth=1)

    # Scatter plot with reward-based coloring
    sc = ax.scatter(x, y, c=rewards, cmap='viridis', s=50, vmin=0, vmax=1)

    # Add a color bar to indicate reward values
    cbar = plt.colorbar(sc, ax=ax, label="Reward Value")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Label key reward levels

    # Set bounds and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_title('Agent Trajectory with Goal Location & Reward Zone') if title is None else ax.set_title(title)
    ax.grid(True)

    # Show the plot
    plt.show()

    return x, y, rewards




def vis_point_mass_multi(obs_list, rewards_list, tasks=None):
    """
    Visualize multiple agent trajectories in the Point Mass environment.

    This function takes lists of observations and rewards (one per episode) and optionally a
    list of tasks (goal locations). It plots all trajectories on a single plot. Each trajectory
    is shown as a gray line with a scatter plot where the color indicates reward values. If a
    task is provided (either as a single tuple or as a list of tuples, one per episode), the
    goal location is marked with an "X" and a transparent circle indicating the reward zone.

    Args:
        obs_list (list of np.ndarray): Each element is an array of shape (T_i, 4) containing observations.
        rewards_list (list of np.ndarray): Each element is an array of shape (T_i,) containing rewards.
        tasks (tuple or list of tuples, optional): If a tuple, it represents the goal location (x, y)
            common to all episodes. If a list, each element is a tuple (x, y) corresponding to each episode.
    """
    # Define environment bounds
    x_min, x_max = -0.3, 0.3
    y_min, y_max = -0.3, 0.3

    fig, ax = plt.subplots(figsize=(8, 6))

    # To eventually attach a colorbar, we store the last scatter instance
    last_scatter = None

    # Loop over each trajectory
    for i, (obs, rewards) in enumerate(zip(obs_list, rewards_list)):
        obs = np.asarray(obs)

        # Handle 1D observation arrays by reshaping
        if obs.ndim == 1:
            print(f"Warning: Received 1D observation, reshaping to (1,4). Original shape: {obs.shape}")
            obs = obs.reshape(1, -1)

        if obs.shape[1] < 2:
            print(
                f"Error: Expected observations with at least 2 columns, got shape {obs.shape}. Skipping trajectory {i}.")
            continue

        # Extract x and y coordinates
        x = obs[:, 0]
        y = obs[:, 1]
        rewards = np.asarray(rewards)

        # Determine the task (goal) for this trajectory.
        task = None
        if tasks is not None:
            task = tasks[i]

        if task is not None:
            # Plot the reward zone (a transparent circle)
            reward_circle = plt.Circle(task, 0.1, color='red', alpha=0.3)
            ax.add_patch(reward_circle)
            # Plot the task (goal location) as an "X"
            ax.scatter(*task, color='red', marker='x', s=100, linewidth=3)

        # Plot the trajectory line
        ax.plot(x, y, color="gray", alpha=0.5, linestyle="-", linewidth=1)

        # Scatter plot with reward-based coloring
        sc = ax.scatter(x, y, c=rewards, cmap='viridis', s=50, vmin=0, vmax=1)
        last_scatter = sc  # save scatter for the colorbar

    # Add a color bar to indicate reward values using the last scatter instance.
    if last_scatter is not None:
        cbar = plt.colorbar(last_scatter, ax=ax, label="Reward Value")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

    # Set bounds, labels, and title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_title('Agent Trajectories with Goal Locations & Reward Zones')
    ax.grid(True)

    plt.show()


