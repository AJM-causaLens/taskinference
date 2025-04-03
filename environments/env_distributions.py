import gymnasium as gym
from dmc_wrapper import DMCGym
import numpy as np
import torch

# env = DMCGym(domain=env_name[0], task=env_name[1])

def convert_to_tensor(array_list):
    tensor_list = [torch.from_numpy(array).float() for array in array_list]
    stacked_tensor = torch.stack(tensor_list)
    return stacked_tensor
class PointMassNoVAE(gym.Wrapper):
    """
    Wraps the Reacher environment to augment observations with VAE beliefs,
    """

    def __init__(self):
        env = DMCGym(domain="point_mass", task="easy")
        super().__init__(env)
        self.current_env_params = None


    def set_params(self, task):
        self.current_env_params = task

    @staticmethod
    def sample_task(num_tasks=1):
        point_mass_test_tasks = []

        for _ in range(num_tasks):
            while True:
                task = (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
                if np.linalg.norm(task) >= 0.1:  # Ensure distance is at least 0.1 from (0,0)
                    rounded_task = tuple(np.round(task, 3))
                    point_mass_test_tasks.append(rounded_task)
                    break  # Task is valid, exit loop

        return point_mass_test_tasks


    def get_reward(self, reward, obs, info):
        """
        Computes a reward inversely proportional to the distance from the goal,
        but only if the agent is within 0.1 units of the goal.
        """
        assert self.current_env_params is not None, "Environment parameters not set."
        position = obs[:2]  # Extract agent position
        distance = np.linalg.norm(position - self.current_env_params)  # Compute Euclidean distance


        if distance < 0.1:
            reward = 1 - (distance / 0.1)  # Inversely proportional to distance (closer = higher reward)
        else:
            reward = 0  # No reward outside 0.1 range

        return reward


    def simulate_episode(self, actions):
        # WHat stable baselines does under the hood
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        observations = []
        rewards = []
        obs, info = self.reset()
        observations.append(obs)
        for action in actions:
            # print(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward = self.get_reward(reward, next_obs, info)
            observations.append(next_obs)
            rewards.append(reward)

        prev_obs = observations[:-1]
        next_obs = observations[1:]
        rewards = torch.from_numpy(np.array(rewards)).float().reshape(-1, 1, 1)
        prev_obs = convert_to_tensor(prev_obs).unsqueeze(1)
        next_obs = convert_to_tensor(next_obs).unsqueeze(1)



        return prev_obs, next_obs, rewards

    def reset(self, **kwargs):
        """
        Reset the environment and ensure the agent always starts at (0, 0).
        """
        # Call the original reset method from BAMDP_deepmind
        _, info = super().reset(**kwargs)
        # obs, info = self.env.reset(**kwargs)

        # Now modify the agent's initial position
        physics = self.env._env.physics
        physics.named.data.geom_xpos['pointmass'][:2] = 0.0
        physics.named.data.qpos['root_x'] = 0.0
        physics.named.data.qpos['root_y'] = 0.0
        physics.named.data.qvel['root_x'] = 0.0
        physics.named.data.qvel['root_y'] = 0.0

        physics.forward()  # Ensure changes take effect
        obs = np.zeros(4)

        info['current_env_params'] = self.current_env_params

        return obs, info
