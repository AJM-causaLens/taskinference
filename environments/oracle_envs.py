import gymnasium as gym
import numpy as np
import copy
import abc
from gymnasium import spaces


class OracleDistribution(gym.Wrapper):
    """
    Wraps the environment to augment observations with VAE beliefs,

    handles both sample generation and VAE.
    """

    def __init__(self, env, args, tasks):
        super().__init__(env)
        self.args = args

        if isinstance(tasks[0], (int, float)):
            self.task_dim = 1
            tasks = [[task] for task in tasks]

        else:
            self.task_dim = len(tasks[0])
        self.tasks = tasks
        self.observation_space = self.create_new_observation_space()
        self.last_obs = None  # Attribute to expose the latest observation
        self.env_name = args.env_name
        assert args.max_rollouts_per_task > 0, "Number of episodes must be greater than 0."
        self.max_episodes = args.max_rollouts_per_task
        self.max_episode_length = args.max_episode_length
        self.friction_factors = None
        self.current_episode = 0
        self.tasks_completed = 0
        self.current_timestep = 0
        self.initial_env_params = self.get_initial_env_params()
        self.current_env_params = None


    def create_new_observation_space(self):
        original_low = self.env.observation_space.low
        original_high = self.env.observation_space.high

        extra_inf = np.full((self.task_dim,), np.inf)
        extra_minus = np.full((self.task_dim,), -np.inf)

        new_low = np.concatenate((original_low, extra_minus))
        new_high = np.concatenate((original_high, extra_inf))

        return spaces.Box(low=new_low, high=new_high, shape=(self.env.observation_space.shape[0] + self.task_dim,), dtype=self.env.observation_space.dtype)
    @abc.abstractmethod
    def get_initial_env_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(self, task):
        """Sample and set new parameters."""
        raise NotImplementedError


    def get_reward(self, reward, obs, info):
        return reward


    def step(self, action):
        """Take a step, compute belief, and add the transition to the VAE buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.get_reward(reward, next_obs, info)
        next_obs = np.concatenate((next_obs, self.current_env_params))
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True

        self.previous_obs = next_obs
        self.previous_action = action
        info['current_env_params'] = self.current_env_params
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the TASK and compute the initial belief.
        This does not reset the underlying environment with the same task. This is handled in the step function.
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_action = None
        self.previous_obs = None
        self.current_episode = 0
        self.tasks_completed += 1
        task = self.tasks[self.tasks_completed % len(self.tasks)]
        self.set_parameters(task)
        obs = np.concatenate((obs, self.current_env_params))

        return obs, info

class CheetahOracle0(OracleDistribution):
    def get_initial_env_params(self):
        return None

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):

        tip_velocity = obs[8]

        # get different between the tip velocity and the goal velocity
        reward = 1 if np.abs(tip_velocity - self.current_env_params) < 0.01 else 0
        return reward

class CheetahOracle1(OracleDistribution):
    def get_initial_env_params(self):
        return None

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):

        tip_velocity = obs[8]

        # get different between the tip velocity and the goal velocity
        reward = 1 if np.abs(tip_velocity - self.current_env_params) < 0.05 else 0
        return reward

class CheetahOracle2(OracleDistribution):
    def get_initial_env_params(self):
        return None

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):

        tip_velocity = obs[8]

        # get different between the tip velocity and the goal velocity
        reward = 1 if np.abs(tip_velocity - self.current_env_params) < 0.1 else 0
        return reward


class PointmassOracle(OracleDistribution):
    """
    Wraps the Reacher environment to augment observations with VAE beliefs,
    """

    def __init__(self, env, args, tasks):
        super().__init__(env, args, tasks)

    def get_initial_env_params(self):
        return copy.deepcopy(self.env._env.physics.named.model.geom_friction[:])

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):
        """
        Computes a reward inversely proportional to the distance from the goal,
        but only if the agent is within 0.1 units of the goal.
        """
        position = obs[:2]  # Extract agent position
        distance = np.linalg.norm(position - self.current_env_params)  # Compute Euclidean distance


        if distance < 0.1:
            reward = 1 - (distance / 0.1)  # Inversely proportional to distance (closer = higher reward)
        else:
            reward = 0  # No reward outside 0.1 range

        return reward


    def reset(self, **kwargs):
        """
        Reset the environment and ensure the agent always starts at (0, 0).
        """
        # Call the original reset method from BAMDP_deepmind
        _, info = super().reset(**kwargs)

        # Now modify the agent's initial position
        physics = self.env._env.physics
        physics.named.data.geom_xpos['pointmass'][:2] = 0.0
        physics.named.data.qpos['root_x'] = 0.0
        physics.named.data.qpos['root_y'] = 0.0
        physics.named.data.qvel['root_x'] = 0.0
        physics.named.data.qvel['root_y'] = 0.0

        physics.forward()  # Ensure changes take effect
        obs = np.zeros(4)

        obs = np.concatenate((obs, self.current_env_params))

        info['current_env_params'] = self.current_env_params

        return obs, info



class ReacherOracle(OracleDistribution):
    """
    Wraps the Reacher environment from the DeepMind Control Suite to augment
    observations with VAE beliefs while ensuring only the reward function varies across tasks.
    """

    def __init__(self, env, args, tasks=None):
        super().__init__(env, args, tasks)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def get_initial_env_params(self):
        """
        Returns the initial environment parameters. The transition dynamics remain constant.
        """
        return copy.deepcopy(self.env._env.physics.named.model.geom_friction[:])

    def set_parameters(self, task=None):
        """
        Randomize the goal location for the Reacher environment while keeping transition dynamics fixed.
        This function is called inside BAMDP_deepmind.reset().
        """
        if self.tasks is not None and task is None:
            task = self.tasks[np.random.randint(0, len(self.tasks))]  # Sample a new goal from predefined task set
        else:
            task = np.random.uniform(low=-0.2, high=0.2, size=(2,))  # Default: Sample goal randomly within reach

        self.current_env_params = task  # The goal position (x, y)

    def get_reward(self, reward, obs, info):
        """
        Computes a sparse reward based on the distance between the reacher's end effector and the goal.
        """
        end_effector_pos = obs[:2]  # Extract end effector position
        distance = np.linalg.norm(end_effector_pos - self.current_env_params)  # Compute Euclidean distance

        if distance < 0.05:  # Reward given if within threshold
            reward = 1 - (distance / 0.05)
        else:
            reward = 0  # No reward if outside threshold

        return reward

    def get_observation(self, obs):
        # remove the middle two from the observation
        return np.concatenate((obs[:2], obs[-2:]))

    def reset(self, **kwargs):
        """
        Reset the environment and set a new goal location.
        The agent always starts from the same neutral position.
        """
        obs, info = super().reset(**kwargs)  # Calls set_parameters() internally

        physics = self.env._env.physics
        physics.named.data.qvel['shoulder'] = 0.0  # Reset the end effector position
        physics.named.data.qvel['wrist'] = 0.0  # Reset the end effector position
        physics.forward()  # Ensure changes take effect

        info['current_env_params'] = self.current_env_params

        # obs = self.get_observation(obs)

        return obs, info


    def step(self, action):
        """Take a step, compute belief, and add the transition to the VAE buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # next_obs = self.get_observation(next_obs)
        reward = self.get_reward(reward, next_obs, info)
        if self.current_timestep >= self.max_episode_length-1:
            terminated = True
            truncated = True
        else:
            self.current_timestep += 1

        # reward_from_info = self.reward_from_info(info)
        if terminated:
            self.current_timestep=0
            self.current_episode += 1
            if self.current_episode == self.max_episodes:
                terminated = True
                truncated = True
            else:
                terminated = False
                truncated = False

                self.previous_obs = next_obs
                self.previous_action = action
                info['current_env_params'] = self.current_env_params
                return next_obs, reward, terminated, truncated, info
        # print(f"augmented_obs: {augmented_obs.shape}")
        self.previous_obs = next_obs
        self.previous_action = action
        info['current_env_params'] = self.current_env_params
        return next_obs, reward, terminated, truncated, info
