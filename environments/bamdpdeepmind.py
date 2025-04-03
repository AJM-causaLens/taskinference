import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import copy
import abc
import torch as th

class BamdpDeepmind(gym.Wrapper):
    """
    Wraps the environment to augment observations with VAE beliefs,

    handles both sample generation and VAE.
    """

    def __init__(self, env, vae, args, tasks):
        super().__init__(env)
        self.args = args
        self.vae = vae
        self.is_oracle = args.is_oracle
        self.condition_on_logvar = args.condition_on_logvar if hasattr(args, 'condition_on_logvar') else True
        tasks = self.get_task_dim(tasks)
        self.tasks = tasks
        self.last_obs = None  # Attribute to expose the latest observation
        self.prior, self.hidden_state = self.get_prior()
        self.observation_space = self.create_new_observation_space()
        assert args.max_rollouts_per_task > 0, "Number of episodes must be greater than 0."
        self.max_episodes = args.max_rollouts_per_task
        self.max_episode_length = args.max_episode_length
        self.current_episode = 0
        self.tasks_completed = 0
        self.current_timestep = 0
        self.initial_env_params = self.get_initial_env_params()
        self.current_env_params = None

    @torch.no_grad()
    def get_prior(self):
        latent_sample, latent_mean, latent_logvar, hidden_state = self.vae.encoder.prior(batch_size=1, sample=False)
        if self.condition_on_logvar:
            prior_belief = np.concatenate([latent_mean.squeeze().numpy(), latent_logvar.squeeze().numpy()])
        else:
            prior_belief = latent_mean.squeeze().numpy()
        prior_hidden_state = hidden_state

        return prior_belief, prior_hidden_state

    def get_task_dim(self, tasks):
        if isinstance(tasks[0], (int, float)):
            self.task_dim = 1
            tasks = [[task] for task in tasks]

        else:
            self.task_dim = len(tasks[0])

        return tasks


    @abc.abstractmethod
    def get_initial_env_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(self, task):
        """Sample and set new parameters."""
        raise NotImplementedError

    def get_initial_hidden_state(self):
        return th.zeros(self.args.num_gru_layers, self.vae.encoder.hidden_size)

    def create_new_observation_space(self):
        if self.is_oracle:
            belief_dim = self.task_dim
        else:
            belief_dim = len(self.prior)
        original_low = self.env.observation_space.low
        original_high = self.env.observation_space.high

        extra_inf = np.full((belief_dim ,), np.inf)
        extra_minus = np.full((belief_dim ,), -np.inf)

        new_low = np.concatenate((original_low, extra_minus))
        new_high = np.concatenate((original_high, extra_inf))

        return spaces.Box(low=new_low, high=new_high, shape=(self.env.observation_space.shape[0] + belief_dim,), dtype=self.env.observation_space.dtype)

    def update_encoding(self, next_obs, action, reward):
        if self.is_oracle:
            return self.current_env_params
        with th.no_grad():
            latent_sample, latent_mean, latent_logvar, hidden_state = self.vae.encoder(
                actions=self.unhobble(action),
                states=self.unhobble(next_obs),
                rewards=self.unhobble([reward]),
                hidden_state=self.hidden_state,
                return_prior=False)

        self.hidden_state = hidden_state
        if self.condition_on_logvar:
            return np.concatenate([latent_mean.squeeze().numpy(), latent_logvar.squeeze().numpy()])
        return latent_mean.squeeze().numpy()

    # @staticmethod
    # def unhobble(x) -> th.Tensor:
    #     # return th.tensor(x, dtype=th.float32)
    #     return th.tensor(x, dtype=th.float32).unsqueeze(0).unsqueeze(1)

    @staticmethod
    def unhobble(x) -> th.Tensor:
        x = np.array(x, dtype=np.float32)
        return th.from_numpy(x).unsqueeze(0).unsqueeze(1)

    def get_reward(self, reward, obs, info):
        return reward

    def step(self, action):
        # Step the underlying environment.
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute the adjusted reward.
        reward = self.get_reward(reward, next_obs, info)

        # Check if we've reached the maximum timestep for the episode.
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True
        # Optional: print a warning if the action is out of bounds.
        if np.max(action) > 1 or np.min(action) < -1:
            print("Action out of bounds:", action)

        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])

        # Update internal state tracking.
        # self.last_obs = augmented_obs
        self.previous_obs = next_obs
        self.previous_action = action
        info['current_env_params'] = self.current_env_params

        return augmented_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the TASK and compute the initial belief.
        This does not reset the underlying environment with the same task. This is handled in the step function.
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_action = None
        self.previous_obs = None
        self.current_episode = 0
        self.tasks_completed += 1
        task = self.tasks[self.tasks_completed % len(self.tasks)]
        self.set_parameters(task)
        self.prior, self.hidden_state = self.get_prior()

        if self.is_oracle:
            belief = self.current_env_params
        else:
            belief = self.prior

        augmented_obs = np.concatenate([obs, copy.copy(belief)])

        return augmented_obs, info

    def load_vae(self, vae):
        self.vae = vae


class BamdpPointmass(BamdpDeepmind):
    """
    Wraps the Reacher environment to augment observations with VAE beliefs,
    """

    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)

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

        if self.is_oracle:
            belief = self.current_env_params
        else:
            belief = self.prior
        augmented_obs = np.concatenate([obs, copy.copy(belief)])

        info['current_env_params'] = self.current_env_params

        return augmented_obs, info


class BamdpPointMassHard(BamdpPointmass):

    def get_reward(self, reward, obs, info):
        """
        Computes a reward inversely proportional to the distance from the goal,
        but only if the agent is within 0.1 units of the goal.
        """
        position = obs[:2]  # Extract agent position
        distance = np.linalg.norm(position - self.current_env_params)  # Compute Euclidean distance

        if distance < 0.05:
            reward = 1 - (distance / 0.05)
        else:
            reward = 0

        return reward


class BamdpCheetahRun(BamdpDeepmind):
    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)
        self.dt = self.env.unwrapped.dt

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

    def get_velocity_reward(self, velocity):
        forward_reward = -1.0 * abs(velocity - self.current_env_params)
        return forward_reward

    def step(self, action):
        # Step the underlying environment.
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = info["x_velocity"]
        ctrl_cost = info["reward_ctrl"]

        forward_reward = self.get_velocity_reward(velocity)

        # Compute the adjusted reward.
        reward = forward_reward + ctrl_cost

        # Check if we've reached the maximum timestep for the episode.
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True
        # Optional: print a warning if the action is out of bounds.
        if np.max(action) > 1 or np.min(action) < -1:
            print("Action out of bounds:", action)

        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])

        # Update internal state tracking.
        # self.last_obs = augmented_obs
        self.previous_obs = next_obs
        self.previous_action = action
        info['current_env_params'] = self.current_env_params

        return augmented_obs, reward, terminated, truncated, info


class BamdpCheetahRunHard(BamdpCheetahRun):
    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)

    def get_reward(self, reward, obs, info):
        tip_velocity = obs[8]

        # get different between the tip velocity and the goal velocity
        reward = 1 if np.abs(tip_velocity - self.current_env_params) < 0.1 else 0
        return reward

    def get_velocity_reward(self, velocity):
        if abs(velocity - self.current_env_params) < 0.2:
            return 1
        return 0

class BamdpSwimmer(BamdpDeepmind):
    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)
        self.dt = self.env.unwrapped.dt

    def get_initial_env_params(self):
        return None

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task



    def step(self, action):
        next_obs, _, terminated, truncated, info = self.env.step(action)
        unweighted_reward_ctrl = info['reward_ctrl'] / 0.0001
        reward = self.current_env_params[0] * info['reward_forward'] + self.current_env_params[1] * unweighted_reward_ctrl

        # Check if we've reached the maximum timestep for the episode.
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True
        # Optional: print a warning if the action is out of bounds.
        if np.max(action) > 1 or np.min(action) < -1:
            print("Action out of bounds:", action)

        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])

        # Update internal state tracking.
        # self.last_obs = augmented_obs
        self.previous_obs = next_obs
        self.previous_action = action
        info['current_env_params'] = self.current_env_params

        return augmented_obs, reward, terminated, truncated, info




