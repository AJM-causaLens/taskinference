import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import abc
import torch as th

class DummyEnv(gym.Env):  # Inherit from gym.Env
    def __init__(self, value=0.0):
        """
        Initialize the dummy environment.

        Args:
            value (float): The underlying parameter that determines the state.
        """
        super().__init__()
        self.value = value  # The parameter to vary
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # No real action, just a placeholder
        self.current_step = 0
        self.max_episode_length = 100
        self.episode_number = 0

    def reset(self, **kwargs):
        """
        Reset the environment to its initial state.
        """
        self.state = np.array([self.value, self.value], dtype=np.float32)
        self.current_step = 0
        self.episode_number +=1
        return self.state, {}

    def step(self, action):
        """
        Step through the environment. The state does not change.

        Args:
            action (int): Placeholder action (not used).
        """
        reward = 1.0  # No meaningful reward
        terminated = False
        truncated = False
        info = {}
        self.current_step += 1
        if self.current_step >= 100:
            terminated = True
            truncated = True

        return self.state, reward, terminated, truncated, info

    def set_value(self, new_value):
        """
        Change the underlying parameter `value`.

        Args:
            new_value (float): The new value to set.
        """
        self.value = new_value

class BAMDP_dummy(gym.Wrapper):
    """
    Wraps the environment to augment observations with VAE beliefs,

    handles both sample generation and VAE.
    """

    def __init__(self, env: DummyEnv, vae, args, tasks=None):
        super().__init__(env)
        self.args = args
        self.vae = vae
        self.tasks = tasks
        self.last_obs = None  # Attribute to expose the latest observation
        self.prior = th.zeros(2 * vae.latent_dim)
        self.prior[self.prior.numel() // 2:] = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(env.observation_space.shape[0] + 2 * vae.latent_dim,),
                                            dtype=np.float32)
        self.hidden_state = self.get_initial_hidden_state()
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

    def get_initial_env_params(self):
        return 0

    def set_parameters(self, task=None):
        """Sample and set new parameters."""
        task = np.random.randint(0, 20)
        self.env.set_value(task)
        self.current_env_params =task

    def get_initial_hidden_state(self):
        return th.zeros(self.args.num_gru_layers, self.vae.encoder.hidden_size)
        # return th.zeros(self.vae.encoder.hidden_size)

    def update_encoding(self, next_obs, action, reward):

        with th.no_grad():
            latent_sample, latent_mean, latent_logvar, hidden_state = self.vae.encoder(
                actions=th.tensor(action, dtype=th.float32),
                states=th.tensor(next_obs, dtype=th.float32),
                rewards=th.tensor([reward], dtype=th.float32),
                hidden_state=self.hidden_state,
                return_prior=False)

        self.hidden_state = hidden_state


        return np.concatenate([latent_mean.numpy(), latent_logvar.numpy()])

    def get_reward(self, reward, obs, info):
        return reward

    def step(self, action):
        """Take a step, compute belief, and add the transition to the VAE buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step(action)
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

                belief = self.update_encoding(next_obs, action, reward)
                obs, info = self.env.reset()
                augmented_obs = np.concatenate([obs, belief])
                self.last_obs = augmented_obs
                self.previous_obs = obs
                self.previous_action = action
                return augmented_obs, reward, terminated, truncated, info
        # print(f"Goal Location: {self.env._env.physics.named.data.geom_xpos['target', :2]}")
        # Update the last and previous observations
        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])
        self.last_obs = augmented_obs
        self.previous_obs = next_obs
        self.previous_action = action

        return augmented_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the TASK and compute the initial belief.
        This does not reset the underlying environment with the same task. This is handled in the step function.
        """
        self.previous_action = None
        self.previous_obs = None
        self.current_episode = 0
        self.tasks_completed += 1
        self.set_parameters()
        self.hidden_state = self.get_initial_hidden_state()

        obs, info = self.env.reset(**kwargs)

        augmented_obs = np.concatenate([obs, copy.copy(self.prior)])

        return augmented_obs, info