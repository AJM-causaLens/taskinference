from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
# from dummy_env import DummySamplingWrapper, DummyEnv, DummyBAMDP
from args_for_vae import get_args
import argparse
from dmc2gymnasium import DMCGym
from environment_wrappers import ParameterSamplingWrapper
from bamdp_wrapper import BAMDP, Oracle, BAMDP_vary
from bamdp_deepmind import BAMDP_reacher, BAMDP_pointmass
from callback_vae import VAECustomCallback
from callback_eval import RigourEvalCallback
from stable_baselines3.common.callbacks import CallbackList

from oracle_env_wrapper import OracleReacher, OraclePointMass
from varibad_vae import VaribadVAE

def get_env_properties(env_name):
    """
    Create a temporary environment to determine its observation space,
    action space dimensions, and maximum episode length.
    If max_episode_steps is unavailable, defaults to 1000 for MuJoCo environments.
    """
    max_length = 100
    if isinstance(env_name, tuple) or isinstance(env_name, list):
        temp_env = DMCGym(domain=env_name[0], task=env_name[1])
    else:
        temp_env = gym.make(env_name, render_mode=None)
    # Observation dimension
    obs_dim = temp_env.observation_space.shape[0]

    # Action dimension (handles discrete and continuous spaces)
    if isinstance(temp_env.action_space, gym.spaces.Discrete):
        action_dim = 1  # Discrete actions are represented as a single value
    else:
        action_dim = temp_env.action_space.shape[0]  # Continuous action space dimension

    # Maximum episode length with a fallback default


    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()
    args = get_args(rest_args)
    max_episode_length = getattr(temp_env.spec, "max_episode_steps", None)
    if max_episode_length is None:
        print(f"Warning: max_episode_steps not found for {env_name}. Defaulting to {args.max_episode_length}.")
    else:
        args.max_episode_length = max_episode_length
    parser.add_argument('--env-type', default=env_name)
    temp_env.close()

    if "point" in env_name[0]:
        args.decode_state = False
    args.action_dim = action_dim
    args.state_dim = obs_dim
    args.num_processes = 1
    args.env_name = env_name
    return obs_dim, action_dim, args

def get_env_and_logdir(config, main_log_dir):
    if isinstance(config.env_name, str):
        environs = make_vec_env(config.env_name, n_envs=config.num_envs)

        log_dir = '{}/{}/{}/{}'.format(main_log_dir, config.env_name, config.policy_type, config.ir_method)

    else:
        domain, task = config.env_name[0], config.env_name[1]

        def make_env():
            return DMCGym(domain=domain, task=task)

        environs = make_vec_env(make_env, n_envs=config.num_envs)
        log_dir = '{}/{}_{}/{}/{}'.format(main_log_dir, domain, task, config.policy_type, config.ir_method)

    return environs, log_dir

def make_env(env_name, vae, agent_name, args, tasks=None):
    """Factory function to create independent environments."""
    if agent_name in ["varibad", "varibad_explore"]:
        if isinstance(env_name, tuple):
            def _init():
                env = DMCGym(domain=env_name[0], task=env_name[1])
                env = BAMDP_reacher(env, vae, args, tasks)
                return env
        else:
            def _init():
                env = gym.make(env_name, render_mode=None)
                env = BAMDP(env, vae, args)
                return env
    elif agent_name == "baseline":
        def _init():
            env = gym.make(env_name, render_mode=None)
            env = ParameterSamplingWrapper(env)
            return env
    elif agent_name == "oracle":
        def _init():
            env = gym.make(env_name, render_mode=None)
            env = Oracle(env, num_episodes=args.max_rollouts_per_task)
            return env
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

    return _init


def make_env_deepmind(env_name, vae, agent_name, args, tasks=None):
    def _init():
        env = DMCGym(domain=env_name[0], task=env_name[1])
        if "reacher" in env_name[0]:
            env = BAMDP_reacher(env, vae, args, tasks)
        elif "point" in env_name[0]:
            env = BAMDP_pointmass(env, vae, args, tasks)
        else:
            raise ValueError(f"Unknown environment: {env_name[0]}")
        return env
    return _init

def make_env_oracle(env_name, args, tasks=None):
    def _init():
        env = DMCGym(domain=env_name[0], task=env_name[1])
        if "reacher" in env_name[0]:
            env = OracleReacher(env, args, tasks)
        elif "point" in env_name[0]:
            env = OraclePointMass(env, args, tasks)
        else:
            raise ValueError(f"Unknown environment: {env_name[0]}")
        return env
    return _init



def make_env_vary(env_name, vae, agent_name, args, tasks=None):
    """Factory function to create independent environments."""
    if agent_name in ["varibad", "varibad_explore"]:
        def _init():
            env = gym.make(env_name, render_mode=None)
            env = BAMDP_vary(env, vae, args, tasks)
            return env
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

    return _init

def make_dummy_env(vae, bayes, num_ep):
    """Factory function to create independent environments."""
    def _init():
        env = DummyEnv()  # Create environment once
        if bayes:
            return DummyBAMDP(env, vae, num_ep)  # Pass shared VAE
        else:
            return DummySamplingWrapper(env, num_ep)  # Or other wrapper
    return _init

def get_appropriate_callbacks(envs, vae, agent_name, args):
    eval_callback = RigourEvalCallback(eval_env=envs, envs_name="train_envs", verbose=0)
    if agent_name == "varibad":
        args.use_ir = False
        vae_callback = VAECustomCallback(vae, args)
        return CallbackList([eval_callback, vae_callback])
    elif agent_name == "varibad_explore":
        args.use_ir = True
        vae_callback = VAECustomCallback(vae, args)
        return CallbackList([eval_callback, vae_callback])

    return CallbackList([eval_callback])


def get_callback_vary(train_envs, test_envs, vae, agent_name, args):
    if train_envs is not None:
        train_callback = RigourEvalCallback(eval_env=train_envs, envs_name="train_envs", verbose=0)
    else:
        train_callback = None
    if test_envs is not None:
        test_callback = RigourEvalCallback(eval_env=test_envs, envs_name="test_envs", verbose=0)
    else:
        test_callback = None
    if agent_name == "varibad":
        args.use_ir = False
        vae_callback = VAECustomCallback(vae, args)
    elif agent_name == "varibad_explore":
        args.use_ir = True
        vae_callback = VAECustomCallback(vae, args)
    else:
        vae_callback = None
    list_of_callbacks = [item for item in [train_callback, test_callback, vae_callback] if item is not None]

    return CallbackList(list_of_callbacks)

def get_agent_name(bayes, ir):
    if not bayes:
        agent_name = "baseline"
    elif ir:
        agent_name = "varibad_explore"
    else:
        agent_name = "varibad"
    return agent_name


def get_dummy_env_properties():
    """
    Create a temporary environment to determine its observation space,
    action space dimensions, and maximum episode length.
    If max_episode_steps is unavailable, defaults to 1000 for MuJoCo environments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='pointrobot_varibad')
    args, rest_args = parser.parse_known_args()

    dum_env = DummyEnv()
    max_steps = dum_env.max_trajectory_length

    args = get_args(rest_args)
    args.action_dim = 2
    args.state_dim = 2
    args.num_processes = 1
    args.max_trajectory_len = max_steps
    return args

def get_dummy_callbacks(envs, vae, bayes_adaptive, use_ir):
    vae_callback = VAECustomCallback(vae, use_ir=use_ir)
    return CallbackList([vae_callback])

def make_vae(input_dim, action_dim, args):
    """Factory function to create a VAE."""

    state_embed_size = input_dim//2

    action_embed_size = max(8, action_dim)

    reward_embed_size = 5

    curr_input_dim = action_embed_size + state_embed_size + reward_embed_size

    layers_before_gru = [min(32, curr_input_dim), min(32, curr_input_dim)]

    layers_after_gru = [min(32, args.latent_dim), min(32, args.latent_dim)]

    args.action_embedding_size = action_embed_size
    args.state_embedding_size = state_embed_size
    args.reward_embedding_size = reward_embed_size
    args.encoder_layers_before_gru = layers_before_gru
    args.encoder_layers_after_gru = layers_after_gru

    vae = VaribadVAE(args)

    return vae


