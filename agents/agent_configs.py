from argparse import Namespace
import numpy as np

base_config = {"contrastive_task_loss": False, "use_kl_loss": True, "multihead_for_reward": False, "rew_pred_type": "deterministic", "det_rew_decoder": True,
               "det_decoder": True, "input_prev_state": False, "input_action": True, 'encoder_layers_before_gru': [32, 32],
               'encoder_layers_after_gru': [32, 32], 'encoder_gru_hidden_size': 64, "reward_embedding_size": 4,
               "num_gru_layers": 2, "lr": 0.0003, "update_every_n": 2000, "batch_size": 64, "save_interval":250,
               "truncate_size": None, "norm_actions_post_sampling": True, "task_pred_type": "task_description",
               "is_oracle": False
               }


# AGENT CONFIGS
varibad_config = {"decode_reward": True, "decode_task": False, "use_decoder": True}
oracle_config = {"is_oracle": True, "decode_reward": False, "decode_task": True, "use_decoder": True}
belief_config = {"decode_reward": False, "decode_task": True, "use_decoder": True}
moss_config = {"decode_reward": True, "decode_task": False, "use_decoder": True, "contrastive_task_loss": True}
tibbe_config = {"decode_reward": True, "decode_task": False, "use_decoder": True, "contrastive_task_loss": True}



# ENVIRONMENTS
def generate_valid_point_mass_tasks_structured(num_tasks=10, radius=0.2, seed=None):
    """
    """
    if seed is not None:
        np.random.seed(seed)

    # Randomly select angles between 0 and 2π
    angles = np.random.uniform(0, 2 * np.pi, num_tasks)
    tasks = [(round(radius * np.cos(angle), 3), round(radius * np.sin(angle), 3))
             for angle in angles]
    return tasks


cheetah_train_all, cheetah_test_all = [(np.random.uniform(0.5, 3)) for _ in range(50)], [(np.random.uniform(0.5, 3)) for _ in range(5)]

swimmer_train_all, swimmer_test_all = [(np.random.uniform(-10, 10), np.random.uniform(0, 1)) for _ in range(50)], [(np.random.uniform(-10, 10), np.random.uniform(0, 1)) for _ in range(5)]


point_mass_easy = {"state_dim": 4, "action_dim": 2, "policy_kwargs":None,
                   "train_tasks": generate_valid_point_mass_tasks_structured(100, 0.1, 42),
                   "test_tasks": generate_valid_point_mass_tasks_structured(20, 0.1, 84),
                    "latent_dim": 2,
                   "reward_decoder_layers": [32, 16], "task_decoder_layers": [32, 16],
                   "state_embedding_size": 4,
                   "action_embedding_size": 2,
                   "max_rollouts_per_task": 1,
                   "max_episode_length": 250,
                   "env_name": "point_mass_easy",
                   "env_register": ("point_mass", "easy"), # used to create the env with DMC or gym
                    "env_difficulty": "easy",
                   "total_timesteps": 3000000,
                   "task_dim": 2,
                   }
point_mass_hard = {"state_dim": 4, "action_dim": 2, "policy_kwargs":None,
                   "train_tasks": generate_valid_point_mass_tasks_structured(100, 0.1, 42),
                   "test_tasks": generate_valid_point_mass_tasks_structured(20, 0.1, 84),
                    "latent_dim": 2,
                   "reward_decoder_layers": [32, 32], "task_decoder_layers": [32, 16],
                   "state_embedding_size": 4,
                   "action_embedding_size": 2,
                   "max_rollouts_per_task": 1,
                   "max_episode_length": 250,
                   "env_name": "point_mass_hard",
                   "env_register": ("point_mass", "easy"), # used to create the env with DMC or gym
                    "env_difficulty": "hard",
                   "total_timesteps": 3000000,
                   "task_dim": 2,
                   }

HalfCheetah_easy = {"state_dim": 17, "action_dim": 6, "policy_kwargs":None,
                "train_tasks": cheetah_train_all,
                "test_tasks": cheetah_test_all,
                "latent_dim": 2,
                "reward_decoder_layers": [32, 16], "task_decoder_layers": [32, 16],
                "state_embedding_size": 17,
                "action_embedding_size": 6,
                "max_rollouts_per_task": 1,
                "max_episode_length": 500,
                "env_name": "HalfCheetahVel_easy",
                "env_register": "HalfCheetah-v5", # used to create the env with DMC or gym
                "total_timesteps": 3000000,
                "task_dim": 1,
                "env_difficulty": "easy",
                "input_prev_action": True,
                "input_prev_state": True,
                "vae_buffer_size": 50
                }


HalfCheetah_hard = {"state_dim": 17, "action_dim": 6, "policy_kwargs":None,
                "train_tasks": cheetah_train_all,
                "test_tasks": cheetah_test_all,
                "latent_dim": 2,
                "reward_decoder_layers": [32, 16], "task_decoder_layers": [32, 16],
                "state_embedding_size": 17,
                "action_embedding_size": 6,
                "max_rollouts_per_task": 1,
                "max_episode_length": 500,
                "env_name": "HalfCheetahVel_hard",
                "env_register": "HalfCheetah-v5", # used to create the env with DMC or gym
                "total_timesteps": 3000000,
                "task_dim": 1,
                "env_difficulty": "hard",
                "input_prev_action": True,
                "input_prev_state": True,
                "vae_buffer_size": 50
                }

Swimmer_easy = {"state_dim": 8, "action_dim": 2, "policy_kwargs":None,
                "train_tasks": swimmer_train_all,
                "test_tasks": swimmer_test_all,
                "latent_dim": 4,
                "reward_decoder_layers": [32, 16], "task_decoder_layers": [32, 16],
                "state_embedding_size": 8,
                "action_embedding_size": 2,
                "max_rollouts_per_task": 1,
                "max_episode_length": 500,
                "env_name": "Swimmer_easy",
                "env_register": "Swimmer-v5", # used to create the env with DMC or gym
                "total_timesteps": 3000000,
                "task_dim": 2,
                "env_difficulty": "easy",
                "input_prev_action": True,
                "input_prev_state": True,
                "vae_buffer_size": 50
                }


env_configs = {"point_mass_easy": point_mass_easy, "point_mass_hard": point_mass_hard, "HalfCheetah-v5_easy": HalfCheetah_easy, "HalfCheetah-v5_hard": HalfCheetah_hard,
               "Swimmer-v5_easy": Swimmer_easy}


configuration = {"varibad": varibad_config,
                 "varibad_local": varibad_config,
                 "belief": belief_config,
                 "naive": belief_config,
                 "tibbe": tibbe_config,
                 "ti": tibbe_config,
                 "ti_b": tibbe_config,
                 "tibbe_alpha": tibbe_config,
                 "tibbe_beta": tibbe_config,
                 "moss": moss_config,
                 "base": base_config,
                 "oracle": oracle_config}



def get_config(agent_name: str, env_name: str) -> Namespace:
    """
    Update the bse config with the values from the dictionary belong into the agent_name key.

    convert the combined dictionary to a argparse Namespace object.

    Return the namespace object.
    """
    updated_dict = {**base_config, **configuration[agent_name], **env_configs[env_name], "agent_name": agent_name}
    return Namespace(**updated_dict)


