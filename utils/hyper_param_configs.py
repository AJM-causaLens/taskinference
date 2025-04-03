config_A = {"decode_task": True, "decode_reward": False, "det_decoder": False, "kl_to_gauss_prior": False} # massively overfits
config_B = {"decode_task": True, "decode_reward": False, "det_decoder": False, "kl_to_gauss_prior": True}
config_C = {"decode_task": True, "decode_reward": False, "det_decoder": True, "kl_to_gauss_prior": False}
config_D = {"decode_task": True, "decode_reward": False, "det_decoder": True, "kl_to_gauss_prior": True}
config_E = {"decode_task": False, "decode_reward": True, "det_decoder": False, "kl_to_gauss_prior": False}
config_F = {"decode_task": False, "decode_reward": True, "det_decoder": False, "kl_to_gauss_prior": True}

config_Z = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False}
config_x = {"decode_task": True, "decode_reward": True, "det_decoder": False, "kl_to_gauss_prior": False, "task_loss_coeff": 5}
config_y = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "task_loss_coeff": 5}


config_p = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "time_weighted_loss": True}
config_s = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "time_weighted_loss": True, "latent_dim": 3}
config_detach_50 = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "time_weighted_loss": True, "latent_dim": 3, "tbptt_stepsize": 50}
config_q = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "task_loss_coeff": 5, "time_weighted_loss": True}


config_z_mini = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "latent_dim": 3}
config_one = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "latent_dim": 3, "kl_strategy": "anneal"}
config_two = {"decode_task": True, "decode_reward": True, "det_decoder": True, "kl_to_gauss_prior": False, "latent_dim": 5, "kl_strategy": "anneal"}

config_primis = {"decode_task": True, "decode_reward": True, "det_rew_decoder": False, "det_decoder": False, "kl_to_gauss_prior": False, "latent_dim": 5, "time_weighted_loss": True}
config_primis_one = {"decode_task": True, "decode_reward": True, "det_rew_decoder": False, "det_decoder": False, "kl_to_gauss_prior": False, "latent_dim": 5, "time_weighted_loss": True, "max_kl": 1}



config_dict = {
    "config_A": config_A,
    "config_B": config_B,
    "config_C": config_C,
    "config_D": config_D,
    "config_E": config_E,
    "config_F": config_F,
    "config_Z": config_Z,
    "config_x": config_x,
    "config_y": config_y,
    "config_p": config_p,
    "config_q": config_q,
    "config_s": config_s,
    "config_detach_50": config_detach_50,
    "config_z_mini": config_z_mini,
    "config_one": config_one,
    "config_two": config_two,
    "config_primis": config_primis,
    "config_primis_one": config_primis_one
}

# config_A|config_B|config_C|config_D
