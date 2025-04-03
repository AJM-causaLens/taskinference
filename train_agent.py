import gc
from custom_PPO import PPOWithInfo
from agent_callbacks import create_agent_callbacks
from agent_learn.environment_builder import build_environment, build_agent
from agents.agent_configs import get_config
debug_mode = True

num_runs = 1
verbose = 0
# env = ("point_mass", "easy")
# env = ("point_mass", "hard")
# env = ("HalfCheetah-v5", "easy")
# env = ("HalfCheetah-v5", "hard")
# env = ("HalfCheetah-v5", "hard")

debug_mode = False

def train_agent(env_name: str|tuple[str, str], agent_name: str, num_experiment:int = 1, BBE=False):
    for experiment in range(num_experiment):
        main_log = "trained_agents" if not debug_mode else "debug_agents"
        if BBE:
            main_log = f"{main_log}_BBE"
        config = get_config(agent_name, env_name if isinstance(env_name, str) else f"{env_name[0]}_{env_name[1]}")
        config.BBE = BBE
        agent = build_agent(agent_name, config)
        environment = build_environment(config.env_register, agent, config, config.train_tasks)()
        test_environment = build_environment(config.env_register, agent, config, config.test_tasks)()

        model = PPOWithInfo("MlpPolicy", environment, verbose=verbose,
                            tensorboard_log=f"{main_log}/{config.env_name}/{agent_name+"_BBE" if config.BBE else agent_name}", n_steps=config.update_every_n)

        print(f"tensorboard --logdir {model.tensorboard_log}")
        callbacks = create_agent_callbacks(agent, agent_name, config, test_environment)
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
        gc.collect()


explore_bonus = False

train_agent(("Swimmer-v5", "easy"), "ti", 1, BBE=explore_bonus)
