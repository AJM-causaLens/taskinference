from agents.tibbe_alpha_agent import TibbeAlphaAgent
from agents.ti_agent import TiAgent
from agents.varibad_local_agent import VaribadLocalAgent
from agents.ti_b_agent import TibAgent
from agents.tibbe_beta_agent import TibbeBetaAgent
from dmc_wrapper import DMCGym
import gymnasium as gym
from environments.bamdpdeepmind import BamdpPointmass, BamdpPointMassHard, BamdpCheetahRun, BamdpCheetahRunHard, BamdpSwimmer



def build_environment(env_name, agent, config, tasks):
    def _init():
        if isinstance(env_name, tuple):
            env = DMCGym(domain=env_name[0], task=env_name[1])
            if "point" in env_name[0]:
                if config.env_difficulty == "easy":
                    env = BamdpPointmass(env, agent, config, tasks)
                elif config.env_difficulty == "hard":
                    env = BamdpPointMassHard(env, agent, config, tasks)
                else:
                    raise ValueError(f"Unknown environment: {env_name[1]}")
            else:
                raise ValueError(f"Unknown environment: {env_name[0]}")
            return env
        elif isinstance(env_name, str):
            env = gym.make(env_name)
            if env_name == "HalfCheetah-v5":
                if config.env_difficulty == "easy":
                    env = BamdpCheetahRun(env, agent, config, tasks)
                elif config.env_difficulty == "hard":
                    env = BamdpCheetahRunHard(env, agent, config, tasks)
                else:
                    raise ValueError(f"Unknown environment: {config.env_difficulty}")
            elif env_name == "Swimmer-v5":
                if config.env_difficulty == "easy":
                    env = BamdpSwimmer(env, agent, config, tasks)
                else:
                    raise ValueError(f"Unknown environment: {config.env_difficulty}")

            else:
                raise ValueError(f"Unknown environment: {env_name}")
            return env

    return _init


def build_agent(agent_name, config):
    if agent_name == "varibad":
        from agents.varibad_agent import VaribadAgent
        return VaribadAgent(config)
    elif agent_name == "varibad_local":
        return VaribadLocalAgent(config)
    elif agent_name == "belief":
        from agents.belief_agent import BeliefAgent
        return BeliefAgent(config)
    elif agent_name == "moss":
        from agents.moss_agent import MossAgent
        return MossAgent(config)
    elif agent_name == "naive":
        from agents.belief_agent import BeliefAgent
        return BeliefAgent(config)
    elif agent_name == "tibbe":
        from agents.tibbe_agent import TibbeAgent
        return TibbeAgent(config)
    elif agent_name == "oracle":
        from agents.belief_agent import BeliefAgent
        return BeliefAgent(config)
    elif agent_name == "tibbe_alpha":
        return TibbeAlphaAgent(config)
    elif agent_name == "tibbe_beta":
        return TibbeBetaAgent(config)
    elif agent_name == "ti":
        return TiAgent(config)
    elif agent_name == "ti_b":
        return TibAgent(config)

    else:
        raise ValueError(f"Unknown agent: {agent_name}")