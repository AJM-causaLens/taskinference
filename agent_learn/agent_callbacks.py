import os
from datetime import datetime
import json
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from call_backs.callback_eval import RigourEvalCallback


class AgentCallback(BaseCallback):
    def __init__(self, agent, args):
        super().__init__()
        self.args = args
        self.agent = agent
        self.batch_size = self.args.batch_size
        self.save_interval = self.args.save_interval
        self.tasks_added = []

        self.buffer = None
        self.belief_dim = 2 * self.agent.latent_dim
        self.num_updates = 0
        self.BBE = args.BBE
        self.total_updates = int(args.total_timesteps / args.update_every_n)
        self.start_at = self.args.start_at * self.total_updates if hasattr(self.args, "start_at") else 0# this is when to start adding IR and when to stop adding ir
        self.end_at = self.args.end_at * self.total_updates if hasattr(self.args, "end_at") else 0
        self.initial_intrinsic_weight  = self.args.intrinsic_weight if hasattr(agent, "intrinsic_weight") else 1.0
        self.intrinsic_weight = self.args.intrinsic_weight if hasattr(agent, "intrinsic_weight") else 1.0
        self.anneal_ir = self.args.anneal_ir if hasattr(self.args, "anneal_ir") else 0

        self.current_update = 0

    def save_args_as_json_or_markdown(self, args, file_path_base):
        os.makedirs(os.path.dirname(file_path_base), exist_ok=True)
        args_dict = vars(args) if not isinstance(args, dict) else args
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"{file_path_base}/{timestamp}"
        # Save as JSON
        with open(file_path + ".json", "w") as json_file:
            json.dump(args_dict, json_file, indent=4)

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer
        self.save_args_as_json_or_markdown(self.args, self.logger.dir)

    def _on_step(self):
        return True

    def _on_rollout_end(self) -> None:
        # Add experiences to buffer
        self.add_experiences_to_buffer()

        self.train_vae()


    def train_vae(self):
        if self.agent.buffer.num_in_buffer > 0:
            infos = self.agent.compute_vae_loss()
            self.num_updates += 1

            if self.num_updates % self.save_interval == 0:
                self.save_models()
            unique_tasks = set(str(x) for x in self.tasks_added)
            self.logger.record(f"VAE/tasks_added", len(unique_tasks))
            for key, value in infos.items():
                self.logger.record(f"VAE/{key}", value)


    def add_experiences_to_buffer(self):
        obs = th.as_tensor(self.buffer.observations, dtype=th.float32)[:, :, :-self.belief_dim]
        actions = th.as_tensor(self.buffer.actions, dtype=th.float32)
        rewards = th.as_tensor(self.buffer.rewards, dtype=th.float32).unsqueeze(-1)
        episode_starts = th.as_tensor(self.buffer.episode_starts, dtype=th.bool).squeeze(-1)
        dones = th.roll(episode_starts, shifts=-1, dims=0)
        dones[-1] = True
        next_obs = th.roll(obs, shifts=-1, dims=0)
        next_obs = th.where(dones.unsqueeze(-1).unsqueeze(-1), obs, next_obs)

        # Are we computing intrinsic rewards?
        add_ir = self.BBE and self.start_at < self.current_update < self.end_at

        episode_end_indices = th.nonzero(dones, as_tuple=False).squeeze(-1)
        exploration_bonuses = []
        start_idx = 0
        for end_idx in episode_end_indices:
            episode_obs = obs[start_idx:end_idx + 1]
            episode_action = actions[start_idx:end_idx + 1]
            episode_reward = rewards[start_idx:end_idx + 1]
            episode_next_obs = next_obs[start_idx:end_idx + 1]
            episode_infos = self.buffer.infos[start_idx:end_idx + 1]

            unique_items = set(str(x) for x in episode_infos)
            assert len(unique_items) == 1

            # assert len(set(episode_infos)) == 1, "All tasks in an episode should be the same"
            task = episode_infos[0]
            self.tasks_added.append(task)

            task = [task] if isinstance(task, float) else task


            self.agent.buffer.add({
                'obs': episode_obs,
                'actions': episode_action,
                'next_obs': episode_next_obs,
                'rewards': episode_reward,
                'task': th.tensor(task)
            })

            if add_ir:
                with th.no_grad():
                    exploration_bonus = self.agent.get_exploration_bonuses(episode_action, episode_next_obs, episode_reward)
                exploration_bonuses.append(exploration_bonus[:-1])

            start_idx = end_idx + 1

        if add_ir:
            exploration_bonuses = th.cat(exploration_bonuses, dim=0)
            exploration_bonuses = exploration_bonuses - th.min(exploration_bonuses)
            exploration_bonuses = exploration_bonuses / th.max(exploration_bonuses)
            current_ir_weight = self.get_current_ir_weight()
            self.buffer.advantages += current_ir_weight * exploration_bonuses.cpu().numpy()
            self.buffer.returns += current_ir_weight * exploration_bonuses.cpu().numpy()
            self.logger.record("callbacks/intrinsic_reward", th.mean(exploration_bonuses).item())
            self.logger.record("callbacks/std_intrinsic_reward", th.std(exploration_bonuses).item())
            self.logger.record("callbacks/intrinsic_reward_weight", self.intrinsic_weight)


        self.current_update += 1

    def save_models(self):
        # save model state_dict
        save_path = f"{self.logger.dir}/vae_{self.num_updates}.pt"
        self.agent.save_model(save_path)

    def get_current_ir_weight(self):
        assert self.start_at < self.current_update < self.end_at, "IR weight should only be updated during the BBE phase"
        if self.anneal_ir:
            self.intrinsic_weight = self.initial_intrinsic_weight - (self.current_update - self.start_at) / (self.end_at - self.start_at)

        return self.intrinsic_weight






def create_agent_callbacks(agent, agent_name, config, test_environment):
    if agent_name == "naive":
        return RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)
    if agent_name == "oracle":
        return RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)

    callback = AgentCallback(agent, config)
    test_callback = RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)
    callbacks = CallbackList([callback, test_callback])
    return callbacks



