from typing import Any, Dict, List, Optional, Union
import os
from datetime import datetime
import json
import torch as th
import gymnasium as gym
import numpy as np
import warnings
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EventCallback
from stable_baselines3.common.callbacks import BaseCallback
from utils.evaluation_utils import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


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




class RigourEvalCallback(EventCallback):

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        envs_name: str = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        name: str|None = None,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = False,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)
        self.name = name
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.envs_name = envs_name

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            if "test" in self.envs_name or "train" in self.envs_name:
                task_mean_rewards = []
                task_max_rewards = []
                task_episode_lengths = []
                assert hasattr(self.eval_env.envs[0], "tasks")
                assert hasattr(self.eval_env.envs[0], "set_parameters")
                assert len(self.eval_env.envs) == 1
                for index, task in enumerate(self.eval_env.envs[0].tasks):
                    task_rewards, task_lengths = evaluate_policy(
                        self.model,
                        self.eval_env,
                        n_eval_episodes=self.n_eval_episodes,
                        deterministic=self.deterministic,
                        return_episode_rewards=True,
                        warn=self.warn,
                        callback=self._log_success_callback,
                        task=task
                    )
                    mean_score = np.mean(task_rewards)
                    max_score = np.max(task_rewards)

                    task_mean_rewards.append(mean_score)
                    task_max_rewards.append(max_score)
                    task_episode_lengths.append(np.asarray(task_rewards).mean())
                    self.logger.record(f"{self.envs_name}/task_{index}/mean_reward", mean_score)
                    self.logger.record(f"{self.envs_name}/task_{index}/max_reward", max_score)
                self.logger.record(f"{self.envs_name}/mean_rewards", np.mean(task_mean_rewards))
                self.logger.record(f"{self.envs_name}/max_rewards", np.mean(task_max_rewards))


            mean_reward, std_reward = np.mean(task_mean_rewards), np.std(task_mean_rewards)

            # mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            # if self.verbose >= 1:
            #     print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            #     print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # # Add to current Logger
            # if self.envs_name is not None:
            #     self.logger.record(f"{self.envs_name}/mean_reward", float(mean_reward))
            #     self.logger.record(f"{self.envs_name}/max_reward", float(np.mean(task_max_rewards)))
            #     # self.logger.record(f"{self.envs_name}/mean_ep_length", mean_ep_length)
            #     # self.logger.record(f"{self.envs_name}/ep_rewards", episode_rewards)
            # else:
            #     if self.name is not None:
            #         self.logger.record(f"eval/mean_reward_{self.name}", float(mean_reward))
            #         self.logger.record(f"eval/max_reward_{self.name}", float(max(episode_rewards)))
            #
            #         self.logger.record(f"eval/mean_ep_length_{self.name}", mean_ep_length)
            #         self.logger.record(f"eval/ep_rewards_{self.name}", episode_rewards)
            #     else:
            #         self.logger.record("eval/mean_reward", float(mean_reward))
            #         self.logger.record("eval/max_reward", float(max(episode_rewards)))
            #         self.logger.record("eval/mean_ep_length", mean_ep_length)
            #         self.logger.record("eval/ep_rewards", episode_rewards)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

def create_agent_callbacks(agent, agent_name, config, test_environment):
    if agent_name == "naive":
        return RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)
    if agent_name == "oracle":
        return RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)

    callback = AgentCallback(agent, config)
    test_callback = RigourEvalCallback(eval_env=test_environment, envs_name="test_envs", verbose=0)
    callbacks = CallbackList([callback, test_callback])
    return callbacks



