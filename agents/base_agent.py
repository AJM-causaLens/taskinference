from argparse import Namespace
from torch.nn import functional as F
import torch.nn as nn
import torch
from models.decoder import RewardDecoder, RewardDecoderProbabilistic, TaskDecoder, TaskDecoderProbabilistic
from models.encoder import RNNEncoder
from models.varibad_buffer import Vae_Buffer


class BaseAgent(nn.Module):
    def __init__(self, config: Namespace):
        super(BaseAgent, self).__init__()
        self.args: Namespace = config
        self.latent_dim: int = config.latent_dim
        self.buffer: Vae_Buffer = Vae_Buffer(max_size=config.vae_buffer_size if hasattr(config, 'vae_buffer_size') else 150,)
        self.kl_weight: float = 0.1
        self.decode_reward: bool = config.decode_reward
        self.decode_task: bool = config.decode_task
        self.contrastive_task_loss: bool = config.contrastive_task_loss
        self.use_kl_loss: bool = config.use_kl_loss
        self.use_decoder: bool = config.use_decoder
        self.det_rew_decoder: bool = config.det_rew_decoder
        self.det_task_decoder: bool = config.det_decoder
        self.reward_decoder, self.task_decoder = self.initialise_decoder()
        self.encoder = self.initialise_encoder()
        self.optimiser = self.get_optimiser()
        self.truncate_size = config.truncate_size
        self.reward_weight, self.task_weight, self.contrastive_weight = 1, 1, 1
    def compute_vae_loss(self):
        """
        Compute the loss of the VAE
        """
        elbo_loss, reward_loss, task_loss, kl_loss, contrastive_loss = 0, 0, 0, 0, 0

        batch_indices = self.buffer.get_batches(batch_size=self.args.batch_size)

        for batch_index in batch_indices:
            task_recon_loss, reward_recon_loss, kl_divergence, contrastive = 0, 0, 0, 0
            batch, trajectory_lens = self.buffer.get_indexed_episode(batch_index)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']
            vae_tasks = batch['task']

            latent_samples, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.truncate_size,
                                                            )

            if self.args.decode_task:
                task_recon_loss = self.get_task_reconstruction_loss(latent_samples, vae_tasks)
                task_loss += task_recon_loss.item()

            if self.decode_reward:
                reward_recon_loss = self.get_reward_recon_loss(vae_prev_obs, vae_actions, vae_next_obs, latent_samples, vae_rewards)
                reward_loss += reward_recon_loss.item()
            if self.contrastive_task_loss:
                contrastive = self.get_contrastive_task_loss(latent_mean, vae_prev_obs, vae_actions, vae_rewards, vae_tasks)
                contrastive_loss += contrastive.item()

            if self.use_kl_loss:
                kl_divergence = self.get_kl_loss(latent_mean, latent_logvar)
                kl_loss += kl_divergence.item()

            loss = (self.reward_weight * reward_recon_loss +
                    self.task_weight * task_recon_loss +
                    self.kl_weight * kl_divergence +
                    self.contrastive_weight * contrastive)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        infos = {
            "task_loss": task_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss,
            "contrastive_loss": contrastive_loss,
        }
        return infos

    def get_kl_loss(self, latent_mean, latent_logvar) -> torch.Tensor:
        # Unpack dimensions: T = number of timesteps, B = batch size, D = latent dimension.
        T, B, D = latent_mean.shape

        # Compute KL divergence for t=0 with fixed prior N(0,I)
        mu_0 = latent_mean[0]  # shape: (B, D)
        logvar_0 = latent_logvar[0]  # shape: (B, D)
        kl_0 = 0.5 * torch.sum(torch.exp(logvar_0) + mu_0 ** 2 - 1 - logvar_0, dim=-1)  # shape: (B,)

        # Compute KL divergence for t=1..T-1 between consecutive timesteps.
        mu_t = latent_mean[1:]  # shape: (T-1, B, D)
        mu_tm1 = latent_mean[:-1]  # shape: (T-1, B, D)
        logvar_t = latent_logvar[1:]  # shape: (T-1, B, D)
        logvar_tm1 = latent_logvar[:-1]  # shape: (T-1, B, D)

        kl_consecutive = 0.5 * (
                torch.sum(logvar_tm1 - logvar_t, dim=-1) +
                torch.sum(torch.exp(logvar_t - logvar_tm1), dim=-1) -
                D +
                torch.sum((mu_tm1 - mu_t) ** 2 / torch.exp(logvar_tm1), dim=-1)
        )  # shape: (T-1, B)

        # Concatenate the KL for t=0 with the consecutive KL divergences
        kl_all = torch.cat([kl_0.unsqueeze(0), kl_consecutive], dim=0)  # shape: (T, B)

        kl_all= kl_all.sum(dim=(0, 1))
        return kl_all

    def get_contrastive_task_loss(
            self,
            latent_mean: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            task_labels: torch.Tensor,
            temperature: float = 0.1
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_task_reconstruction_loss(self, latent_samples, vae_tasks) -> torch.Tensor:
        task_preds = self.task_decoder(latent_samples[:-1, :, :])
        task_reconstruction_loss = self.task_decoder.get_loss(task_preds, vae_tasks)
        return task_reconstruction_loss

    def get_reward_recon_loss(self, vae_prev_obs, vae_actions, vae_next_obs, latent_samples, vae_rewards) -> torch.Tensor:
        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape))
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))

        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

        rew_pred = self.reward_decoder(dec_embedding, dec_next_obs, dec_prev_obs, dec_actions.float())
        if self.det_rew_decoder:
            rew_reconstruction_loss = torch.nn.MSELoss(reduction='sum')(rew_pred, dec_rewards)
        else:
            rew_reconstruction_loss = self.reward_decoder.get_loss(rew_pred, dec_rewards)

        return rew_reconstruction_loss


    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """
        reward_decoder, task_decoder = None, None
        latent_dim = self.args.latent_dim

        if self.args.decode_reward:
            if self.det_rew_decoder:
                reward_decoder = RewardDecoder(
                    args=self.args,
                    layers=self.args.reward_decoder_layers,
                    latent_dim=latent_dim,
                    state_dim=self.args.state_dim,
                    state_embed_dim=self.args.state_embedding_size,
                    action_dim=self.args.action_dim,
                    action_embed_dim=self.args.action_embedding_size,
                    num_states=0,
                    multi_head=self.args.multihead_for_reward,
                    pred_type=self.args.rew_pred_type,
                    input_prev_state=self.args.input_prev_state,
                    input_action=self.args.input_action,
                )
            else:
                reward_decoder = RewardDecoderProbabilistic(
                    args=self.args,
                    layers=self.args.reward_decoder_layers,
                    latent_dim=latent_dim,
                    state_dim=self.args.state_dim,
                    state_embed_dim=self.args.state_embedding_size,
                    action_dim=self.args.action_dim,
                    action_embed_dim=self.args.action_embedding_size,
                    input_prev_state=self.args.input_prev_state,
                    input_action=self.args.input_action,
                )

        if self.args.decode_task:
            if self.det_task_decoder:
                task_decoder = TaskDecoder(
                    latent_dim=latent_dim,
                    layers=self.args.task_decoder_layers,
                    task_dim=self.args.task_dim,
                    num_tasks=None,
                    pred_type=self.args.task_pred_type,
                    time_weighted_loss=self.args.time_weighted_loss if hasattr(self.args,
                                                                               'time_weighted_loss') else False,
                )
            else:
                task_decoder = TaskDecoderProbabilistic(
                    latent_dim=latent_dim,
                    layers=self.args.task_decoder_layers,
                    task_dim=self.args.task_dim,
                    num_tasks=None,
                    pred_type=self.args.task_pred_type,
                    time_weighted_loss=self.args.time_weighted_loss if hasattr(self.args,
                                                                               'time_weighted_loss') else False,
                )

        return reward_decoder, task_decoder

    def initialise_encoder(self):
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        )

        return encoder

    def get_optimiser(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def save_model(self, path: str):
        """
        Save the model's state dictionary to the specified path.

        Args:
            path (str): The file path to save the model.
        """
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'reward_decoder': self.reward_decoder.state_dict() if self.reward_decoder else None,
            'task_decoder': self.task_decoder.state_dict() if self.task_decoder else None,
            'optimizer': self.optimiser.state_dict()
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")


    def get_exploration_bonuses(self, episode_action, episode_next_obs, episode_reward):
        latent_samples, latent_mean, latent_logvar, _ = self.encoder(actions=episode_action,
                                                                     states=episode_next_obs,
                                                                     rewards=episode_reward,
                                                                     hidden_state=None,
                                                                     return_prior=True,
                                                                     detach_every=self.truncate_size,
                                                                     )
        k_terms = self.get_kl_sequence(latent_mean, latent_logvar)
        return k_terms

    @staticmethod
    def get_kl_sequence(latent_mean, latent_logvar):
        """
        Compute the KL divergence between the prior and the approximate posterior for each timestep in the sequence.
        :param latent_mean: The mean of the approximate posterior.
        :param latent_logvar: The log-variance of the approximate posterior.
        :return: The KL divergence for each timestep in the sequence.
        """
        T, B, D = latent_mean.shape

        # Compute KL divergence for t=0 with fixed prior N(0,I)
        mu_0 = latent_mean[0]  # shape: (B, D)
        logvar_0 = latent_logvar[0]  # shape: (B, D)
        kl_0 = 0.5 * torch.sum(torch.exp(logvar_0) + mu_0 ** 2 - 1 - logvar_0, dim=-1)  # shape: (B,)

        # Compute KL divergence for t=1..T-1 between consecutive timesteps.
        mu_t = latent_mean[1:]  # shape: (T-1, B, D)
        mu_tm1 = latent_mean[:-1]  # shape: (T-1, B, D)
        logvar_t = latent_logvar[1:]  # shape: (T-1, B, D)
        logvar_tm1 = latent_logvar[:-1]  # shape: (T-1, B, D)

        kl_consecutive = 0.5 * (
                torch.sum(logvar_tm1 - logvar_t, dim=-1) +
                torch.sum(torch.exp(logvar_t - logvar_tm1), dim=-1) -
                D +
                torch.sum((mu_tm1 - mu_t) ** 2 / torch.exp(logvar_tm1), dim=-1)
        )  # shape: (T-1, B)

        # Concatenate the KL for t=0 with the consecutive KL divergences
        kl_all = torch.cat([kl_0.unsqueeze(0), kl_consecutive], dim=0)  # shape: (T, B)
        return kl_all




