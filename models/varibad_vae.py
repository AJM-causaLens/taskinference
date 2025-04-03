from datetime import datetime
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import json

from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder
from models.varibad_buffer import Vae_Buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VaribadVAE(nn.Module):
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args):
        super(VaribadVAE, self).__init__()
        self.args = args
        self.num_traj = args.vae_batch_num_trajs
        self.task_dim = args.task_dim if hasattr(args, 'task_dim') else 0
        self.num_tasks = None
        self.num_vae_updates = args.num_vae_updates

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2
        self.latent_dim = latent_dim

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder, self.reward_decoder, self.task_decoder = self.initialise_decoder()

        self.vae_buffer = Vae_Buffer(max_size=args.size_vae_buffer, batch_size=args.vae_batch_num_trajs, decode_task=args.decode_task)
        self.optimiser_vae = self.get_optimizer()

        self.kl_weight = self.args.kl_weight

    def save_args_as_json_or_markdown(self, args, file_path_base):
        # Convert args to a dictionary if it's an object
        os.makedirs(os.path.dirname(file_path_base), exist_ok=True)
        args_dict = vars(args) if not isinstance(args, dict) else args
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"{file_path_base}/{timestamp}"
        # Save as JSON
        with open(file_path + ".json", "w") as json_file:
            json.dump(args_dict, json_file, indent=4)


    def get_optimizer(self):
        encoder_params = []
        decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
            if self.args.decode_task:
                decoder_params.extend(self.task_decoder.parameters())

        encoder_params.extend(self.encoder.parameters())
        return torch.optim.Adam([*encoder_params, *decoder_params], lr=self.args.lr_vae)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
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
        ).to(device)
        return encoder

    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """

        if self.args.disable_decoder:
            return None, None, None

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        # initialise state decoder for VAE
        if self.args.decode_state:
            state_decoder = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
            ).to(device)
        else:
            state_decoder = None

        # initialise reward decoder for VAE
        if self.args.decode_reward:
            reward_decoder = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                latent_dim=2,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                num_states=0,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
            ).to(device)
        else:
            reward_decoder = None

        # initialise task decoder for VAE
        if self.args.decode_task:
            assert self.task_dim != 0
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.task_pred_type,
            ).to(device)
        else:
            task_decoder = None

        return state_decoder, reward_decoder, task_decoder

    def compute_state_reconstruction_loss(self, latent, prev_obs, next_obs, action, return_predictions=False):
        """ Compute state reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        state_pred = self.state_decoder(latent, prev_obs, action)

        if self.args.state_pred_type == 'deterministic':
            loss_state = (state_pred - next_obs).pow(2).mean(dim=-1)
        elif self.args.state_pred_type == 'gaussian':  # TODO: untested!
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = torch.distributions.normal.Normal(state_pred_mean, state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state, state_pred
        else:
            return loss_state

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward, return_predictions=False):
        """ Compute reward reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """
        rew_pred = self.reward_decoder(latent, next_obs, prev_obs, action.float())
        dumb_rew_pred = reward.mean().expand_as(reward)
        if self.args.rew_pred_type == 'bernoulli':  # TODO: untested!
            rew_pred = torch.sigmoid(rew_pred)
            rew_target = (reward == 1).float()  # TODO: necessary?
            loss_rew = F.binary_cross_entropy(rew_pred, rew_target, reduction='none').mean(dim=-1)
            dumb_loss_rew = None
        elif self.args.rew_pred_type == 'deterministic':
            loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            dumb_loss_rew = (dumb_rew_pred - reward).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_rew, dumb_loss_rew, rew_pred
        else:
            return loss_rew, dumb_loss_rew

    def compute_kl_loss(self, latent_mean, latent_logvar):

        kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))

        kl_divergence_sequental = self.compute_kl_divergence_consecutive(latent_mean, latent_logvar)

        return kl_divergences, kl_divergence_sequental


    def compute_kl_divergence_consecutive(self,
                                          latent_mean: torch.Tensor,
                                          latent_logvar: torch.Tensor
                                          ) -> torch.Tensor:
        """
        Compute the KL divergence between a fixed Gaussian prior and the latent
        distribution at t=0, and then between consecutive latent distributions
        for t=1..T-1.

        For t=0:
            .. math::
                \text{KL}\Big(N(\mu_0, \sigma_0^2) \parallel N(0,I)\Big)
                = \frac{1}{2}\sum_{d=1}^{D}\left(\exp(\text{logvar}_{0,d})
                + \mu_{0,d}^2 - 1 - \text{logvar}_{0,d}\right)

        For t=1..T-1:
            .. math::
                \text{KL}\Big(N(\mu_t, \sigma_t^2) \parallel N(\mu_{t-1}, \sigma_{t-1}^2)\Big)
                = \frac{1}{2}\sum_{d=1}^{D}\Bigg[\text{logvar}_{t-1,d} - \text{logvar}_{t,d}
                + \exp(\text{logvar}_{t,d} - \text{logvar}_{t-1,d}) - 1
                + \frac{(\mu_{t-1,d}-\mu_{t,d})^2}{\exp(\text{logvar}_{t-1,d})}\Bigg]

        Args:
            latent_mean (torch.Tensor): Mean of latent Gaussians with shape (T, B, D).
            latent_logvar (torch.Tensor): Log-variance of latent Gaussians with shape (T, B, D).

        Returns:
            torch.Tensor: KL divergences for each timestep, shape (T, B).
        """
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

        return kl_all

    # def compute_kl_divergence_consecutive(self,
    #         latent_mean: torch.Tensor,
    #         latent_logvar: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute the KL divergence between consecutive diagonal Gaussians at each
    #     timestep. For t=1..T-1, compute:
    #
    #     KL(N(μₜ, Σₜ) || N(μₜ₋₁, Σₜ₋₁))
    #
    #     where μₜ, logvarₜ ∈ ℝ^(B×D) (B=batch, D=latent dim).
    #     The return shape will be (T-1, B), since we compute a scalar per
    #     batch element for each consecutive pair of timesteps.
    #
    #     Args:
    #         latent_mean (torch.Tensor): Mean of latent Gaussians with shape (T, B, D).
    #         latent_logvar (torch.Tensor): Log-variance of latent Gaussians with shape (T, B, D).
    #
    #     Returns:
    #         torch.Tensor: KL divergence at each consecutive pair of timesteps,
    #                       shape (T-1, B).
    #     """
    #     # Sizes
    #     gauss_dim = latent_mean.shape[-1]
    #
    #     # Shift by one to compare consecutive steps
    #     mu_t = latent_mean[1:]  # (T-1, B, D)
    #     mu_tm1 = latent_mean[:-1]  # (T-1, B, D)
    #     logvar_t = latent_logvar[1:]  # (T-1, B, D)
    #     logvar_tm1 = latent_logvar[:-1]  # (T-1, B, D)
    #
    #     # KL(N(μ_t, Σ_t) || N(μ_{t-1}, Σ_{t-1}))
    #     # = 0.5 * [ sum(logvar_{tm1} - logvar_t)
    #     #           + sum(exp(logvar_t - logvar_tm1))
    #     #           - D
    #     #           + sum((mu_tm1 - mu_t)^2 / exp(logvar_tm1)) ]
    #     kl_elementwise = 0.5 * (
    #             torch.sum(logvar_tm1 - logvar_t, dim=-1)
    #             + torch.sum(torch.exp(logvar_t - logvar_tm1), dim=-1)
    #             - gauss_dim
    #             + torch.sum((mu_tm1 - mu_t) ** 2 / torch.exp(logvar_tm1), dim=-1)
    #     )
    #     # kl_elementwise shape: (T-1, B)
    #
    #     return kl_elementwise

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        # num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        # assert (num_unique_trajectory_lens == 1) or (self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
        assert not self.args.decode_only_past

        # take one sample for each ELBO term
        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        elbo_indices = None

        # expand the state/rew/action inputs to the decoder (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape))
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))
        dec_tasks = vae_tasks.unsqueeze(0).expand((num_elbos, *vae_tasks.shape)) if self.args.decode_task else None

        # expand the latent (to match the number of state/rew/action inputs to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

        if self.args.decode_task:
            task_reconstruction_loss = self.compute_task_recon_loss(latent_samples, vae_tasks)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                task_reconstruction_loss = task_reconstruction_loss.mean(dim=0)
            else:
                task_reconstruction_loss = task_reconstruction_loss.sum(dim=0)
            # sum the elbos, average across tasks
            task_reconstruction_loss = task_reconstruction_loss.sum(dim=0).mean()
        else:
            task_reconstruction_loss = 0

        if self.args.decode_reward:
            # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
            # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]
            rew_reconstruction_loss, dumb_rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs,
                                                                           dec_actions, dec_rewards)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
                dumb_rew_reconstruction_loss = dumb_rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
                dumb_rew_reconstruction_loss = dumb_rew_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
                dumb_rew_reconstruction_loss = dumb_rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
                dumb_rew_reconstruction_loss = dumb_rew_reconstruction_loss.sum(dim=0)
            # average across tasks
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
            dumb_rew_reconstruction_loss = dumb_rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0
            dumb_rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = self.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs,
                                                                               dec_next_obs, dec_actions)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
            # average across tasks
            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            kl_loss, kl_loss_sequential = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
                kl_loss_sequential = kl_loss_sequential.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
                kl_loss_sequential = kl_loss_sequential.sum(dim=0)
            # average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
            kl_loss_sequential = kl_loss_sequential.sum(dim=0).mean()
        else:
            kl_loss = 0
            kl_loss_sequential = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss, kl_loss_sequential, dumb_rew_reconstruction_loss

    def compute_task_recon_loss(self, latent, task):
        """ Compute task reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        task_pred = self.task_decoder(latent[:250, :, :])

        loss_task = (task_pred - task).pow(2).mean(dim=-1)

        return loss_task, task_pred

    def compute_vae_loss(self):
        """ Returns the VAE loss """
        epoch_elbo_loss = 0
        epoch_rew_reconstruction_loss = 0
        epoch_state_reconstruction_loss = 0
        epoch_dumb_rew_loss = 0
        epoch_dumb_state_loss = 0
        epoch_kl_loss = 0
        epoch_fixed_kl = 0
        epoch_sequence_kl = 0
        batch_indices = self.vae_buffer.get_batches(batch_size=self.num_traj)
        reward_predictions = []
        for batch_index in batch_indices:
            batch, trajectory_lens = self.vae_buffer.get_indexed_episode(batch_index)
            # batch_orig, trajectory_lens = self.vae_buffer.sample(self.num_traj)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']

            # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
            _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                            )
            losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                       vae_rewards, trajectory_lens)
            rew_reconstruction_loss, state_reconstruction_loss, kl_loss_fixed, kl_loss_sequential, dumb_rew_recon_loss = losses

            if self.args.kl_to_gauss_prior:
                kl_loss = kl_loss_fixed
            else:
                kl_loss = kl_loss_sequential

            # VAE loss = KL loss + reward reconstruction + state transition reconstruction
            # take average (this is the expectation over p(M))
            loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                    self.args.state_loss_coeff * state_reconstruction_loss +
                    self.kl_weight * kl_loss).mean()

            # make sure we can compute gradients
            if not self.args.disable_kl_term:
                assert kl_loss.requires_grad
            if self.args.decode_reward:
                assert rew_reconstruction_loss.requires_grad
            if self.args.decode_state:
                assert state_reconstruction_loss.requires_grad

            # overall loss
            elbo_loss = loss.mean()

            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                if self.args.decode_reward:
                    nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_state:
                    nn.utils.clip_grad_norm_(self.state_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_task:
                    nn.utils.clip_grad_norm_(self.task_decoder.parameters(), self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()
            epoch_elbo_loss += elbo_loss.item()
            epoch_kl_loss += kl_loss.item() * self.args.kl_weight

            if self.args.decode_reward:
                epoch_rew_reconstruction_loss += rew_reconstruction_loss.item() * self.args.rew_loss_coeff
                epoch_dumb_rew_loss += dumb_rew_recon_loss.item() * self.args.rew_loss_coeff
            if self.args.decode_state:
                epoch_state_reconstruction_loss += state_reconstruction_loss.item() * self.args.state_loss_coeff


        # infos = {'state_reconstruction_loss': epoch_state_reconstruction_loss/self.vae_buffer.num_in_buffer, 'reward_reconstruction_loss': epoch_rew_reconstruction_loss/self.vae_buffer.num_in_buffer, 'kl_loss': epoch_kl_loss/self.vae_buffer.num_in_buffer}
        infos = {'state_reconstruction_loss': epoch_state_reconstruction_loss/self.vae_buffer.num_in_buffer, 'reward_reconstruction_loss': epoch_rew_reconstruction_loss/self.vae_buffer.num_in_buffer, 'kl_loss': epoch_kl_loss/self.vae_buffer.num_in_buffer, 'dumb_rew_loss': epoch_dumb_rew_loss/self.vae_buffer.num_in_buffer}
        epoch_elbo_loss /= self.vae_buffer.num_in_buffer

        return epoch_elbo_loss, infos


    def save_model(self, path: str):
        """
        Save the model's state dictionary to the specified path.

        Args:
            path (str): The file path to save the model.
        """
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'state_decoder': self.state_decoder.state_dict() if self.state_decoder else None,
            'reward_decoder': self.reward_decoder.state_dict() if self.reward_decoder else None,
            'task_decoder': self.task_decoder.state_dict() if self.task_decoder else None,
            'optimizer': self.optimiser_vae.state_dict()
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load the model's state dictionary from the specified path.

        Args:
            path (str): The file path to load the model from.
        """
        # Load the saved state dictionary
        checkpoint = torch.load(path)

        # Restore the encoder
        self.encoder.load_state_dict(checkpoint['encoder'])

        # Restore the decoders if they exist
        if self.state_decoder and checkpoint['state_decoder']:
            self.state_decoder.load_state_dict(checkpoint['state_decoder'])
        if self.reward_decoder and checkpoint['reward_decoder']:
            self.reward_decoder.load_state_dict(checkpoint['reward_decoder'])
        if self.task_decoder and checkpoint['task_decoder']:
            self.task_decoder.load_state_dict(checkpoint['task_decoder'])

        # Restore the optimizer state
        self.optimiser_vae.load_state_dict(checkpoint['optimizer'])

        print(f"Model loaded from {path}")

