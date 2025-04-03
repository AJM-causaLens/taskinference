import torch
from models.varibad_buffer import Vae_Buffer
from models.lstm_encoder import LSTMEncoder
from models.encoder import RNNEncoder
from models.transformer_encoder import TransformerEncoder
from models.transformer_encoder_causal import TransformerEncoderCausal
from models.decoder import TaskDecoder, TaskDecoderProbabilistic, RewardDecoder, RewardDecoderProbabilistic
from models.varibad_vae import VaribadVAE
from kl_strategies.strategies import AnnealKL, ConstantKL, CyclicalKL, BaseKLStrategy, QuasiCyclicalKL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TibbeVae(VaribadVAE):

    def __init__(self, args):
        self.det_decoder = args.det_decoder if hasattr(args, 'det_decoder') else False
        self.det_rew_decoder = args.det_rew_decoder if hasattr(args, 'det_rew_decoder') else True
        super().__init__(args)
        self.total_updates = args.total_updates if hasattr(args, "total_updates") else args.num_frames / args.update_every_n
        self.kl_strategy = self.initialize_kl_strategy(args)
        self.current_update = 0
        self.dumb_rew_loss = None
        self.truncate_size = self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None
        self.task_mean = None
        self.std = None
        # self.reward_loss_function = torch.nn.MSELoss()
        self.reward_loss_function = torch.nn.L1Loss()

    def load_vae_buffer(self, buffer_path: str):
        self.vae_buffer.load(buffer_path)

        episodes = self.vae_buffer.buffer

        task_tensors = torch.stack([d["task"] for d in episodes])  # Shape: (N, 2)

        mean = task_tensors.mean(dim=0)  # Shape: (2,)
        std = task_tensors.std(dim=0, unbiased=False)  # Shape: (2,)

        self.task_mean = list(mean.numpy())
        self.task_std = list(std.numpy())

        # print(f"must remember to add back in the normalization of rewards")
        for d in self.vae_buffer.buffer:
            d["task"] = (d["task"] - mean) / (std + 1e-8)  # Normalize with small epsilon to avoid division by zero


    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        if hasattr(self.args, 'encoder_model'):
            if self.args.encoder_model == 'lstm':
                encoder = LSTMEncoder(
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
            elif self.args.encoder_model == 'transformer':
                encoder = TransformerEncoder(
                    args=self.args,
                    latent_dim=self.args.latent_dim,
                    action_dim=self.args.action_dim,
                    action_embed_dim=self.args.action_embedding_size,
                    state_dim=self.args.state_dim,
                    state_embed_dim=self.args.state_embedding_size,
                    reward_size=1,
                    reward_embed_size=self.args.reward_embedding_size,
                ).to(device)
                return encoder
            elif self.args.encoder_model == 'transformer_causal':
                encoder = TransformerEncoderCausal(
                    args=self.args,
                    latent_dim=self.args.latent_dim,
                    action_dim=self.args.action_dim,
                    action_embed_dim=self.args.action_embedding_size,
                    state_dim=self.args.state_dim,
                    state_embed_dim=self.args.state_embedding_size,
                    reward_size=1,
                    reward_embed_size=self.args.reward_embedding_size,
                ).to(device)
                return encoder

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


    def initialize_kl_strategy(self, args) -> BaseKLStrategy:
        strategy_name = getattr(args, "kl_strategy", 'constant')
        max_kl = getattr(args, "max_kl", 0.1)
        if strategy_name == 'constant':
            return ConstantKL(initial_kl=max_kl, max_kl=max_kl, total_updates=self.total_updates)
        elif strategy_name == "anneal":
            return AnnealKL(initial_kl=0, max_kl=max_kl, total_updates=self.total_updates)
        elif strategy_name == "cycle":
            return CyclicalKL(initial_kl=0, max_kl=max_kl, total_updates=self.total_updates, cycles=2)
        elif strategy_name == "quasi_cycle":
            return QuasiCyclicalKL(initial_kl=0, max_kl=max_kl, total_updates=self.total_updates, cycles=2)
        else:
            raise NotImplementedError(f"Do not recognise KL strategy {strategy_name}")

    @torch.no_grad()
    def compute_validation_loss(self, validation_buffer: Vae_Buffer):
        epoch_elbo_loss = 0
        epoch_rew_reconstruction_loss = 0
        epoch_task_reconstruction_loss = 0
        epoch_kl_loss = 0
        epoch_fixed_kl = 0
        epoch_sequential_kl = 0
        batch_indices = validation_buffer.get_batches(batch_size=self.num_traj)
        for batch_index in batch_indices:
            batch, trajectory_lens = validation_buffer.get_indexed_episode(batch_index)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']
            vae_tasks = batch.get('task', None)

            # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
            _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.truncate_size,
                                                            )

            rew_reconstruction_loss, task_reconstruction_loss, kl_loss_fixed, kl_loss_sequential = self.compute_loss(
                latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                vae_rewards, vae_tasks)

            if self.args.kl_to_gauss_prior:
                kl_loss = kl_loss_fixed
            else:
                kl_loss = kl_loss_sequential

            # VAE loss = KL loss + reward reconstruction + state transition reconstruction
            # take average (this is the expectation over p(M))
            loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                    self.args.task_loss_coeff * task_reconstruction_loss +
                    self.kl_weight * kl_loss)

            # overall loss
            elbo_loss = loss.mean()
            epoch_elbo_loss += elbo_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_fixed_kl += kl_loss_fixed.item()
            epoch_sequential_kl += kl_loss_sequential.item()
            if self.args.decode_reward:
                epoch_rew_reconstruction_loss += rew_reconstruction_loss.item() * self.args.rew_loss_coeff
            if self.args.decode_task:
                epoch_task_reconstruction_loss += task_reconstruction_loss.item() * self.args.task_loss_coeff

        infos = {'kl_loss': (epoch_kl_loss * self.kl_weight)/ validation_buffer.num_in_buffer,
                 'current_kl_weight': self.kl_weight,
                 'kl_fixed': (epoch_fixed_kl / validation_buffer.num_in_buffer),
                 'kl_sequential': (epoch_sequential_kl / validation_buffer.num_in_buffer)}

        if self.args.decode_reward:
            infos['reward_reconstruction_loss'] = epoch_rew_reconstruction_loss / validation_buffer.num_in_buffer

        if self.args.decode_task:
            infos['task_reconstruction_loss'] = epoch_task_reconstruction_loss / validation_buffer.num_in_buffer



        epoch_elbo_loss /= validation_buffer.num_in_buffer

        return epoch_elbo_loss, infos


    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """
        state_decoder, reward_decoder, task_decoder = None, None, None
        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        # print("Must remove this, set latent dim to 2 in init decoder")
        # latent_dim = 2

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
            if self.det_decoder:
                task_decoder = TaskDecoder(
                    latent_dim=latent_dim,
                    layers=self.args.task_decoder_layers,
                    task_dim=self.task_dim,
                    num_tasks=self.num_tasks,
                    pred_type=self.args.task_pred_type,
                    time_weighted_loss=self.args.time_weighted_loss if hasattr(self.args,
                                                                               'time_weighted_loss') else False,
                )
            else:
                task_decoder = TaskDecoderProbabilistic(
                    latent_dim=latent_dim,
                    layers=self.args.task_decoder_layers,
                    task_dim=self.task_dim,
                    num_tasks=self.num_tasks,
                    pred_type=self.args.task_pred_type,
                    time_weighted_loss=self.args.time_weighted_loss if hasattr(self.args, 'time_weighted_loss') else False,
                )

        return state_decoder, reward_decoder, task_decoder

    def compute_vae_loss(self):
        """ Returns the VAE loss """
        epoch_elbo_loss = 0
        epoch_rew_reconstruction_loss = 0
        epoch_task_reconstruction_loss = 0
        epoch_kl_loss = 0
        epoch_fixed_kl = 0
        epoch_sequential_kl = 0
        total_norm = 0
        batch_indices = self.vae_buffer.get_batches(batch_size=self.num_traj)
        self.kl_weight = self.kl_strategy.get_kl_weight(current_update=self.current_update)
        for batch_index in batch_indices:
            batch, trajectory_lens = self.vae_buffer.get_indexed_episode(batch_index)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']
            vae_tasks = batch.get('task', None)

            # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
            _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.truncate_size,
                                                            )

            rew_reconstruction_loss, task_reconstruction_loss, kl_loss_fixed, kl_loss_sequential = self.compute_loss(latent_mean,
                                                                                                                     latent_logvar,
                                                                                                                     vae_prev_obs,
                                                                                                                     vae_next_obs,
                                                                                                                     vae_actions,
                                                                                                                     vae_rewards,
                                                                                                                     vae_tasks)

            if self.args.kl_to_gauss_prior:
                kl_loss = kl_loss_fixed
            else:
                kl_loss = kl_loss_sequential

            # VAE loss = KL loss + reward reconstruction + state transition reconstruction
            # take average (this is the expectation over p(M))
            loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                    self.args.task_loss_coeff * task_reconstruction_loss +
                    self.kl_weight * kl_loss)

            # make sure we can compute gradients
            if not self.args.disable_kl_term:
                assert kl_loss.requires_grad
            if self.args.decode_reward:
                assert rew_reconstruction_loss.requires_grad
            if self.args.decode_task:
                assert task_reconstruction_loss.requires_grad

            # overall loss
            elbo_loss = loss.mean()
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # update
            self.optimiser_vae.step()

            epoch_elbo_loss += elbo_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_fixed_kl += kl_loss_fixed.item()
            epoch_sequential_kl += kl_loss_sequential.item()
            if self.args.decode_reward:
                epoch_rew_reconstruction_loss += rew_reconstruction_loss.item() * self.args.rew_loss_coeff
            if self.args.decode_task:
                epoch_task_reconstruction_loss += task_reconstruction_loss.item() * self.args.task_loss_coeff

        infos = {'kl_loss': (epoch_kl_loss * self.kl_weight)/ self.vae_buffer.num_in_buffer,
                 'current_kl_weight': self.kl_weight,
                 'kl_fixed': (epoch_fixed_kl / self.vae_buffer.num_in_buffer),
                 'kl_sequential': (epoch_sequential_kl / self.vae_buffer.num_in_buffer)}
        if self.args.decode_reward:
            infos['reward_reconstruction_loss'] = epoch_rew_reconstruction_loss / self.vae_buffer.num_in_buffer
        if self.args.decode_task:
            infos['task_reconstruction_loss'] = epoch_task_reconstruction_loss / self.vae_buffer.num_in_buffer
        infos['gradient_norm'] = total_norm

        epoch_elbo_loss /= self.vae_buffer.num_in_buffer
        self.current_update += 1

        return epoch_elbo_loss, infos

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens=None):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """
        kl_loss_fixed, kl_loss_sequential, rew_reconstruction_loss, task_reconstruction_loss = (0,0,0,0)

        latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)

        if self.args.decode_task:
            task_preds = self.task_decoder(latent_samples[:-1, :, :])
            task_reconstruction_loss = self.task_decoder.get_loss(task_preds, vae_tasks)


        if self.args.decode_reward:
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
                # handle loss inside reward decoder if probabilistic
                rew_reconstruction_loss = self.reward_decoder.get_loss(rew_pred, dec_rewards)


        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            kl_loss_fixed, kl_loss_sequential = self.compute_kl_loss(latent_mean, latent_logvar)
            kl_loss_fixed = kl_loss_fixed.sum(dim=(0, 1))
            kl_loss_sequential = kl_loss_sequential.sum(dim=(0, 1))

        return rew_reconstruction_loss, task_reconstruction_loss, kl_loss_fixed, kl_loss_sequential

