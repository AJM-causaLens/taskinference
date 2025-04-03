import torch
from models.tibbe_vae import TibbeVae
from models.varibad_buffer import Vae_Buffer
from models.decoder import TaskDecoder, TaskDecoderProbabilistic
import torch.nn.functional as F



class TaskInferenceModel(TibbeVae):

    def __init__(self, args):
        args.decode_state = False
        args.decode_reward = False
        args.decode_task = True
        self.det_decoder = args.det_decoder
        super().__init__(args)
        self.task_statistics = None

        self.tasks_encountered = []
        self.num_samples = args.num_samples
        self.task_mean = None
        self.std = None


    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2


        if self.det_decoder:
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.task_pred_type,
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




        return None, None, task_decoder


    def get_task_statistics(self):
        episodes = self.vae_buffer.buffer

        # Stack all task tensors (Shape: (num_episodes, 2))
        task_tensors = torch.stack([episode['task'] for episode in episodes])

        # Compute mean and variance across episodes
        mean_task = task_tensors.mean(dim=0)  # Shape: (2,)
        var_task = task_tensors.var(dim=0, unbiased=False)  # Shape: (2,)

        # Convert variance to log variance for stability
        logvar_task = torch.log(var_task + 1e-6)  # Small epsilon to prevent log(0)

        # Store statistics
        self.task_statistics = {'mean': mean_task, 'logvar': logvar_task}

    def compute_vae_loss(self):
        epoch_elbo_loss = 0
        epoch_task_reconstruction_loss = 0
        epoch_kl_loss = 0
        epoch_fixed_kl = 0
        epoch_sequential_kl = 0
        epoch_naive_loss = 0
        if self.task_statistics is None:
            self.get_task_statistics()
        batch_indices = self.vae_buffer.get_batches(batch_size=self.num_traj)
        self.kl_weight = self.kl_strategy.get_kl_weight(current_update=self.current_update)
        for batch_index in batch_indices:
            batch, trajectory_lens = self.vae_buffer.get_indexed_episode(batch_index)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']
            vae_tasks = batch['task']

            # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
            _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.truncate_size,
                                                            )

            # take one sample for each ELBO term
            if not self.args.disable_stochasticity_in_latent:
                latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar, self.num_samples)
            else:
                latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)


            task_preds = self.task_decoder(latent_samples[:250, :, :])
            task_reconstruction_loss = self.task_decoder.get_loss(task_preds, vae_tasks)
            naive_preds = (self.task_statistics['mean'].view(1, 1, 2).expand(vae_tasks.size()), self.task_statistics['logvar'].view(1, 1, 2).expand(vae_tasks.size()))
            naive_loss = self.task_decoder.get_loss(naive_preds, vae_tasks)

            kl_loss_fixed, kl_loss_sequential = self.compute_kl_loss(latent_mean, latent_logvar)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss_fixed = kl_loss_fixed.mean(dim=0)
                kl_loss_sequential = kl_loss_sequential.mean(dim=0)
            else:
                kl_loss_fixed = kl_loss_fixed.sum(dim=0)
                kl_loss_sequential = kl_loss_sequential.sum(dim=0)
            # average across tasks
            kl_loss_fixed = kl_loss_fixed.sum(dim=0).mean()
            kl_loss_sequential = kl_loss_sequential.sum(dim=0).mean()

            if self.args.kl_to_gauss_prior:
                kl_loss = kl_loss_fixed
            else:
                kl_loss = kl_loss_sequential

            loss = (self.args.task_loss_coeff * task_reconstruction_loss +
                    self.kl_weight * kl_loss).mean()

            elbo_loss = loss.mean()

            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            self.optimiser_vae.step()

            epoch_elbo_loss += elbo_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_fixed_kl += kl_loss_fixed.item()
            epoch_sequential_kl += kl_loss_sequential.item()
            epoch_task_reconstruction_loss += task_reconstruction_loss.item() * self.args.task_loss_coeff
            epoch_naive_loss += naive_loss.item() * self.args.task_loss_coeff

        infos = {'kl_loss': (epoch_kl_loss * self.kl_weight) / self.vae_buffer.num_in_buffer,
                 'current_kl_weight': self.kl_weight,
                 'kl_fixed': (epoch_fixed_kl / self.vae_buffer.num_in_buffer),
                 'kl_sequential': (epoch_sequential_kl / self.vae_buffer.num_in_buffer),
                 'task_reconstruction_loss': (epoch_task_reconstruction_loss / self.vae_buffer.num_in_buffer),
                 'naive_task_loss': epoch_naive_loss / self.vae_buffer.num_in_buffer}

        epoch_elbo_loss /= self.vae_buffer.num_in_buffer
        self.current_update += 1

        return epoch_elbo_loss, infos

    @torch.no_grad()
    def compute_validation_loss(self, buffer: Vae_Buffer):
        epoch_elbo_loss = 0
        epoch_task_reconstruction_loss = 0
        epoch_kl_loss = 0
        epoch_fixed_kl = 0
        epoch_sequential_kl = 0
        total_norm = 0
        batch_indices = buffer.get_batches(batch_size=self.num_traj)
        for batch_index in batch_indices:
            batch, trajectory_lens = buffer.get_indexed_episode(batch_index)
            vae_prev_obs = batch['obs']
            vae_next_obs = batch['next_obs']
            vae_actions = batch['actions']
            vae_rewards = batch['rewards']
            vae_tasks = batch['task']

            # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
            _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                            states=vae_next_obs,
                                                            rewards=vae_rewards,
                                                            hidden_state=None,
                                                            return_prior=True,
                                                            detach_every=self.truncate_size,
                                                            )

            # take one sample for each ELBO term
            if not self.args.disable_stochasticity_in_latent:
                latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar, self.num_samples)
            else:
                latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)


            task_preds = self.task_decoder(latent_samples[:250, :, :])
            task_reconstruction_loss = self.task_decoder.get_loss(task_preds, vae_tasks)

            kl_loss_fixed, kl_loss_sequential = self.compute_kl_loss(latent_mean, latent_logvar)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss_fixed = kl_loss_fixed.mean(dim=0)
                kl_loss_sequential = kl_loss_sequential.mean(dim=0)
            else:
                kl_loss_fixed = kl_loss_fixed.sum(dim=0)
                kl_loss_sequential = kl_loss_sequential.sum(dim=0)
            # average across tasks
            kl_loss_fixed = kl_loss_fixed.sum(dim=0).mean()
            kl_loss_sequential = kl_loss_sequential.sum(dim=0).mean()

            if self.args.kl_to_gauss_prior:
                kl_loss = kl_loss_fixed
            else:
                kl_loss = kl_loss_sequential

            loss = (self.args.task_loss_coeff * task_reconstruction_loss +
                    self.kl_weight * kl_loss).mean()

            elbo_loss = loss.mean()

            epoch_elbo_loss += elbo_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_fixed_kl += kl_loss_fixed.item()
            epoch_sequential_kl += kl_loss_sequential.item()
            epoch_task_reconstruction_loss += task_reconstruction_loss.item() * self.args.task_loss_coeff

        infos = {'kl_loss': (epoch_kl_loss * self.kl_weight) / buffer.num_in_buffer,
                 'current_kl_weight': self.kl_weight,
                 'kl_fixed': (epoch_fixed_kl / buffer.num_in_buffer),
                 'kl_sequential': (epoch_sequential_kl / buffer.num_in_buffer),
                 'task_reconstruction_loss': (epoch_task_reconstruction_loss / buffer.num_in_buffer)}
        epoch_elbo_loss /= buffer.num_in_buffer

        return epoch_elbo_loss, infos

