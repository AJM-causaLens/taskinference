from datetime import datetime
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import json

from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder
from models.varibad_buffer import Vae_Buffer


class TaskEncoder(nn.Module):
    def __init__(self, args):
        super(TaskEncoder, self).__init__()
        self.args = args
        self.args = args
        self.num_traj = args.vae_batch_num_trajs
        self.task_dim = args.task_dim
        self.vae_buffer = Vae_Buffer(max_size=args.size_vae_buffer, batch_size=args.vae_batch_num_trajs,
                                     decode_task=args.decode_task)
        self.encoder = self.initialise_encoder()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_vae)


    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.task_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        )
        return encoder

    def compute_encoder_loss(self):
        """
        Computes the encoder loss including the reconstruction term,
        the KL divergence, and also computes a dumb reward loss
        based on constant predictions from the task distribution.
        """
        epoch_elbo_loss = 0.0
        epoch_task_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_dumb_loss = 0.0

        # Get batch indices from your VAE buffer
        batch_indices = self.vae_buffer.get_batches(batch_size=self.num_traj)

        for batch_index in batch_indices:
            # Get the batch and corresponding trajectory lengths
            batch, trajectory_lens = self.vae_buffer.get_indexed_episode(batch_index)
            vae_tasks = batch.get('task')
            vae_actions = batch['actions']
            vae_next_obs = batch['next_obs']
            vae_rewards = batch['rewards']

            # Pass through encoder.
            # Expected output shape: (max_traj_len+1) x num_rollouts x latent_dim.
            # The encoder returns the prior as well; here we assume that the outputs are:
            # (_, task_mean, task_logvar, _)
            _, task_mean, task_logvar, _ = self.encoder(
                actions=vae_actions,
                states=vae_next_obs,
                rewards=vae_rewards,
                hidden_state=None,
                return_prior=True,
                detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
            )

            # Remove final time step to match the task targets
            task_mean = task_mean[:-1]
            task_logvar = task_logvar[:-1]
            task_var = torch.exp(task_logvar)

            # Compute reconstruction loss (negative log likelihood) of the encoder.
            # Equation: 0.5 * (log(sigma_t^2) + ((x_t - mu_t)^2 / (sigma_t^2 + 1e-6)))
            recon_nll = 0.5 * (task_logvar + (vae_tasks - task_mean) ** 2 / (task_var + 1e-6))
            recon_loss = recon_nll.sum()

            # Compute KL divergence term.
            # Equation: -0.5 * sum(1 + log(sigma_t^2) - mu_t^2 - sigma_t^2)
            kl_loss = -0.5 * torch.sum(1 + task_logvar - task_mean ** 2 - task_var)

            # Compute the dumb predictions:
            # Calculate the overall mean and standard deviation of the task targets.
            # We compute these statistics over all timesteps and rollouts.
            dumb_mean = torch.tensor(self.task_mean).unsqueeze(0).unsqueeze(0)
            dumb_std = torch.tensor(1).unsqueeze(0).unsqueeze(0)
            dumb_logvar = torch.log(dumb_std ** 2)

            # Expand dumb predictions to match the shape of the encoder predictions.
            # Note: vae_tasks might have an extra time step compared to task_mean.
            dumb_mean_expanded = dumb_mean.expand_as(vae_tasks)
            dumb_logvar_expanded = dumb_logvar.expand_as(task_logvar)

            # Compute dumb negative log likelihood loss.
            dumb_nll = 0.5 * (dumb_logvar_expanded +
                              (vae_tasks - dumb_mean_expanded) ** 2 /
                              (torch.exp(dumb_logvar_expanded) + 1e-6))
            dumb_loss = dumb_nll.sum()

            # Total loss is the ELBO: reconstruction loss + KL divergence.
            # The dumb_loss is computed for comparison purposes and is logged.
            total_loss = recon_loss + kl_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            epoch_elbo_loss += total_loss.item()
            epoch_task_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_dumb_loss += dumb_loss.item()

        infos = {
            'elbo_loss': epoch_elbo_loss/self.vae_buffer.num_in_buffer,
            'task_loss': epoch_task_loss/self.vae_buffer.num_in_buffer,
            'kl_loss': epoch_kl_loss/self.vae_buffer.num_in_buffer,
            'dumb_loss': epoch_dumb_loss/self.vae_buffer.num_in_buffer,
        }

        return epoch_elbo_loss, infos

    def compute_validation_loss(self, val_buffer):
        """
        Computes the encoder validation loss including the reconstruction term,
        the KL divergence, and also computes a dumb reward loss based on constant
        predictions from the task distribution. This function uses a validation
        buffer and does not compute gradients.

        Returns:
            epoch_elbo_loss (float): Total ELBO loss over the validation set.
            infos (dict): Dictionary containing individual loss components.
        """
        epoch_elbo_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_task_loss = 0.0
        epoch_dumb_loss = 0.0

        # Get batch indices from the validation buffer.
        batch_indices = val_buffer.get_batches(batch_size=self.num_traj)

        with torch.no_grad():
            for batch_index in batch_indices:
                # Retrieve batch data and trajectory lengths.
                batch, trajectory_lens = val_buffer.get_indexed_episode(batch_index)
                vae_tasks = batch.get('task')
                vae_actions = batch['actions']
                vae_next_obs = batch['next_obs']
                vae_rewards = batch['rewards']

                # Pass through the encoder.
                # Expected output shape: (max_traj_len+1) x num_rollouts x latent_dim.
                # The encoder returns the prior as well; here we assume that the outputs are:
                # (_, task_mean, task_logvar, _)
                _, task_mean, task_logvar, _ = self.encoder(
                    actions=vae_actions,
                    states=vae_next_obs,
                    rewards=vae_rewards,
                    hidden_state=None,
                    return_prior=True,
                    detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                )

                # Remove the final time step to align with the task targets.
                task_mean = task_mean[:-1]
                task_logvar = task_logvar[:-1]
                task_var = torch.exp(task_logvar)

                # Compute the reconstruction loss (negative log likelihood).
                recon_nll = 0.5 * (task_logvar + (vae_tasks - task_mean) ** 2 / (task_var + 1e-6))
                recon_loss = recon_nll.sum()

                # Compute the KL divergence term.
                kl_loss = -0.5 * torch.sum(1 + task_logvar - task_mean ** 2 - task_var)

                # Compute the dumb predictions using constant values.
                # Here, self.task_mean is used as the constant mean and we set the std to 1.
                dumb_mean = torch.tensor(self.task_mean).unsqueeze(0).unsqueeze(0)
                dumb_std = torch.tensor(1).unsqueeze(0).unsqueeze(0)
                dumb_logvar = torch.log(dumb_std ** 2)

                # Expand dumb predictions to match the encoder outputs.
                dumb_mean_expanded = dumb_mean.expand_as(vae_tasks)
                dumb_logvar_expanded = dumb_logvar.expand_as(task_logvar)

                # Compute the dumb negative log likelihood loss.
                dumb_nll = 0.5 * (dumb_logvar_expanded +
                                  (vae_tasks - dumb_mean_expanded) ** 2 /
                                  (torch.exp(dumb_logvar_expanded) + 1e-6))
                dumb_loss = dumb_nll.sum()

                # Total ELBO loss.
                total_loss = recon_loss + kl_loss

                epoch_elbo_loss += total_loss.item()
                epoch_task_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_dumb_loss += dumb_loss.item()

        infos = {
            'elbo_loss': epoch_elbo_loss/val_buffer.num_in_buffer,
            'task_loss': epoch_task_loss/val_buffer.num_in_buffer,
            'kl_loss': epoch_kl_loss/val_buffer.num_in_buffer,
            'dumb_loss': epoch_dumb_loss/val_buffer.num_in_buffer,
        }

        return epoch_elbo_loss, infos


    def save_args_as_json_or_markdown(self, args, file_path_base):
        # Convert args to a dictionary if it's an object
        os.makedirs(os.path.dirname(file_path_base), exist_ok=True)
        args.model = self.__class__.__name__
        args_dict = vars(args) if not isinstance(args, dict) else args
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"{file_path_base}/{timestamp}"
        # Save as JSON
        with open(file_path + ".json", "w") as json_file:
            json.dump(args_dict, json_file, indent=4)

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

    def save_model(self, path: str):
        """
        Save the model's state dictionary to the specified path.

        Args:
            path (str): The file path to save the model.
        """
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
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

        print(f"Model loaded from {path}")


