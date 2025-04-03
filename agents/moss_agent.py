import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent

class MossAgent(BaseAgent):
    def __init__(self, config):
        super(MossAgent, self).__init__(config)
        self.temperature = 0.1
        assert self.decode_reward
        assert not self.decode_task
        assert self.contrastive_task_loss
        assert self.use_kl_loss

    def get_contrastive_task_loss(
            self,
            latent_mean: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            task_labels: torch.Tensor,
            temperature: float = 0.1,
            num_random_pairs: int = 3
    ) -> torch.Tensor:
        """
        Reimplementation of an InfoNCE-style contrastive task loss that:
          1) Uses Euclidean distance as the similarity metric (negated distance).
          2) Randomly samples pairs of time steps rather than consecutive time steps.

        Args:
            latent_mean (torch.Tensor): Shape (num_timesteps, batch_size, latent_dim).
                `latent_mean[t, i, :]` is the embedding for task i at time t.
            states, actions, rewards, task_labels (torch.Tensor): Unused here,
                but included for interface consistency.
            temperature (float): Temperature scaling factor for the InfoNCE softmax.
            num_random_pairs (int): Number of (t1, t2) time-step pairs to sample per forward pass.

        Returns:
            torch.Tensor: Averaged InfoNCE loss over the randomly sampled time-step pairs.
        """
        t_steps, batch_size, dim = latent_mean.shape
        device = latent_mean.device

        total_loss = 0.0
        for _ in range(num_random_pairs):
            # 1) Randomly sample two distinct time steps
            t1 = torch.randint(low=0, high=t_steps, size=(1,), device=device)
            t2 = torch.randint(low=0, high=t_steps, size=(1,), device=device)
            while t2 == t1:
                t2 = torch.randint(low=0, high=t_steps, size=(1,), device=device)

            t1, t2 = t1.item(), t2.item()

            # 2) Queries and keys from the randomly chosen time steps
            queries = latent_mean[t1]  # shape: (batch_size, dim)
            keys = latent_mean[t2]  # shape: (batch_size, dim)

            # 3) Compute pairwise Euclidean distances
            #    cdist(queries, keys) => shape (batch_size, batch_size)
            distances = torch.cdist(queries, keys, p=2)

            # 4) Convert distance to "similarity" for InfoNCE: similarity = - distance^2 / temperature
            similarities = - (distances ** 2) / temperature

            # 5) Diagonal in row i is the positive key for query i
            labels = torch.arange(batch_size, device=device)

            # 6) Standard cross-entropy (InfoNCE) for each row
            loss_t = F.cross_entropy(similarities, labels)
            total_loss += loss_t

        # Average over all sampled (t1, t2) pairs
        return total_loss / num_random_pairs


    def get_reward_recon_loss(self, vae_prev_obs, vae_actions, vae_next_obs, latent_samples,
                              vae_rewards) -> torch.Tensor:
        latent_samples = latent_samples[:-1, :, :]
        rew_pred = self.reward_decoder(latent_samples, vae_next_obs, vae_prev_obs, vae_actions.float())
        if self.det_rew_decoder:
            rew_reconstruction_loss = torch.nn.MSELoss(reduction='sum')(rew_pred, vae_rewards)
        else:
            rew_reconstruction_loss = self.reward_decoder.get_loss(rew_pred, vae_rewards)

        return rew_reconstruction_loss



