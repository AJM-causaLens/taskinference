from agents.base_agent import BaseAgent

import torch
import torch.nn.functional as F

class TiAgent(BaseAgent):
    def __init__(self, config):
        super(TiAgent, self).__init__(config)
        assert self.decode_reward
        assert not self.decode_task
        assert self.contrastive_task_loss
        assert self.use_kl_loss

    def get_reward_recon_loss(self, vae_prev_obs, vae_actions, vae_next_obs, latent_samples,
                              vae_rewards) -> torch.Tensor:
        latent_samples = latent_samples[:-1, :, :]
        rew_pred = self.reward_decoder(latent_samples, vae_next_obs, vae_prev_obs, vae_actions.float())
        if self.det_rew_decoder:
            rew_reconstruction_loss = torch.nn.MSELoss(reduction='sum')(rew_pred, vae_rewards)
        else:
            rew_reconstruction_loss = self.reward_decoder.get_loss(rew_pred, vae_rewards)

        return rew_reconstruction_loss

    def get_contrastive_task_loss(
            self,
            latent_mean: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            task_labels: torch.Tensor,
            temperature: float = 0.1,
    ) -> torch.Tensor:
        """

        """
        t_steps, batch_size, dim = latent_mean.shape
        device = latent_mean.device

        total_loss = 0.0
        for timestep in range(t_steps - 1):
            queries = latent_mean[timestep]  # shape: (batch_size, dim)
            keys = latent_mean[timestep + 1]  # shape: (batch_size, dim)

            # 3) Compute pairwise Euclidean distances
            #    cdist(queries, keys) => shape (batch_size, batch_size)
            distances = torch.cdist(queries, keys, p=2)

            # 4) Convert distance to "similarity" for InfoNCE: similarity = - distance^2 / temperature
            similarities = - (distances ** 2) / temperature

            # 5) Diagonal in row i is the positive key for query i
            labels = torch.arange(batch_size, device=device)

            # 6) Standard cross-entropy (InfoNCE) for each row
            loss_t = F.cross_entropy(similarities, labels)
            total_loss += (2 * timestep / t_steps) * loss_t

        return total_loss / t_steps













