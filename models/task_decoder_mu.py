import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MetaVAE(nn.Module):
    """
    A Variational Autoencoder for meta-reinforcement learning.

    This VAE uses a recurrent encoder (LSTM) to process trajectories of transitions,
    where each transition is a concatenation of observation, action, reward, and next observation.
    The encoder produces a latent Gaussian distribution, parameterized by its mean $\mu$
    and log-variance $\log \sigma^2$. A latent sample is drawn via the reparameterization trick:

        $$
        z = \mu + \exp\!\Bigl(0.5\,\log \sigma^2\Bigr) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I),
        $$

    and is fed into a task decoder which is trained to predict the task description (i.e. the
    reward function parameters for a given MDP).

    The training loss is a sum of a reconstruction loss (here, mean-squared error) and a KL divergence term:

        $$
        \mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \mathcal{L}_{\mathrm{KL}},
        $$
    where
        $$
        \mathcal{L}_{\mathrm{KL}} = -\frac{1}{2}\mathbb{E}\Bigl[1 + \log\sigma^2 - \mu^2 - \sigma^2\Bigr].
        $$

    The training routine expects a buffer of transitions. Each element of the buffer should be a
    dictionary with the keys: 'observation', 'action', 'rewards', 'next_observation', and 'task'
    (the ground truth task description for the MDP from which the transition came).
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, task_dim, num_layers=1):
        """
        Initialize the MetaVAE.

        :param input_dim: Dimension of the concatenated input (observation, action, reward, next_observation)
        :param hidden_dim: Hidden dimension of the LSTM encoder and intermediate layers in the decoder
        :param latent_dim: Dimension of the latent variable z
        :param task_dim: Dimension of the task description output
        :param num_layers: Number of LSTM layers (default: 1)
        """
        super(MetaVAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Recurrent encoder: LSTM processes sequences of transitions.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Task decoder: predicts the task description from the latent sample.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_dim)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample from the latent Gaussian.

        :param mu: Mean tensor of the latent Gaussian.
        :param logvar: Log variance tensor of the latent Gaussian.
        :return: A latent sample z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the encoder and decoder.

        :param x: Input sequence tensor of shape (batch, sequence_length, input_dim).
        :return: A tuple (task_pred, mu, logvar), where task_pred is the predicted task description.
        """
        # Run the LSTM encoder.
        # h_n has shape (num_layers, batch, hidden_dim); we use the last layer's hidden state.
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # shape: (batch, hidden_dim)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)
        task_pred = self.decoder(z)
        return task_pred, mu, logvar

    def compute_loss(self, task_pred, task_target, mu, logvar):
        """
        Compute the total VAE loss.

        The loss is the sum of the reconstruction loss and the KL divergence loss.

        :param task_pred: Predicted task description.
        :param task_target: Ground truth task description.
        :param mu: Mean tensor of the latent Gaussian.
        :param logvar: Log variance tensor of the latent Gaussian.
        :return: A tuple (total_loss, rec_loss, kl_loss).
        """
        rec_loss = F.mse_loss(task_pred, task_target, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_loss + kl_loss, rec_loss, kl_loss

    def train_vae(self, buffer, batch_size=32, num_epochs=100):
        """
        Train the VAE using transitions from a buffer.

        The buffer is expected to be a list of dictionaries, where each dictionary contains:
            - 'observation'
            - 'action'
            - 'rewards'
            - 'next_observation'
            - 'task' (ground truth task description)

        This method concatenates the features of each transition to form a sequence.
        It then performs a standard training loop over the provided number of epochs.

        :param buffer: List of transition dictionaries.
        :param batch_size: (Unused in this simple implementation; assuming one trajectory per call)
        :param num_epochs: Number of training epochs.
        :raises ValueError: If no task target is found in the buffer.
        """
        self.train()  # Set module to training mode.

        # Build the sequence from the buffer.
        data = []
        task_targets = []
        for transition in buffer:
            # Extract features.
            obs = transition['observation']
            action = transition['action']
            reward = transition['rewards']
            next_obs = transition['next_observation']

            # Ensure all features are tensors.
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float)
            if not torch.is_tensor(action):
                action = torch.tensor(action, dtype=torch.float)
            # Reward is treated as a scalar; wrap it in a tensor.
            if not torch.is_tensor(reward):
                reward = torch.tensor([reward], dtype=torch.float)
            if not torch.is_tensor(next_obs):
                next_obs = torch.tensor(next_obs, dtype=torch.float)

            # Concatenate along the last dimension.
            transition_tensor = torch.cat([obs, action, reward, next_obs], dim=-1)
            data.append(transition_tensor.unsqueeze(0))  # add sequence dimension (1, feature_dim)

            # Gather the task target if provided.
            task_target = transition.get('task', None)
            if task_target is not None:
                if not torch.is_tensor(task_target):
                    task_target = torch.tensor(task_target, dtype=torch.float)
            task_targets.append(task_target)

        # Assume the buffer represents one trajectory; stack into a sequence tensor.
        # Final shape: (1, sequence_length, input_dim)
        sequence = torch.cat(data, dim=0).unsqueeze(0)

        # Use the first available task target from the buffer.
        task_target = None
        for t in task_targets:
            if t is not None:
                task_target = t
                break
        if task_target is None:
            raise ValueError("No task target found in buffer transitions.")
        # Expand to batch dimension.
        task_target = task_target.unsqueeze(0)  # shape: (1, task_dim)

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            task_pred, mu, logvar = self.forward(sequence)
            loss, rec_loss, kl_loss = self.compute_loss(task_pred, task_target, mu, logvar)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Rec: {rec_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}")


# Example usage:
if __name__ == "__main__":
    # Dimensions (example values):
    obs_dim = 10      # e.g., 10-dimensional observation
    action_dim = 4    # e.g., 4-dimensional action
    reward_dim = 1    # scalar reward (wrapped as a 1D tensor)
    next_obs_dim = 10
    input_dim = obs_dim + action_dim + reward_dim + next_obs_dim
    hidden_dim = 64
    latent_dim = 16
    task_dim = 8      # e.g., 8-dimensional task description

    # Initialize the VAE.
    vae = MetaVAE(input_dim, hidden_dim, latent_dim, task_dim)

    # Create a dummy buffer (list of transitions).
    # Here we create a buffer with 20 transitions; each transition has random data.
    # Note: each dictionary includes a 'task' key. In practice, this should be the true task description.
    dummy_buffer = []
    task_example = torch.randn(task_dim)  # ground truth task description for this MDP
    for _ in range(20):
        transition = {
            "observation": torch.randn(obs_dim),
            "action": torch.randn(action_dim),
            "rewards": torch.randn(1).item(),  # reward as a scalar
            "next_observation": torch.randn(next_obs_dim),
            "task": task_example  # same task description for all transitions in this buffer
        }
        dummy_buffer.append(transition)

    # Train the VAE on the dummy buffer.
    vae.train_vae(dummy_buffer, num_epochs=50)
