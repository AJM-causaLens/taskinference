import random
import numpy as np
import os

import torch as th
from typing import Dict, List


class Vae_Buffer:
    def __init__(self, max_size=150, batch_size=32, decode_task=False):
        """
        Initialize a VAE buffer for variable-length episodes.

        Args:
            max_size (int): Maximum number of episodes to store.
            batch_size (int): Number of trajectories to sample during training.
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer: List[Dict[str, th.Tensor]] = []  # List to store variable-length episodes
        self.ptr = 0  # Pointer to indicate where to overwrite
        self.num_in_buffer = 0
        self.decode_task = decode_task

    def add(self, episode: Dict[str, th.Tensor]):
        """
        Add an entire episode to the buffer.

        Args:
            episode (dict): Contains 'obs', 'next_obs', 'actions', 'rewards', 'dones'.
                            Each value is a tensor of shape (episode_length, feature_dim).
        """
        # Ensure input tensors are of valid dimensions
        assert all(isinstance(v, th.Tensor) for v in episode.values()), "All episode data must be tensors"

        if self.num_in_buffer < self.max_size:  # If buffer is not full, append the episode
            self.buffer.append(episode)
        else:  # Overwrite oldest episode
            self.buffer[self.ptr] = episode

        # Update pointer and buffer size
        self.ptr = (self.ptr + 1) % self.max_size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.max_size)

    def save(self, path, combined=False, file_name="vae_buffer"):
        """
        Saves the buffer to a file and logs buffer statistics in a Markdown file.

        Args:
            path (str): Directory where the buffer should be saved.
        """
        # Ensure save directory exists
        os.makedirs(path, exist_ok=True)

        if file_name is None:
            file_name = "vae_buffer.pth" if not combined else "vae_buffer_combined.pth"

        # Save buffer as usual
        if combined:
            file_path = os.path.join(path, file_name)

        else:
            file_path = os.path.join(path, file_name)
            # Compute buffer statistics
            num_episodes = len(self.buffer)
            episode_lengths = [len(episode['obs']) for episode in self.buffer]
            episode_rewards = [episode['rewards'].sum().item() for episode in self.buffer]
            tasks = [episode['task'] for episode in self.buffer]
            task = self.assert_tensors_equal_and_extract(tasks)

            # Save buffer info to a Markdown file
            md_file_path = os.path.join(path, "buffer_info.md")
            with open(md_file_path, "w") as f:
                f.write("# VAE Buffer Info\n\n")
                f.write(f"- **Task**: {task}\n")
                f.write(f"- **Number of episodes**: {num_episodes}\n")
                f.write(f"- **Episode lengths**: {episode_lengths}\n")
                f.write(f"- **Total rewards per episode**: {episode_rewards}\n")

            # print(f"Buffer saved at {file_path} with metadata in {md_file_path}")

        th.save(self.buffer, file_path)

    def load(self, file_path):
        self.buffer = th.load(file_path)
        self.num_in_buffer = len(self.buffer)
        self.ptr = self.num_in_buffer % self.max_size



    def get_batches(self, batch_size, total_size=None):
        assert self.num_in_buffer > 0, "Buffer is empty, cannot generate batches."
        assert batch_size > 0, "Batch size must be greater than 0."

        # Generate all indices in the buffer
        indices = list(range(self.num_in_buffer))

        # Shuffle indices for randomness
        random.shuffle(indices)
        if total_size is not None and total_size < len(indices):
            indices = indices[:total_size]

        # Create batches by slicing the shuffled indices
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        return batches


    def get_indexed_episode(self, indices: list[int]) -> [Dict[str, th.Tensor], list]:
        sampled_episodes = [self.buffer[i] for i in indices]

        lengths = [ep['obs'].shape[0] for ep in sampled_episodes]
        num_unique_trajectory_lens = len(np.unique(lengths))
        assert num_unique_trajectory_lens == 1, "All episodes must have the same length."

        batch = {
            'obs': th.cat([ep['obs'] for ep in sampled_episodes], dim=1),
            'next_obs': th.cat([ep['next_obs'] for ep in sampled_episodes], dim=1),
            'actions': th.cat([ep['actions'] for ep in sampled_episodes], dim=1),
            'rewards': th.cat([ep['rewards'] for ep in sampled_episodes], dim=1),
            "task": th.cat([ep['task'].expand(lengths[0], 1, -1) for ep in sampled_episodes], dim=1)
        }

        return batch, lengths


    def sample(self, num_traj) -> [Dict[str, th.Tensor], list]:
        """
        Sample a batch of episodes from the buffer.

        Returns:
            A dictionary of concatenated tensors: 'obs', 'next_obs', 'actions', 'rewards', 'dones'.
        """
        assert self.num_in_buffer > 0, "Buffer is empty, cannot sample."
        sampled_indices = th.randint(0, self.num_in_buffer, (num_traj,))
        # Randomly sample episodes
        sampled_episodes = [self.buffer[i] for i in sampled_indices]

        lengths = [ep['obs'].shape[0] for ep in sampled_episodes]

        # Get the maximum episode length
        max_episode_length = max(lengths)

        # Initialize padded tensors
        batch = {
            'obs': th.zeros((max_episode_length, num_traj, sampled_episodes[0]['obs'].shape[-1])),
            'next_obs': th.zeros((max_episode_length, num_traj, sampled_episodes[0]['next_obs'].shape[-1])),
            'actions': th.zeros((max_episode_length, num_traj, sampled_episodes[0]['actions'].shape[-1])),
            'rewards': th.zeros((max_episode_length, num_traj, sampled_episodes[0]['rewards'].shape[-1])),
            'dones': th.zeros((max_episode_length, num_traj, 1))  # Assuming dones is 1D
        }

        # Pad episodes and fill the batch
        for i, episode in enumerate(sampled_episodes):
            length = episode['obs'].shape[0]  # Actual episode length

            # Align dimensions by unsqueezing along the batch axis (dim=1)
            batch['obs'][:length, i] = episode['obs'].squeeze()
            batch['next_obs'][:length, i] = episode['next_obs'].squeeze()
            batch['actions'][:length, i] = episode['actions'].squeeze()
            batch['rewards'][:length, i] = episode['rewards'].squeeze(1)

        return batch, lengths

    @staticmethod
    def assert_tensors_equal_and_extract(tensor_list):
        """
        Checks if all tensors in a list are equal and extracts their value.

        Args:
            tensor_list (list of torch.Tensor): List of tensors to check.

        Returns:
            scalar value: The extracted value if all tensors are equal.

        Raises:
            AssertionError: If any tensor differs from the others.
        """
        # Convert all tensors to CPU and detach them if needed
        tensor_list = [t.cpu().detach() if t.requires_grad else t.cpu() for t in tensor_list]

        # Ensure all tensors are equal
        for t in tensor_list[1:]:
            assert th.all(t == tensor_list[0]), f"Tensors are not equal: {tensor_list}"

        # Extract scalar value (assuming all tensors are equal)
        extracted_value = tensor_list[0].tolist()
        return extracted_value
