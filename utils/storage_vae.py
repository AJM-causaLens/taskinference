import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorageVAE(object):
    def __init__(self, max_trajectory_len, zero_pad, max_num_rollouts,
                 state_dim, action_dim, vae_buffer_add_thresh, task_dim):
        """
        Store everything that is needed for the VAE update
        """

        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  # prob of adding new trajectories
        self.max_buffer_size = max_num_rollouts  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_traj_len = max_trajectory_len
        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        # storage for each running process (stored on GPU)
        self.curr_timestep = torch.tensor(0).long()  # count environment steps so we know where to insert
        self.running_prev_state = torch.zeros((self.max_traj_len, state_dim)).to(device)  # for each episode will have obs 0...N-1
        self.running_next_state = torch.zeros((self.max_traj_len, state_dim)).to(device)  # for each episode will have obs 1...N
        self.running_rewards = torch.zeros((self.max_traj_len, 1)).to(device)
        self.running_actions = torch.zeros((self.max_traj_len, action_dim)).to(device)
        if task_dim is not None:
            self.running_tasks = torch.zeros((task_dim)).to(device)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):

        assert isinstance(done, int), "done must be an integer"

        # add to temporary buffer
        # print("rewards", rewards)
        # print("current index", self.insert_idx)
        # print("current timestep", self.curr_timestep)
        # print("done", done)
        self.running_prev_state[self.curr_timestep] = prev_state
        self.running_next_state[self.curr_timestep] = next_state
        self.running_rewards[self.curr_timestep] = rewards
        self.running_actions[self.curr_timestep] = actions
        if task is not None:
            self.running_tasks = task
        self.curr_timestep += 1


        if done == 1:  # check if we can process the entire batch at once

            # add to permanent (up to max_buffer_len) buffer
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    # check where to insert data
                    if self.insert_idx + 1 > self.max_buffer_size:
                        # keep track of how much we filled the buffer (for sampling from it)
                        self.buffer_len = self.insert_idx
                        # this will keep some entries at the end of the buffer without overwriting them,
                        # but the buffer is large enough to make this negligible
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    # add; note: num trajectories are along dim=1,
                    # trajectory length along dim=0, to match pytorch RNN interface
                    self.prev_state[:, self.insert_idx] = self.running_prev_state
                    self.next_state[:, self.insert_idx] = self.running_next_state
                    self.actions[:, self.insert_idx] = self.running_actions
                    self.rewards[:, self.insert_idx] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx].shape
                        self.tasks[self.insert_idx] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx] = self.curr_timestep.clone()
                    self.insert_idx += 1

            # empty running buffer
            self.running_prev_state *= 0
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0


    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)

        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]

        # select the rollouts we want
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(device)
        else:
            tasks = None

        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
               rewards.to(device), tasks, trajectory_lens
