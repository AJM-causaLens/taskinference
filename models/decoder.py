import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardDecoderProbabilistic(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 input_prev_state=True,
                 input_action=True,
                 ):
        super(RewardDecoderProbabilistic, self).__init__()

        self.args = args
        self.input_prev_state = input_prev_state
        self.input_action = input_action
        self.criterion = nn.MSELoss(reduction='none')
        # get state as input and predict reward prob
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.leaky_relu)
        if self.input_action:
            self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.leaky_relu)
        else:
            self.action_encoder = None
        curr_input_dim = latent_dim + state_embed_dim
        if input_prev_state:
            curr_input_dim += state_embed_dim
        if input_action:
            curr_input_dim += action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]
        self.fc_mean = nn.Linear(curr_input_dim, 1)
        self.fc_logvar = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):

        hns = self.state_encoder(next_state)
        h = torch.cat((latent_state, hns), dim=-1)
        if self.input_action:
            ha = self.action_encoder(actions)
            h = torch.cat((h, ha), dim=-1)
        if self.input_prev_state:
            hps = self.state_encoder(prev_state)
            h = torch.cat((h, hps), dim=-1)

        for layer in self.fc_layers:
            h = F.leaky_relu(layer(h))

        return self.fc_mean(h), self.fc_logvar(h)

    def get_loss(self, pred_reward, reward_labels):
        mean, logvar = pred_reward
        var = torch.exp(logvar)

        neg_log_likelihood = 0.5 * (logvar + (reward_labels - mean) ** 2 / (var + 1e-6))

        return neg_log_likelihood.sum()


class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        # get state as input and predict reward prob
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.leaky_relu)
        if self.input_action:
            self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.leaky_relu)
        else:
            self.action_encoder = None
        curr_input_dim = latent_dim + state_embed_dim
        if input_prev_state:
            curr_input_dim += state_embed_dim
        if input_action:
            curr_input_dim += action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):

        # we do the action-normalisation (the env bounds) here
        if actions is not None:
            actions = utl.squash_action(actions, self.args)

        if self.multi_head:
            h = latent_state.clone()
        else:
            hns = self.state_encoder(next_state)
            h = torch.cat((latent_state, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)

        for layer in self.fc_layers:
            h = F.leaky_relu(layer(h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                time_weighted_loss,
                 ):
        super(TaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type
        self.time_weighted_loss = time_weighted_loss
        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out = nn.Linear(curr_input_dim, output_dim)
        if self.time_weighted_loss:
            self.criterion = nn.MSELoss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction="sum")
            # self.criterion = nn.L1Loss()

    def forward(self, latent_state):

        h = latent_state

        for layer in self.fc_layers:
            h = F.leaky_relu(layer(h))

        return self.fc_out(h)

    def get_loss(self, task, task_pred):
        """ Compute task reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """
        if self.time_weighted_loss:
            loss = self.criterion(task_pred.float(), task)
            sequence_length = len(loss)  # Avoid div by zero
            weights = torch.linspace(0, 1, steps=sequence_length)  # Shape: (250,)
            weights = weights.view(sequence_length, 1, 1)  # Shape: (250, 1, 1)
            weighted_loss = loss * weights  # Shape: (250, 8, 2)
            return weighted_loss.sum()

        else:
            return self.criterion(task_pred.float(), task)



class TaskDecoderProbabilistic(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 time_weighted_loss=False,
                 ):
        """
        Probabilistic Task Decoder: Outputs a mean and log variance instead of a deterministic task prediction.

        Args:
            layers (list): List of layer sizes for the feedforward network.
            latent_dim (int): Dimensionality of the latent state input.
            pred_type (str): 'task_description' or 'task_id', determines output size.
            task_dim (int): Dimensionality of the task representation (if 'task_description').
            num_tasks (int): Number of tasks (if 'task_id').
        """
        super(TaskDecoderProbabilistic, self).__init__()

        self.time_weighted_loss = time_weighted_loss

        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks

        # Separate layers for mean and log variance
        self.fc_mean = nn.Linear(curr_input_dim, output_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):
        """
        Forward pass through the network.

        Args:
            latent_state (Tensor): Input latent state of shape (..., latent_dim).

        Returns:
            mean (Tensor): Predicted mean task representation (same shape as task_dim).
            logvar (Tensor): Predicted log variance.
        """
        h = latent_state

        for layer in self.fc_layers:
            h = F.leaky_relu(layer(h))

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)  # No activation to keep logvar unrestricted

        return mean, logvar

    def get_loss(self, pred_task, task_labels):
        """
        Computes the Gaussian negative log-likelihood (NLL) loss.

        Args:
            pred_task (tuple): Output from forward() -> (mean, logvar)
            task_labels (Tensor): True task labels of shape (..., task_dim).

        Returns:
            loss (Tensor): Computed NLL loss.
        """
        mean, logvar = pred_task  # Unpack predicted mean and logvar

        # Compute negative log-likelihood loss
        var = torch.exp(logvar)  # Convert log variance to variance
        neg_log_likelihood = 0.5 * (logvar + (task_labels - mean) ** 2 / (var + 1e-6))

        if self.time_weighted_loss:
            sequence_length = len(neg_log_likelihood)# Avoid div by zero
            weights = torch.linspace(0, 1, steps=sequence_length)  # Shape: (250,)
            weights = weights.view(sequence_length, 1, 1)  # Shape: (250, 1, 1)
            weighted_nll = neg_log_likelihood * weights  # Shape: (250, 8, 2)
            return weighted_nll.sum()


        return neg_log_likelihood.sum()  # Sum the weighted loss



class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 pred_type='deterministic'
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.leaky_relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.leaky_relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for layer in self.fc_layers:
            h = F.leaky_relu(layer(h))

        return self.fc_out(h)