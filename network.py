import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F




def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.GRUCell):
        nn.init.xavier_uniform_(layer.weight_ih)
        nn.init.xavier_uniform_(layer.weight_hh)
        nn.init.zeros_(layer.bias_ih)
        nn.init.zeros_(layer.bias_hh)


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) for learning latent dynamics.
    """
    def __init__(self, input_latent_shape, action_dim, hidden_dim, num_parallel_grus=8, num_serial_grus=20):
        """
        Initializes the RSSM.

        Args:
            input_latent_shape (int): The shape of the input latent state.
            action_dim (int): The dimension of the action space.
            hidden_dim (int): The dimension of the hidden state.
            num_parallel_grus (int): The number of parallel GRU cells.
            num_serial_grus (int): The number of serial GRU cells.
        """
        super(RSSM, self).__init__()
        self.input_latent_shape = input_latent_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_parallel_grus = num_parallel_grus
        self.num_serial_grus = num_serial_grus

        # Split hidden_dim among parallel GRUs
        self.split_hidden_dim = hidden_dim // num_parallel_grus
        print(hidden_dim, num_parallel_grus)
        assert hidden_dim % num_parallel_grus == 0, "hidden_dim must be divisible by num_parallel_grus"

        # Original layers for processing z_t and a_t
        self.process_zt_and_at_for_gru = nn.Linear(input_latent_shape + action_dim, 512)
        # Layer to process the output of the GRU
        self.process_gru_output = nn.Linear(hidden_dim, 512)
        # Layer to generate z_t from the GRU output
        self.zt_from_gru_output = nn.Linear(512, input_latent_shape)
        # Layer to generate z_{t+1} from the previous hidden state and current observation
        self.zt_from_ht_1_and_zt1 = nn.Linear(input_latent_shape + hidden_dim, input_latent_shape)

        # Multiple parallel and serial GRU cells
        self.gru_cells = nn.ModuleList([
            nn.ModuleList([
                nn.GRUCell(512 + self.split_hidden_dim, self.split_hidden_dim)
                for _ in range(num_serial_grus)
            ])
            for _ in range(num_parallel_grus)
        ])

        # Mixing layers between serial GRUs
        self.mixing_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_serial_grus - 1)
        ])

    def forward(self, prev_obs, prev_action, obs, hidden):
        """
        Forward pass of the RSSM.

        Args:
            prev_obs (torch.Tensor): The previous observation. Shape: (batch_size, ...)
            prev_action (torch.Tensor): The previous action. Shape: (batch_size, action_dim)
            obs (torch.Tensor): The current observation. Shape: (batch_size, ...)
            hidden (torch.Tensor): The previous hidden state. Shape: (batch_size, hidden_dim)

        Returns:
            tuple: A dictionary containing logits and samples for zt_gru and zt1, and the next hidden state ht.
        """
        # prev_obs_shape (batch_size, 32,32)
        # obs_shape (batch_size, 32,32)
        # prev_action_shape (batch_size, action_dim)
        # hidden_shape (batch_size, hidden_dim)
        batch_size = prev_obs.size(0)

        # Calculate zt_gru from previous observation, action, and hidden state
        zt_gru_logits, zt_gru, ht = self.calculate_zt_gru_from_prev_obs(prev_obs, prev_action, hidden)

        # Using current observation and hidden state to predict the current state zt1
        obs_flat = obs.view(batch_size, -1)  # Flatten the observation
        zt1_and_ht = torch.cat([obs_flat, ht], dim=-1)
        zt1_logits = self.zt_from_ht_1_and_zt1(zt1_and_ht)
        zt1 = F.gumbel_softmax(zt1_logits, tau=1.0, hard=True)

        return {
            "zt_gru_logits": zt_gru_logits,
            "zt_gru": zt_gru,
            "zt1_logits": zt1_logits,
            "zt1": zt1,
        }, ht

    def kl_loss(self, zt_gru_logits, zt1_logits):
        """
        Calculates the KL divergence between the predicted state and the actual state.

        Args:
            zt_gru_logits (torch.Tensor): Logits of the predicted state.
            zt1_logits (torch.Tensor): Logits of the actual state.

        Returns:
            torch.Tensor: The KL divergence loss.
        """
        zt_gru_dist = torch.distributions.Categorical(logits=zt_gru_logits)
        zt1_dist = torch.distributions.Categorical(logits=zt1_logits)
        kl_loss = torch.distributions.kl.kl_divergence(zt_gru_dist, zt1_dist)
        return kl_loss

    def calculate_zt_gru_from_prev_obs(self, prev_obs, prev_action, hidden):
        """
        Calculates the predicted next latent state (zt_gru) using the previous observation, action, and hidden state.

        Args:
            prev_obs (torch.Tensor): The previous observation. Shape: (batch_size, ...)
            prev_action (torch.Tensor): The previous action. Shape: (batch_size, action_dim)
            hidden (torch.Tensor): The previous hidden state. Shape: (batch_size, hidden_dim)

        Returns:
            tuple: Logits for zt_gru, the sampled zt_gru, and the next hidden state ht.
        """
        batch_size = prev_obs.size(0)

        # using gru as dynamic model to predict future state which is zt_gru
        prev_obs_flat = prev_obs.view(batch_size, -1)  # Flatten obs to (batch_size, input_latent_size)
        # initialized with obs at first time step then it will become the previous predicted observation
        # (like zt_gru from previous timestep) not the actual observation

        prev_obs_flat_and_action = torch.cat([prev_obs_flat, prev_action], dim=-1)  # size (batch_size, input_latent_size + action_dim)
        zt_and_at = self.process_zt_and_at_for_gru(prev_obs_flat_and_action)  # size (batch_size, 512)
        zt_and_at = F.relu(zt_and_at)

        # Split hidden state for parallel processing
        hidden_chunks = torch.chunk(hidden, self.num_parallel_grus, dim=-1)

        # Process through parallel and serial GRU cells
        next_hidden_chunks = []
        current_hidden_chunks = hidden_chunks

        for serial_idx in range(self.num_serial_grus):
            next_hidden_states = []

            # Process each parallel branch
            for parallel_idx in range(self.num_parallel_grus):
                # Prepare input for each GRU cell
                gru_input = torch.cat([zt_and_at, current_hidden_chunks[parallel_idx]], dim=-1)
                # Process through GRU cell
                next_hidden = self.gru_cells[parallel_idx][serial_idx](gru_input, current_hidden_chunks[parallel_idx])
                next_hidden = F.relu(next_hidden)
                next_hidden_states.append(next_hidden)

            # Combine hidden states for mixing
            combined_hidden = torch.cat(next_hidden_states, dim=-1)

            # Apply mixing layer between serial steps (except for the last step)
            if serial_idx < self.num_serial_grus - 1:
                mixed_hidden = self.mixing_layers[serial_idx](combined_hidden)
                mixed_hidden = F.relu(mixed_hidden)
                current_hidden_chunks = torch.chunk(mixed_hidden, self.num_parallel_grus, dim=-1)
            else:
                next_hidden_chunks = next_hidden_states

        # Combine final hidden states
        ht = torch.cat(next_hidden_chunks, dim=-1)

        # Process final hidden state
        ht_processed = self.process_gru_output(ht)  # size (batch_size, 512)
        ht_processed = F.relu(ht_processed)
        zt_gru_logits = self.zt_from_gru_output(ht_processed)
        zt_gru_logits = self.add_uniform_noise(zt_gru_logits)
        zt_gru = F.gumbel_softmax(zt_gru_logits, tau=1.0, hard=True)

        return zt_gru_logits, zt_gru, ht

    def add_uniform_noise(self, zt_logits, mix_ratio=0.01):
        """
        Adds uniform noise to the logits for exploration.

        Args:
            zt_logits (torch.Tensor): The logits to add noise to.
            mix_ratio (float): The mixing ratio for the uniform distribution.

        Returns:
            torch.Tensor: The noisy logits.
        """
        probs = F.softmax(zt_logits, dim=-1)
        uniform = torch.ones_like(probs) / probs.size(-1)
        probs = (1 - mix_ratio) * probs + mix_ratio * uniform
        return torch.log(probs)

class Reward_predictor(nn.Module):
    def __init__(self, input_shape, num_bin=32):
        super(Reward_predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bin)
        )
        self.fc.apply(xavier_init)

    def forward(self, x):
        return F.relu(self.fc(x))

class Continue_predictor(nn.Module):
    def __init__(self, input_shape):
        super(Continue_predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc.apply(xavier_init)

    def forward(self, x):
        return F.relu(self.fc(x))

class Value_predictor(nn.Module):
    def __init__(self, input_shape, num_bin=32):
        super(Value_predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bin)
        )
        self.fc.apply(xavier_init)

    def forward(self, x):
        return F.relu(self.fc(x))

class WorldModel(nn.Module):
    """
    Combines the RSSM, Reward_predictor, Continue_predictor, and Value_predictor.
    """
    def __init__(self, config):
        """
        Initializes the WorldModel.

        Args:
            input_latent_shape (int): The shape of the input latent state.
            action_dim (int): The dimension of the action space.
            hidden_dim (int): The dimension of the hidden state.
            num_bins (int): The number of bins for reward prediction.
            num_parallel_grus (int): The number of parallel GRU cells in the RSSM.
            num_serial_grus (int): The number of serial GRU cells in the RSSM.
        """
        
        
        super(WorldModel, self).__init__()

        # initialize the rssm model, reward_predictor, continue_predictor, value_predictor
        self.rssm = RSSM(config.world_model_input_latent_shape, config.world_model_action_dim, config.world_model_hidden_dim , num_parallel_grus=config.world_model_num_parallel_grus, num_serial_grus=config.world_model_num_serial_grus)
        self.reward_predictor = Reward_predictor(config.world_model_hidden_dim  + config.world_model_input_latent_shape, config.world_model_num_bins)
        self.continue_predictor = Continue_predictor(config.world_model_hidden_dim  + config.world_model_input_latent_shape)
        self.value_predictor = Value_predictor(
            config.world_model_hidden_dim + config.world_model_input_latent_shape,
            config.world_model_num_bins  # Pass num_bins
        )
    def forward(self, prev_obs, prev_action, obs, hidden):
        """
        Forward pass of the WorldModel.

        Args:
            prev_obs (torch.Tensor): The previous observation. Shape: (batch_size, ...)
            prev_action (torch.Tensor): The previous action. Shape: (batch_size, action_dim)
            obs (torch.Tensor): The current observation. Shape: (batch_size, ...)
            hidden (torch.Tensor): The previous hidden state. Shape: (batch_size, hidden_dim)

        Returns:
            tuple: A tuple containing the state information from RSSM, the next hidden state,
                   the predicted reward, the prediction of whether the episode continues, and the predicted value.
        """
        state, ht = self.rssm(prev_obs, prev_action, obs, hidden)
        batch_size = prev_obs.size(0)
        stoch_sample = F.gumbel_softmax(state["zt_gru_logits"], tau=1.0, hard=True).view(batch_size, -1) # stoch means not using orignal but using the predicted one
        reward_cont_value_input = torch.cat([stoch_sample, ht], dim=-1)
        reward = self.reward_predictor(reward_cont_value_input)
        cont = self.continue_predictor(reward_cont_value_input)
        value = self.value_predictor(reward_cont_value_input)
        return state, ht, reward,cont, value

    def imagine_ahead(self, prev_obs, prev_action, hidden):
        """
        Imagines the future states, rewards, continue signals, and values.

        Args:
            prev_obs (torch.Tensor): The previous observation. Shape: (batch_size, ...)
            prev_action (torch.Tensor): The previous action. Shape: (batch_size, action_dim)
            hidden (torch.Tensor): The previous hidden state. Shape: (batch_size, hidden_dim)

        Returns:
            tuple: The predicted next latent state, hidden state, reward, continue signal, and value.
        """
        batch_size = prev_obs.size(0)
        zt_gru_logits, zt_gru, ht = self.rssm.calculate_zt_gru_from_prev_obs(prev_obs, prev_action, hidden)
        stoch_sample = F.gumbel_softmax(zt_gru_logits, tau=1.0, hard=True).view(batch_size, -1)
        reward_cont_value_input = torch.cat([stoch_sample, ht], dim=-1)
        reward = self.reward_predictor(reward_cont_value_input)
        cont = self.continue_predictor(reward_cont_value_input)
        value = self.value_predictor(reward_cont_value_input)

        return zt_gru, ht, reward, cont, value

Experience = namedtuple('Experience', ('obs', 'action', 'reward', 'done', 'next_obs'))
class Memory:
    """
    Replay buffer for storing and sampling experiences.
    """
    def __init__(self, config):
        """
        Initializes the Memory buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.capacity = config.memory_capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves an experience to the memory buffer.

        Args:
            *args: Arguments for the Experience namedtuple (obs, action, reward, done, next_obs).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Experience: A namedtuple containing batches of observations, actions, rewards, dones, and next observations.
        """
        batch = random.sample(self.memory, batch_size)
        return Experience(*zip(*batch))

    def __len__(self):
        """
        Returns the current size of the memory buffer.
        """
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.actor_input_latent_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc_mean = nn.Linear(512, config.world_model_action_dim)
        self.fc_std = nn.Linear(512, config.world_model_action_dim)
        
        # Initialize all layers
        self.fc.apply(xavier_init)
        xavier_init(self.fc_mean)
        xavier_init(self.fc_std)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mean = self.fc_mean(x)
        std = 0.1 + 0.9 * F.softplus(self.fc_std(x))
        return mean, std

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.critic_input_latent_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config.critic_num_bins)
        )
        self.fc.apply(xavier_init)

    def forward(self, x):
        return F.relu(self.fc(x))