import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

from fortnite_env import FortniteEnv
from pushbullet import Pushbullet
from dotenv import load_dotenv
load_dotenv()
import os

access_token = os.getenv("ACCESS_TOKEN")
title = "GAME STOPPED GAME STOPPED GAME STOPPED"
body = "GAME STOPPED GAME STOPPED GAME STOPPED"

# Hyperparameters
LEARNING_RATE = 1e-6
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.1
MAX_GRAD_NORM = 0.5
NUM_MINI_BATCHES = 1
PPO_EPOCHS = 4
BATCH_SIZE = 256
STEPS_PER_EPISODE = 2048
NO_OF_EPISODES = 1000
FRAME_STACK = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrameStackEnv:
    def __init__(self, env, num_stack):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

    def reset(self):
        obs, _ = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self.get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward == 696969:
            Pushbullet(access_token).push_note(title, body)
            print("Ending environment")
            exit()
        self.frames.append(obs)
        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

class PerceptionComponent(nn.Module):
    def __init__(self, input_shape):
        super(PerceptionComponent, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0] * FRAME_STACK, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(1, input_shape[0] * FRAME_STACK, *input_shape[1:])
            self.feature_size = self.conv(sample_input).shape[1]
        
        self.fc = nn.Linear(self.feature_size, 512)
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class ReactivePolicy(nn.Module):
    def __init__(self, input_size, discrete_action_dims):
        super(ReactivePolicy, self).__init__()
        self.input_size = input_size
        self.discrete_action_dims = discrete_action_dims
        
        # Common layers
        self.common_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Discrete action heads
        self.discrete_action_heads = nn.ModuleDict()
        
        for action_name, action_dim in discrete_action_dims.items():
            self.discrete_action_heads[action_name] = nn.Linear(256, action_dim)
    
    def forward(self, x):
        common_features = self.common_layers(x)
        
        discrete_action_logits = {}
        
        for action_name in self.discrete_action_dims.keys():
            discrete_action_logits[action_name] = self.discrete_action_heads[action_name](common_features)
        
        return discrete_action_logits

    def sample_action(self, x):
        discrete_action_logits = self.forward(x)
        
        discrete_actions = {}
        log_probs = {}
        
        for action_name, logits in discrete_action_logits.items():
            dist = Categorical(logits=logits)
            action = dist.sample()
            discrete_actions[action_name] = action
            log_probs[action_name] = dist.log_prob(action)
        
        return discrete_actions, log_probs

    def evaluate_actions(self, x, discrete_actions):
        discrete_action_logits = self.forward(x)
        
        log_probs = {}
        entropies = {}
        
        for action_name, logits in discrete_action_logits.items():
            dist = Categorical(logits=logits)
            log_probs[action_name] = dist.log_prob(discrete_actions[action_name])
            entropies[action_name] = dist.entropy()
        
        return log_probs, entropies

class ValueFunction(nn.Module):
    def __init__(self, input_size):
        super(ValueFunction, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.fc(x)

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

    
class FortniteAgent(nn.Module):
    def __init__(self, state_dim, discrete_action_dims):
        super(FortniteAgent, self).__init__()
        self.perception = PerceptionComponent(state_dim)
        self.policy = ReactivePolicy(512, discrete_action_dims)
        self.value = ValueFunction(512)
        total_action_dim = sum([1 for v in discrete_action_dims.keys()])
        self.transition_model = TransitionModel(512, total_action_dim)
        
    def forward(self, state):
        features = self.perception(state)
        discrete_action_logits = self.policy(features)
        value = self.value(features)
        return discrete_action_logits, value
    
    def predict_next_state(self, state, discrete_actions):
        features = self.perception(state)
        # print(discrete_actions)
        # Convert discrete actions to float and concatenate
        discrete_part = torch.cat([action.float().unsqueeze(-1) for action in discrete_actions.values()], dim=-1)
        
        # Use discrete actions only
        action = discrete_part.squeeze(-1)
        # print(action.shape)
        
        return self.transition_model(features, action)
    
class PPOAgent:
    def __init__(self, state_dim, discrete_action_dims):
        self.agent = FortniteAgent(state_dim, discrete_action_dims).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=BATCH_SIZE)
        self.discrete_action_dims = discrete_action_dims
        self.discrete_action_names = list(discrete_action_dims.keys())

    def select_action(self, state):
        with torch.no_grad():
            features = self.agent.perception(state)
            discrete_actions, _ = self.agent.policy.sample_action(features)
        return discrete_actions

    def update(self):
        batch = list(self.memory)
        states, discrete_actions, rewards, next_states = map(np.array, zip(*batch))
        
        states = torch.FloatTensor(states).to(device)
        
        # Handle discrete actions
        discrete_actions_dict = {}
        for i, action_name in enumerate(self.discrete_action_names):
            discrete_actions_dict[action_name] = torch.LongTensor(discrete_actions[:, i]).to(device)

        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        with torch.no_grad():
            _, values = self.agent(states)
            _, next_values = self.agent(next_states)
            
        advantages = self.compute_gae(rewards, values, next_values).to(device)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for indices in self.get_minibatch_indices():
                mini_batch_states = states[indices]
                mini_batch_discrete_actions = {k: v[indices].unsqueeze(-1) for k, v in discrete_actions_dict.items()}
                mini_batch_advantages = advantages[indices]
                mini_batch_returns = returns[indices]
                mini_batch_values = values[indices]

                features = self.agent.perception(mini_batch_states)
                discrete_action_logits, new_values = self.agent(mini_batch_states)
                
                new_log_probs, entropies = self.agent.policy.evaluate_actions(features, mini_batch_discrete_actions)
                old_log_probs, _ = self.agent.policy.evaluate_actions(features, mini_batch_discrete_actions)

                ratio = torch.exp(sum(new_log_probs.values()) - sum(old_log_probs.values()))
                surr1 = ratio * mini_batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * mini_batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values, mini_batch_returns)

                entropy = sum(entropies.values()).mean()

                # Add transition model loss
                predicted_next_states = self.agent.predict_next_state(mini_batch_states, mini_batch_discrete_actions)
                transition_loss = nn.MSELoss()(predicted_next_states, self.agent.perception(next_states[indices]))

                loss = actor_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy + 0.1 * transition_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.memory.clear()

    def compute_gae(self, rewards, values, next_values):
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + GAMMA * next_values[step] - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def get_minibatch_indices(self):
        indices = np.arange(BATCH_SIZE)
        np.random.shuffle(indices)
        return np.array_split(indices, NUM_MINI_BATCHES)
    

def train(env, agent):
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    total_steps = 0

    for _ in range(NO_OF_EPISODES):
        for _ in range(STEPS_PER_EPISODE):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            discrete_actions = agent.select_action(state_tensor)
            # print("discretetete",discrete_actions)
            # Convert actions to numpy arrays for the environment
            discrete_actions_np = {k: v.cpu().numpy().squeeze() for k, v in discrete_actions.items()}
            
            # Use discrete actions only for the environment step
            combined_action = discrete_actions_np
            
            next_state, reward, terminated, truncated, _ = env.step(combined_action)
            
            # Flatten the actions for storage in memory
            flat_discrete_actions = np.array([v for v in discrete_actions_np.values()])
            
            agent.memory.append((state, flat_discrete_actions, reward, next_state))
            episode_reward += reward
            state = next_state

            total_steps += 1
            episode_steps += 1

            if len(agent.memory) == BATCH_SIZE:
                print("Updating agent")
                agent.update()
                agent.memory.clear()
                torch.save(agent.agent.state_dict(), f'models/model{total_steps}.pth')

            if terminated or truncated or episode_steps >= 1024:
                print(f"Episode {episode_count}, Steps: {episode_steps}, Total Reward: {episode_reward}")
                state, _ = env.reset()
                episode_count += 1
                episode_reward = 0
                episode_steps = 0

# Initialize environment and agent
env = FrameStackEnv(FortniteEnv(), FRAME_STACK)
state_dim = env.env.observation_space.shape

# Simplified action space
discrete_action_dims = {'fire': 2}    # Binary fire action (fire or not fire)

agent = PPOAgent(state_dim, discrete_action_dims)

agent.agent.load_state_dict(torch.load('models/model15360.pth'))

train(env, agent)