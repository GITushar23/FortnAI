import torch
import numpy as np
import gymnasium as gym
import time
import os
from fortnite_env import FortniteEnv, FrameStackEnv

def run_env_and_save_data(env, num_steps, save_dir, policy=None):
    """
    Runs the given environment for a specified number of steps, collecting
    observations, actions, rewards, and done flags, and saves these as
    separate PyTorch tensors in chunks of 100, with a suffix.

    Args:
        env: The Gymnasium environment to run.
        num_steps: The number of steps to run the environment for.
        save_dir: The directory where the tensors will be saved.
        policy: An optional function that takes the current observation and
                returns an action. If None, random actions are used.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    observations = []
    actions = []
    rewards = []
    dones = []
    obs, _ = env.reset()
    time_now = time.time()
    chunk_count = 101
    for step in range(num_steps):
        print(f"time in {step}: {time.time()-time_now}")
        time_now = time.time()
        if policy:
            action = policy(obs)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        observations.append(obs)
        action_tensor = torch.tensor([
            action['fire'],
            action['look_left_or_right_or_up_or_down'][0],
            action['look_left_or_right_or_up_or_down'][1],
            action['move'][0],
            action['move'][1],
            action['move'][2],
            action['move'][3],
            action['move'][4],
            action['move'][5],
        ])
        actions.append(action_tensor)
        rewards.append(reward)
        dones.append(done)

        if done:
            break
        obs = next_obs
        if step % 100 == 0:
            print(f"Step: {step}/{num_steps}")

        if (step + 1) % 100 == 0 or step == num_steps - 1 or done:
            print(f"Saving data at step: {step + 1}")
            obs_tensor = torch.tensor(np.array(observations))
            act_tensor = torch.stack(actions)
            rew_tensor = torch.tensor(rewards)
            done_tensor = torch.tensor(dones)

            chunk_count += 1
            obs_path = os.path.join(save_dir, f'observations_{chunk_count}.pt')
            act_path = os.path.join(save_dir, f'actions_{chunk_count}.pt')
            rew_path = os.path.join(save_dir, f'rewards_{chunk_count}.pt')
            done_path = os.path.join(save_dir, f'dones_{chunk_count}.pt')

            torch.save(obs_tensor, obs_path)
            torch.save(act_tensor, act_path)
            torch.save(rew_tensor, rew_path)
            torch.save(done_tensor, done_path)

            observations = []
            actions = []
            rewards = []
            dones = []
    print(f"Saved environment data to {save_dir}")

def random_policy(obs):
    return {"fire" : np.random.uniform(-1, 1),
            "look_left_or_right_or_up_or_down" : (np.random.uniform(-1, 1),np.random.uniform(-0.005,0.005)),
            "move": tuple(np.random.uniform(-1, 1) for i in range(6))
            }

env = FrameStackEnv(FortniteEnv(),1)
run_env_and_save_data(env, 100000, 'world_model_data', random_policy)

# import numpy as np
# import torch
# from collections import deque
# import os
# import json
# from typing import Callable, Dict, List, Optional
# import gymnasium as gym
# import time
# from fortnite_env import FortniteEnv, FrameStackEnv

# import numpy as np
# import torch
# from collections import deque
# import os
# import json
# from typing import Callable, Dict, List, Optional
# import gymnasium as gym
# import sys
# # Add the path for import
# sys.path.append(r"C:\Users\Tushar\Projects\ML\DSG\RL_MINE\dreamer\Cosmos-Tokenizer")

# # Import the required class
# from cosmos_tokenizer.image_lib import ImageTokenizer

# # Remove the added path to return to original state
# sys.path.remove(r"C:\Users\Tushar\Projects\ML\DSG\RL_MINE\dreamer\Cosmos-Tokenizer")

# class TokenizedWorldModelExplorer:
#     def __init__(self, encoder_checkpoint: str, world_model: torch.nn.Module):
#         self.encoder = ImageTokenizer(checkpoint_enc=encoder_checkpoint)
#         self.world_model = world_model
#         self.curiosity_buffer = deque(maxlen=1000)
        
#     def preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
#         """Convert observation to tokenized representation"""
#         # Convert numpy array to torch tensor and add batch dimension if needed
#         if isinstance(obs, np.ndarray):
#             obs = torch.from_numpy(obs).unsqueeze(0)
        
#         # Ensure correct shape and type
#         obs = obs.to(torch.float32)
#         if obs.shape[1] == 1:  # If grayscale
#             obs = obs.repeat(1, 3, 1, 1)
            
#         # Move to GPU if available
#         device = next(self.world_model.parameters()).device
#         obs = obs.to(device).to(torch.bfloat16)
        
#         # Get tokenized representation
#         _, obs_enc_latent = self.encoder.encode(obs)
#         return obs_enc_latent
    
#     def calculate_curiosity(self, obs: np.ndarray) -> float:
#         """Calculate curiosity reward using tokenized representation"""
#         with torch.no_grad():
#             # Get tokenized representation
#             obs_tokens = self.preprocess_observation(obs)
            
#             # Get world model prediction
#             _, _, pred_tokens = self.world_model(obs_tokens)
            
#             # Calculate prediction error in token space
#             error = torch.mean((obs_tokens - pred_tokens)**2).item()
            
#             # Store tokenized representation for novelty calculation
#             self.curiosity_buffer.append(obs_tokens.cpu().numpy())
            
#             # Calculate novelty bonus based on token-space distances
#             if len(self.curiosity_buffer) > 0:
#                 distances = [np.linalg.norm(obs_tokens.cpu().numpy() - buf_tokens) 
#                            for buf_tokens in self.curiosity_buffer]
#                 novelty = 1.0 / (1.0 + np.mean(distances))
#             else:
#                 novelty = 1.0
                
#             return error * novelty

# def run_env_and_save_data(env, num_steps: int, save_path: str, 
#                          exploration_type: str = "curiosity",
#                          policy: Optional[Callable] = None,
#                          world_model: Optional[torch.nn.Module] = None,
#                          encoder_checkpoint: Optional[str] = None):
#     """
#     Collects environment data with different exploration strategies
#     optimized for sparse reward scenarios
    
#     Args:
#         env: Fortnite-style environment with sparse rewards
#         num_steps: Total steps to collect
#         save_path: Directory to save collected data
#         exploration_type: One of ["random", "ou_noise", "curiosity", "population"]
#         policy: Existing policy network (for guided exploration)
#         world_model: World model for curiosity-driven exploration
#         encoder_checkpoint: Path to Cosmos tokenizer checkpoint
#     """
#     # Initialize buffers
#     obs_buffer: List[np.ndarray] = []
#     action_buffer: List[np.ndarray] = []
#     reward_buffer: List[float] = []
#     done_buffer: List[bool] = []
#     info_buffer: List[Dict] = []
    
#     # Initialize exploration modules
#     if exploration_type == "ou_noise":
#         explorer = OUNoise(action_dim=env.action_space.shape[0])
#     elif exploration_type == "curiosity":
#         if world_model is None or encoder_checkpoint is None:
#             raise ValueError("World model and encoder checkpoint required for curiosity exploration")
#         explorer = TokenizedWorldModelExplorer(encoder_checkpoint, world_model)
#         world_model.eval()
#     elif exploration_type == "population":
#         explorer = [OUNoise(action_dim=env.action_space.shape[0]) for _ in range(5)]

#     # Tracking variables
#     sparse_reward_counter = 0
#     episode_data = {
#         "states": [],
#         "actions": [],
#         "rewards": [],
#         "intrinsic_rewards": []
#     }

#     obs, _ = env.reset()
#     total_steps = 0
    
#     while total_steps < num_steps:
#         # Select action based on exploration strategy
#         if exploration_type == "random":
#             action = env.action_space.sample()
#         elif exploration_type == "ou_noise":
#             base_action = policy(obs) if policy else np.zeros(env.action_space.shape)
#             noise = explorer.sample()
#             action = np.clip(base_action + noise, -1, 1)
#         elif exploration_type == "curiosity":
#             intrinsic_reward = explorer.calculate_curiosity(obs)
#             action = _guided_exploration(obs, intrinsic_reward, policy, env)
#         elif exploration_type == "population":
#             current_explorer = np.random.choice(explorer)
#             action = current_explorer.sample()
            
#         # Environment step
#         next_obs, reward, done, truncated, info = env.step(action)
#         total_steps += 1
        
#         # Handle sparse rewards
#         is_sparse_reward = reward != 0
#         if is_sparse_reward:
#             sparse_reward_counter += 1
#             priority = 2.0
#         else:
#             priority = 0.5
            
#         # Store transition
#         _store_transition(obs, action, reward, done, info, priority, episode_data)
#         obs = next_obs
        
#         if done or truncated:
#             _save_episode_data(save_path, episode_data, exploration_type)
#             obs, _ = env.reset()
#             episode_data = {"states": [], "actions": [], "rewards": [], "intrinsic_rewards": []}
            
#     # Save final data
#     _save_dataset(save_path, obs_buffer, action_buffer, reward_buffer, done_buffer, info_buffer)
#     if exploration_type == "curiosity":
#         _save_exploration_metrics(save_path, sparse_reward_counter, total_steps, explorer.curiosity_buffer)


# class OUNoise:
#     """Ornstein-Uhlenbeck process for exploration noise"""
#     def __init__(self, action_dim: int, mu: float = 0, theta: float = 0.15, sigma: float = 0.2):
#         self.mu = mu * np.ones(action_dim)
#         self.theta = theta
#         self.sigma = sigma
#         self.state = np.copy(self.mu)
        
#     def reset(self):
#         self.state = np.copy(self.mu)
        
#     def sample(self) -> np.ndarray:
#         dx = self.theta * (self.mu - self.state)
#         dx += self.sigma * np.random.randn(len(self.state))
#         self.state += dx
#         return self.state

# def _calculate_curiosity(obs: np.ndarray, world_model: torch.nn.Module, buffer: deque) -> float:
#     """Compute intrinsic reward using world model prediction error"""
#     with torch.no_grad():
#         obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
#         _, _, pred_obs = world_model(obs_tensor)
#         error = torch.mean((obs_tensor - pred_obs)**2).item()
    
#     # Novelty bonus
#     novelty = 1.0 / (1.0 + np.mean([np.linalg.norm(obs - buf_obs) for buf_obs in buffer]))
#     buffer.append(obs)
#     return error * novelty

# def _guided_exploration(obs: np.ndarray, intrinsic_reward: float, 
#                        policy: Optional[Callable], env) -> np.ndarray:
#     """Hybrid exploration using policy and intrinsic rewards"""
#     if policy and np.random.rand() < 0.7:  # 70% policy guidance
#         base_action = policy(obs)
#         noise_scale = 0.3 * (1 - intrinsic_reward)
#         return np.clip(
#             base_action + noise_scale * np.random.randn(*base_action.shape),
#             env.action_space.low,
#             env.action_space.high
#         )
#     else:
#         return env.action_space.sample()


# def _store_transition(obs: np.ndarray, action: np.ndarray, reward: float, done: bool, info: Dict, priority: float, episode_data: Dict):
#     """Prioritized experience storage for sparse rewards"""
#     episode_data["states"].append(obs)
#     episode_data["actions"].append(action)
#     episode_data["rewards"].append(reward)
#     episode_data["intrinsic_rewards"].append(info.get("intrinsic_reward", 0))

# def _save_episode_data(path: str, data: Dict, exploration_type: str):
#     """Save episode data with exploration-specific metrics"""
#     os.makedirs(path, exist_ok=True)
#     episode_id = len(os.listdir(path))
#     filename = f"{path}/{exploration_type}_episode_{episode_id}.npz"
#     np.savez_compressed(
#         filename,
#         states=np.array(data["states"]),
#         actions=np.array(data["actions"]),
#         rewards=np.array(data["rewards"]),
#         intrinsic_rewards=np.array(data["intrinsic_rewards"])
#     )

# def _save_dataset(path: str, *buffers):
#     """Save final dataset in RL-friendly format"""
#     dataset = {
#         "observations": np.array(buffers[0]),
#         "actions": np.array(buffers[1]),
#         "rewards": np.array(buffers[2]),
#         "dones": np.array(buffers[3]),
#         "infos": buffers[4]
#     }
#     np.savez_compressed(f"{path}/full_dataset.npz", **dataset)

# def _save_exploration_metrics(path: str, sparse_rewards: int, total_steps: int, curiosity_buffer: deque):
#     """Log exploration effectiveness metrics"""
#     metrics = {
#         "sparse_reward_rate": sparse_rewards / total_steps,
#         "states_visited": len(curiosity_buffer),
#         "exploration_coverage": len(curiosity_buffer) / total_steps
#     }
#     with open(f"{path}/exploration_metrics.json", "w") as f:
#         json.dump(metrics, f)

# # Example usage
# env = FrameStackEnv(FortniteEnv(), 1)
# run_env_and_save_data(env, 10000, 'world_model_data', exploration_type="curiosity")