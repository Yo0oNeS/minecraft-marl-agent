import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# --- 1. THE PREPROCESSOR (JSON -> Tensor) ---
def preprocess_obs(obs_json):
    """
    Converts the raw Malmo JSON into a clean float vector.
    Features: [XPos, ZPos, Yaw, HasDiamond] (Size: 4)
    """
    if obs_json is None:
        return np.zeros(4, dtype=np.float32)

    # 1. Extract Position (Ignore Y because it's flat)
    x = obs_json.get('XPos', 0)
    z = obs_json.get('ZPos', 0)
    yaw = obs_json.get('Yaw', 0)

    # Normalize Yaw to be between -1 and 1 (roughly)
    # Minecraft yaw is 0-360 or -180 to 180. Let's simplify it to radians/pi or just simple scaling
    yaw_norm = yaw / 180.0

    # 2. Extract Inventory (Has Diamond?)
    has_diamond = 0.0
    # Scan slots like we did in the env
    # Note: This is a simplified check for the Neural Net
    inventory_size = obs_json.get('InventorySize', 40)
    for i in range(inventory_size):
        key = f"InventorySlot_{i}_item"
        if key in obs_json:
            if obs_json[key] == 'diamond':
                has_diamond = 1.0
                break

    return np.array([x, z, yaw_norm, has_diamond], dtype=np.float32)

# --- 2. THE NEURAL NETWORK ---
class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        
        # --- ACTOR (The Player) ---
        # Input: Local Observation (4 values)
        # Output: Probability of taking each action (6 actions)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1) # Output probabilities (sum to 1)
        )
        
        # --- CRITIC (The Coach) ---
        # Input: Centralized Observation (Agent A Obs + Agent B Obs) = 8 values
        # Output: Value estimation (How good is this state?)
        self.critic = nn.Sequential(
            nn.Linear(input_dim * 2, 64), # Takes input from BOTH agents
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Outputs a single score
        )

    def get_action(self, obs):
        """
        Forward pass for the Actor.
        """
        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
            
        probs = self.actor(obs)
        
        # Create a distribution to sample from
        dist = torch.distributions.Categorical(probs)
        
        # Sample an action (e.g., 3)
        action = dist.sample()
        
        # Return action and the log_prob (needed for PPO math later)
        return action.item(), dist.log_prob(action)

    def get_value(self, global_obs):
        """
        Forward pass for the Critic.
        global_obs should be [obs_agent_1, obs_agent_2] concatenated.
        """
        if not isinstance(global_obs, torch.Tensor):
            global_obs = torch.tensor(global_obs, dtype=torch.float32)
            
        return self.critic(global_obs)