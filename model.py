import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# --- 1. THE PREPROCESSOR (JSON -> Tensor) ---
def preprocess_obs(obs_json):
    """
    Converts JSON into a Relative Vector (Virtual Eyes).
    Features: [Delta_X, Delta_Z, Yaw, HasDiamond] (Size: 4)
    """
    if obs_json is None:
        return np.zeros(4, dtype=np.float32)

    # 1. Get My Position
    my_x = obs_json.get('XPos', 0)
    my_z = obs_json.get('ZPos', 0)
    yaw = obs_json.get('Yaw', 0) / 180.0 # Normalize

    # 2. Find Target Position (The Diamond)
    # Default: If I can't see it, assume it is right on top of me (Delta 0)
    target_x = my_x
    target_z = my_z
    
    # Malmo sends a list of "nearby_entities"
    if 'nearby_entities' in obs_json:
        for entity in obs_json['nearby_entities']:
            if entity['name'] == 'diamond': # Found the target!
                target_x = entity['x']
                target_z = entity['z']
                break
                
    # 3. Calculate DELTA (The Vector)
    delta_x = target_x - my_x
    delta_z = target_z - my_z
    
    # Clip values to keep the Neural Network happy (avoid huge numbers)
    delta_x = np.clip(delta_x, -10, 10)
    delta_z = np.clip(delta_z, -10, 10)

    # 4. Check Inventory
    has_diamond = 0.0
    inventory_size = obs_json.get('InventorySize', 40)
    for i in range(inventory_size):
        key = f"InventorySlot_{i}_item"
        if key in obs_json and obs_json[key] == 'diamond':
            has_diamond = 1.0
            break

    return np.array([delta_x, delta_z, yaw, has_diamond], dtype=np.float32)

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