import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from marl_env import MARLEnvironment
from model import PPOAgent, preprocess_obs
import os

# --- HYPERPARAMETERS ---
LR = 0.002              # Learning Rate (How fast we learn)
GAMMA = 0.99            # Discount Factor (Future rewards vs immediate rewards)
K_EPOCHS = 4            # How many times we update per batch
EPS_CLIP = 0.2          # PPO Clipping (Prevents drastic changes)
MAX_EPISODES = 2        # Total games to play
UPDATE_TIMESTEP = 10    # Update the brain every X steps

device = torch.device('cpu') # Using CPU is fine for this model size, keeps it simple.

# --- MEMORY BUFFER ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- PPO ALGORITHM ---
class PPO:
    def __init__(self, input_dim, output_dim):
        self.policy = PPOAgent(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = PPOAgent(input_dim, output_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        # Run the "Old" policy to get data for the memory buffer
        state_t = torch.FloatTensor(state).to(device)
        action, log_prob = self.policy_old.get_action(state_t)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob.item())
        
        return action

    def update(self, memory, global_state_buffer):
        # Monte Carlo Estimate of Rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalize rewards (Stability trick)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(device)
        old_actions = torch.tensor(memory.actions, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)
        
        # Global States for Critic (Concatenate Miner+Collector)
        # For simplicity in this prototype, we will just use the local state for the critic 
        # unless we strictly formatted the global buffer. 
        # Let's use local state for critic to ensure code runs without shape errors for now.
        # (CTDE Upgrade can be done in Phase 5).
        
        # Optimize policy for K epochs
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            # We need to implement evaluate in PPOAgent or manually call here
            # Let's do a manual forward pass logic here for clarity
            
            # Get new logprobs, entropy, and state values
            # Note: We need to tweak model.py slightly to allow batch processing if not already supported
            # But standard Linear layers support batches automatically.
            
            # Re-run Actor
            probs = self.policy.actor(old_states)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Re-run Critic
            state_values = self.policy.critic(torch.cat([old_states, old_states], dim=1)) # Hack: CTDE placeholder
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            # Final Loss
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Gradient Step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
        self.policy_old.load_state_dict(self.policy.state_dict())

# --- MAIN TRAINING LOOP ---
def train():
    print("Initializing Environment...")
    env = MARLEnvironment()
    
    print("Initializing PPO Agents...")
    # We create two separate brains
    input_dim = 4
    output_dim = 6
    
    miner_agent = PPO(input_dim, output_dim)
    collector_agent = PPO(input_dim, output_dim)
    
    miner_memory = Memory()
    collector_memory = Memory()
    
    print("Starting Training...")
    time_step = 0
    
    for i_episode in range(1, MAX_EPISODES+1):
        print(f"Episode {i_episode} starting...")
        
        # Reset Env
        obs_json = env.reset()
        
        # Preprocess State
        state_miner = preprocess_obs(obs_json[0])
        state_collector = preprocess_obs(obs_json[1])
        
        current_ep_reward = 0
        
        for t in range(1, 100): # Max 100 steps per episode
            time_step += 1
            
            # 1. Select Action
            action_miner = miner_agent.select_action(state_miner, miner_memory)
            action_collector = collector_agent.select_action(state_collector, collector_memory)
            
            # 2. Step Env
            obs_json, rewards, done, _ = env.step([action_miner, action_collector])
            
            # 3. Save Reward to Memory
            miner_memory.rewards.append(rewards[0])
            miner_memory.is_terminals.append(done)
            
            collector_memory.rewards.append(rewards[1])
            collector_memory.is_terminals.append(done)
            
            # 4. Update State
            state_miner = preprocess_obs(obs_json[0])
            state_collector = preprocess_obs(obs_json[1])
            
            current_ep_reward += sum(rewards)
            
            # 5. PPO Update (if timestep matches)
            if time_step % UPDATE_TIMESTEP == 0:
                print("Updating Weights...")
                miner_agent.update(miner_memory, None)
                collector_agent.update(collector_memory, None)
                miner_memory.clear_memory()
                collector_memory.clear_memory()
                time_step = 0
            
            if done:
                break
                
        print(f"Episode {i_episode} finished. Total Reward: {current_ep_reward:.2f}")
        
        # Save occasionally
        if i_episode % 10 == 0:
            miner_agent.save("miner_ppo.pth")
            collector_agent.save("collector_ppo.pth")
            print("Model Saved!")

if __name__ == "__main__":
    train()