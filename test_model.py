from model import PPOAgent, preprocess_obs
import torch
import json

# 1. Simulate Raw Data from Minecraft
fake_json_miner = {
    'XPos': -2.5, 'ZPos': 0.0, 'Yaw': -90.0, 
    'InventorySlot_0_item': 'diamond'
}
fake_json_collector = {
    'XPos': 2.5, 'ZPos': 0.0, 'Yaw': 90.0
    # No diamond
}

# 2. Test Preprocessor
print("Testing Preprocessor...")
vec_miner = preprocess_obs(fake_json_miner)
vec_collector = preprocess_obs(fake_json_collector)

print(f"Miner Vector: {vec_miner} (Expected: [-2.5, 0, -0.5, 1.0])")
print(f"Collector Vector: {vec_collector} (Expected: [2.5, 0, 0.5, 0.0])")

# 3. Test Neural Network
print("\nTesting PPO Brain...")
input_dim = 4  # [x, z, yaw, diamond]
output_dim = 6 # [stop, fwd, back, right, left, drop]

brain = PPOAgent(input_dim, output_dim)

# Test Actor (Getting an action)
action, log_prob = brain.get_action(vec_miner)
print(f"Miner chose Action: {action} (LogProb: {log_prob.item():.4f})")

# Test Critic (Judging the situation)
# Critic needs BOTH vectors concatenated
import numpy as np
global_state = np.concatenate([vec_miner, vec_collector]) # Size 8
value = brain.get_value(global_state)

print(f"Critic Value Estimation: {value.item():.4f}")

print("\nSUCCESS: Model is ready for training.")