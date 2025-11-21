from marl_env import MARLEnvironment
import random
import time

# 1. Initialize
env = MARLEnvironment()

# 2. Reset (Connects to clients)
print("Resetting environment...")
obs = env.reset()
print("Environment Ready!")

if obs[0] is None or obs[1] is None:
    print("Warning: Initial observations are empty.")

# 3. Run a loop (The "Game Loop")
print("Starting Step Loop (Running for 10 steps)...")
try:
    for step in range(10):
        # Pick random actions
        # Miner (0): Randomly move (1) or turn (3,4)
        act0 = random.choice([0, 1, 3, 4])
        # Collector (1): Randomly move (1) or turn (3,4)
        act1 = random.choice([0, 1, 3, 4])
        
        actions = [act0, act1]
        
        print(f"Step {step}: Actions {actions}")
        
        # Execute Step
        obs, rewards, done, info = env.step(actions)
        
        # Debug print
        if obs[0]:
            print(f" - Miner Pos: {obs[0].get('XPos', 0):.2f}, {obs[0].get('ZPos', 0):.2f}")
        
        if done:
            print("Mission ended early.")
            break
            
except KeyboardInterrupt:
    print("Stopped by user.")

print("Test Complete.")