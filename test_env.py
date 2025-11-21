from marl_env import MARLEnvironment
import time

def run_winning_sequence():
    env = MARLEnvironment()
    print("Resetting...")
    env.reset()
    
    print("Executing Scripted Win...")
    
    # We will construct a sequence of actions to force a win.
    # Actions Mapping: 0:Stop, 1:Fwd, 2:Back, 3:Right, 4:Left, 5:Drop
    
    # SEQUENCE:
    # 1. Miner (0) looks down (We can't pitch via simple actions yet, assumes pre-set or ignores pitch)
    #    Wait! Our current action list doesn't have "Pitch". 
    #    In RL, we usually simplify by removing pitch and just letting them drop forward.
    #    Let's see if "discardCurrentItem" works without looking down. (It usually throws it forward).
    
    steps = []
    
    # Step 1: Miner Drops (Action 5), Collector Waits (Action 0)
    steps.append([5, 0])
    
    # Step 2: Miner Backs Up (Action 2), Collector Waits (Action 0)
    steps.append([2, 0])
    steps.append([2, 0]) # Back up more
    
    # Step 3: Collector Moves Forward (Action 1) into the diamond
    for _ in range(6): # 6 steps forward
        steps.append([0, 1])
        
    # Execute the list
    total_reward = 0
    for i, actions in enumerate(steps):
        print(f"Step {i}: Actions {actions}")
        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards)
        
        if done:
            print("Mission Done Triggered!")
            break
            
    print(f"Total Reward: {total_reward}")
    if total_reward > 50:
        print("TEST PASSED: High reward received.")
    else:
        print("TEST FAILED: Diamond not detected or not reached.")

if __name__ == "__main__":
    run_winning_sequence()