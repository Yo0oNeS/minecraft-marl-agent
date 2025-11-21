import torch
import random
import time
import numpy as np
from marl_env import MARLEnvironment
from model import PPOAgent, preprocess_obs

# --- CONFIGURATION ---
MODEL_PATH_MINER = "miner_ppo.pth"
MODEL_PATH_COLLECTOR = "collector_ppo.pth"
TEST_EPISODES = 10

def get_randomized_xml():
    """
    Generates the Arena XML with random spawn points.
    """
    # Randomize Collector Position
    col_x = random.choice([-3.5, -2.5, 0.5, 2.5, 3.5]) 
    col_z = random.choice([-3.5, -1.5, 0, 1.5, 3.5])
    col_yaw = random.choice([0, 90, 180, 270])

    print(f"--- GENERATING LEVEL: Collector at X={col_x}, Z={col_z}, Yaw={col_yaw} ---")

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About><Summary>Generalization Test</Summary></About>
  <ServerSection>
    <ServerInitialConditions>
        <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
      <DrawingDecorator>
        <DrawCuboid x1="-5" y1="227" z1="-5" x2="5" y2="232" z2="5" type="air"/>
        <DrawCuboid x1="-4" y1="227" z1="-4" x2="4" y2="229" z2="-4" type="bedrock"/>
        <DrawCuboid x1="-4" y1="227" z1="4" x2="4" y2="229" z2="4" type="bedrock"/>
        <DrawCuboid x1="-4" y1="227" z1="-4" x2="-4" y2="229" z2="4" type="bedrock"/>
        <DrawCuboid x1="4" y1="227" z1="-4" x2="4" y2="229" z2="4" type="bedrock"/>
      </DrawingDecorator>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <!-- MINER (Fixed Target) -->
  <AgentSection mode="Survival">
    <Name>MinerBot</Name>
    <AgentStart>
      <Placement x="-1.5" y="227" z="0" yaw="-90"/>
      <Inventory><InventoryItem slot="0" type="diamond"/></Inventory>
    </AgentStart>
    <AgentHandlers>
      <ContinuousMovementCommands/>
      <InventoryCommands/>
      <ChatCommands/>
      <ObservationFromFullStats/>
      <ObservationFromFullInventory/>
      <ObservationFromNearbyEntities>
        <Range name="entities" xrange="10" yrange="2" zrange="10"/>
      </ObservationFromNearbyEntities>
    </AgentHandlers>
  </AgentSection>

  <!-- COLLECTOR (RANDOMIZED) -->
  <AgentSection mode="Survival">
    <Name>CollectorBot</Name>
    <AgentStart>
      <Placement x="{col_x}" y="227" z="{col_z}" yaw="{col_yaw}"/>
    </AgentStart>
    <AgentHandlers>
      <ContinuousMovementCommands/>
      <InventoryCommands/>
      <ChatCommands/>
      <ObservationFromFullStats/>
      <ObservationFromFullInventory/>
      <ObservationFromNearbyEntities>
         <Range name="entities" xrange="10" yrange="2" zrange="10"/>
      </ObservationFromNearbyEntities>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

def run_test():
    print("Initializing Environment...")
    env = MARLEnvironment() # This now uses the robust class logic
    
    print("Loading Trained Brains...")
    miner_brain = PPOAgent(4, 6)
    miner_brain.load_state_dict(torch.load(MODEL_PATH_MINER))
    miner_brain.eval()

    collector_brain = PPOAgent(4, 6)
    collector_brain.load_state_dict(torch.load(MODEL_PATH_COLLECTOR))
    collector_brain.eval()
    
    success_count = 0

    for episode in range(1, TEST_EPISODES + 1):
        print(f"\n=== TEST EPISODE {episode} ===")
        
        # 1. Generate Random Map
        random_xml = get_randomized_xml()
        
        # 2. Reset Env with Custom XML (Uses Robust Logic)
        try:
            obs_json = env.reset(custom_xml=random_xml)
        except RuntimeError as e:
            print(f"Critical Reset Error: {e}")
            continue

        # 3. Play Loop
        done = False
        step = 0
        while not done and step < 50:
            # Preprocess
            state_miner = preprocess_obs(obs_json[0])
            state_collector = preprocess_obs(obs_json[1])
            
            # Get Actions
            with torch.no_grad():
                act0, _ = miner_brain.get_action(state_miner)
                act1, _ = collector_brain.get_action(state_collector)
            
            # Step
            obs_json, rewards, done, _ = env.step([act0, act1])
            
            if rewards[1] > 50: # Check for win reward
                print(">>> SUCCESS! Diamond Collected! <<<")
                success_count += 1
                done = True
            
            step += 1
            
        if not done:
            print("FAILED: Time limit.")
            
    print(f"\nTEST COMPLETE: {success_count}/{TEST_EPISODES} Success Rate.")

if __name__ == "__main__":
    run_test()