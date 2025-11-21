import MalmoPython
import os
import sys
import time
import threading

# --- HELPER FUNCTION TO LOAD MISSION XML FROM FILE ---
def load_mission_xml(filename):
    mission_path = os.path.join(os.getcwd(), 'missions', filename)
    if not os.path.exists(mission_path):
        print(f"Error: Cannot find mission file at {mission_path}")
        exit(1)
    with open(mission_path, 'r') as f:
        return f.read()

# --- 1. PATH SETUP ---
project_root = os.getcwd()
os.environ['MALMO_XSD_PATH'] = os.path.join(project_root, 'Malmo', 'Schemas')

# --- 2. LOAD MISSION XML ---
print("Loading mission XML...")
xml_content = load_mission_xml("handoff_v1.xml")

# Validate and Create the Mission Object globally so threads can access it
try:
    my_mission = MalmoPython.MissionSpec(xml_content, True)
except RuntimeError as e:
    print("XML ERROR:", e)
    exit(1)

# --- 3. THREADED WORKER FUNCTION ---
def run_agent(role_index, client_pool, experimentID):
    agent_host = MalmoPython.AgentHost()
    
    if role_index == 1:
        print("[Role 1] Pausing 2s to allow Server (Role 0) to initialize...")
        time.sleep(2)

    print(f"[Role {role_index}] Connecting...")
    
    # Retry Loop
    my_mission_record = MalmoPython.MissionRecordSpec()
    max_retries = 5
    connected = False

    for retry in range(max_retries):
        try:
            agent_host.startMission(
                my_mission, 
                client_pool, 
                my_mission_record, 
                role_index, 
                experimentID
            )
            connected = True
            print(f"[Role {role_index}] Connected to Client!")
            break
        except RuntimeError as e:
            print(f"[Role {role_index}] Retry {retry+1}/{max_retries}: {e}")
            time.sleep(2)

    if not connected:
        print(f"[Role {role_index}] FAILED to connect after retries.")
        return

    # Wait for Mission Start
    print(f"[Role {role_index}] Waiting for mission start...", end=' ')
    has_begun = False
    while not has_begun:
        world_state = agent_host.getWorldState()
        if world_state.has_mission_begun:
            has_begun = True
            print(f"GO!")
        else:
            time.sleep(0.1)
        
        # Check for quit errors
        if len(world_state.errors) > 0:
             print(f"[Role {role_index}] Error: {world_state.errors[0].text}")

    # Control Loop
    for i in range(15):
        if role_index == 0:
            agent_host.sendCommand("turn 0.5")
        else:
            agent_host.sendCommand("turn -0.5")
        time.sleep(0.5)

    print(f"[Role {role_index}] Finished.")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    
    client_pool = MalmoPython.ClientPool()
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

    experimentID = str(time.time())

    t1 = threading.Thread(target=run_agent, args=(0, client_pool, experimentID))
    t2 = threading.Thread(target=run_agent, args=(1, client_pool, experimentID))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("Multi-Agent Test Complete.")