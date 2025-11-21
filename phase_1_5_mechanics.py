import MalmoPython
import os
import sys
import time
import json
import threading

# --- HELPER: LOAD XML ---
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

# --- 2. LOAD MISSION ---
print("Loading v2 XML...")
xml_content = load_mission_xml("handoff_v2.xml")

# Validate Global Mission Object
try:
    my_mission = MalmoPython.MissionSpec(xml_content, True)
except RuntimeError as e:
    print("XML ERROR:", e)
    exit(1)

# --- 3. WORKER FUNCTION ---
def run_agent(role, client_pool, experimentID):
    agent_host = MalmoPython.AgentHost()
    
    # STAGGER
    if role == 1: 
        print("[Role 1] Pausing 2s to allow Server to initialize...")
        time.sleep(2)

    # CONNECT
    print(f"[Role {role}] Connecting...")
    my_mission_record = MalmoPython.MissionRecordSpec()
    max_retries = 5
    connected = False
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, client_pool, my_mission_record, role, experimentID)
            connected = True
            print(f"[Role {role}] Connected!")
            break
        except RuntimeError as e:
            time.sleep(2)

    if not connected:
        print(f"[Role {role}] FAILED to connect.")
        return

    # WAIT FOR START
    print(f"[Role {role}] Waiting for spawn...", end=' ')
    while not agent_host.getWorldState().has_mission_begun:
        time.sleep(0.1)
    print("GO!")

    # --- MECHANICS ---
    
    # === ROLE 0: MINER ===
    if role == 0:
        print("[Miner] Looking down...")
        agent_host.sendCommand("pitch 45") 
        time.sleep(1)
        
        print("[Miner] Dropping Diamond...")
        agent_host.sendCommand("discardCurrentItem") 
        time.sleep(0.5) 
        
        print("[Miner] Backing off!")
        agent_host.sendCommand("move -0.5") 
        time.sleep(1.0) 
        agent_host.sendCommand("move 0")    
        
        agent_host.sendCommand("pitch -45") 
        time.sleep(5) # Wait longer to keep server alive for Collector

    # === ROLE 1: COLLECTOR ===
    elif role == 1:
        print("[Collector] Waiting for drop...")
        time.sleep(2.5) 
        
        print("[Collector] Approaching...")
        agent_host.sendCommand("move 0.5") 
        time.sleep(2.5) 
        agent_host.sendCommand("move 0")   
        
        # --- ROBUST VERIFICATION LOOP ---
        print("[Collector] Checking Inventory (Scanning for 5 seconds)...")
        
        diamond_found = False
        
        # Check 10 times (every 0.5 seconds)
        for check in range(10):
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                obs_json = json.loads(msg)
                
                # Scan slots
                for i in range(40):
                    key = f"InventorySlot_{i}_item"
                    if key in obs_json:
                        if obs_json[key] == 'diamond':
                            diamond_found = True
                            print(f"\n[Check {check+1}] Found diamond in Slot {i}!")
                            break
            
            if diamond_found:
                break
                
            # Wait for next packet
            sys.stdout.write(".")
            time.sleep(0.5)

        if diamond_found:
            print("\n*********************************")
            print(" SUCCESS: DIAMOND ACQUIRED! :D")
            print("*********************************")
        else:
            print("\nFAILURE: Diamond never arrived in inventory data. :(")

    print(f"[Role {role}] Finished.")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    client_pool = MalmoPython.ClientPool()
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

    # Unique ID ensures they join the same session
    experimentID = str(time.time())

    t1 = threading.Thread(target=run_agent, args=(0, client_pool, experimentID))
    t2 = threading.Thread(target=run_agent, args=(1, client_pool, experimentID))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("Test Complete.")