import MalmoPython
import os
import sys
import time
import json
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MARLEnvironment:
    def __init__(self, mission_xml_path="missions/handoff_arena.xml"):
        self.mission_xml_path = mission_xml_path
        self._setup_malmo_path()
        
        self.actions = {
            0: "move 0", 1: "move 1", 2: "move -1",
            3: "turn 0.5", 4: "turn -0.5", 5: "discardCurrentItem" 
        }
        
        self.agent_hosts = [None, None]

    def _setup_malmo_path(self):
        project_root = os.getcwd()
        os.environ['MALMO_XSD_PATH'] = os.path.join(project_root, 'Malmo', 'Schemas')

    def _load_mission_xml(self):
        if not os.path.exists(self.mission_xml_path):
            raise FileNotFoundError(f"Mission file not found: {self.mission_xml_path}")
        with open(self.mission_xml_path, 'r') as f:
            return f.read()

    def _start_agent_mission(self, role_index, mission_spec, experimentID):
        # 1. Create INTERNAL ClientPool (Robust Logic)
        pool = MalmoPython.ClientPool()
        pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
        pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

        host = self.agent_hosts[role_index]
        mission_record = MalmoPython.MissionRecordSpec()
        
        # Stagger: Agent 1 waits for Agent 0
        if role_index == 1:
            time.sleep(5)
            
        max_retries = 5
        for retry in range(max_retries):
            try:
                host.startMission(mission_spec, pool, mission_record, role_index, experimentID)
                logging.info(f"Role {role_index} connected (Attempt {retry+1}).")
                return
            except RuntimeError as e:
                logging.warning(f"Role {role_index} retry {retry+1} failed: {e}")
                time.sleep(2)
        
        logging.error(f"Role {role_index} GIVING UP.")
        raise RuntimeError(f"Could not connect Role {role_index}")

    def reset(self, custom_xml=None):
        # 1. Create FRESH AgentHosts
        self.agent_hosts = [MalmoPython.AgentHost(), MalmoPython.AgentHost()]

        # 2. Determine XML
        if custom_xml:
            xml_content = custom_xml
            logging.info("Loading CUSTOM Randomized Level...")
        else:
            xml_content = self._load_mission_xml()

        try:
            my_mission = MalmoPython.MissionSpec(xml_content, True)
        except RuntimeError as e:
            logging.error(f"XML Error: {e}")
            raise

        experimentID = str(time.time())
        logging.info(f"Resetting Environment (ID: {experimentID})...")

        # 3. Launch Connection Threads
        threads = []
        for i in range(2):
            # FIX: Removed 'pool' from args to match function definition
            t = threading.Thread(target=self._start_agent_mission, args=(i, my_mission, experimentID))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        # 4. Wait for Start
        logging.info("Waiting for mission to load...")
        has_begun = False
        timeout = 0
        while not has_begun and timeout < 60:
            try:
                state0 = self.agent_hosts[0].getWorldState()
                state1 = self.agent_hosts[1].getWorldState()
                if state0.has_mission_begun and state1.has_mission_begun:
                    has_begun = True
            except:
                pass
            if not has_begun:
                time.sleep(0.5)
                timeout += 0.5
        
        if not has_begun:
            raise RuntimeError("Mission Start Timeout")

        logging.info("Mission Started! Cleaning up...")
        try:
            self.agent_hosts[0].sendCommand("chat /kill @e[type=item]")
            self.agent_hosts[1].sendCommand("chat /clear")
        except:
            logging.warning("Could not send cleanup commands")
            
        time.sleep(1)
        return self._get_obs()
    
    def step(self, action_indices):
        # 1. Send Commands
        for i, action_idx in enumerate(action_indices):
            try:
                cmd = self.actions.get(action_idx, "move 0")
                self.agent_hosts[i].sendCommand(cmd)
            except RuntimeError as e:
                logging.error(f"Failed to send command to Agent {i}: {e}")

        # 2. Wait (The "Tick")
        time.sleep(0.5)

        # 3. Get New State
        obs = self._get_obs()
        
        # 4. Calculate Rewards & Done
        rewards = [-0.1, -0.1]
        done = False
        
        if self._check_if_collector_has_diamond(obs[1]):
            print("\n>>> DIAMOND HANDOFF SUCCESS! REWARD +100 <<<")
            rewards = [100.0, 100.0]
            done = True
        
        # Check if mission is still running
        if not self.agent_hosts[0].getWorldState().is_mission_running:
            done = True

        return obs, rewards, done, {}

    def _get_obs(self):
        observations = [None, None]
        for i in range(2):
            try:
                world_state = self.agent_hosts[i].getWorldState()
                if world_state.number_of_observations_since_last_state > 0:
                    msg = world_state.observations[-1].text
                    observations[i] = json.loads(msg)
            except Exception as e:
                logging.warning(f"JSON Error Agent {i}: {e}")
        return observations

    def _check_if_collector_has_diamond(self, obs_json):
        if obs_json is None: return False
        for i in range(40):
            key = f"InventorySlot_{i}_item"
            if key in obs_json and obs_json[key] == 'diamond':
                return True
        return False