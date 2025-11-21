import MalmoPython
import os
import sys
import time
import json
import threading
import logging

# Configure logging to keep console clean but useful
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MARLEnvironment:
    def __init__(self, mission_xml_path="missions/handoff_v2.xml"):
        self.mission_xml_path = mission_xml_path
        self._setup_malmo_path()
        
        # --- ACTION SPACE ---
        # 0: Stop, 1: Forward, 2: Back, 3: Turn Right, 4: Turn Left, 5: Drop/Use
        self.actions = {
            0: "move 0",
            1: "move 1",
            2: "move -1",
            3: "turn 0.5",
            4: "turn -0.5",
            5: "discardCurrentItem" 
        }
        
        # Keep track of our agents
        self.agent_hosts = [MalmoPython.AgentHost(), MalmoPython.AgentHost()]
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

    def _setup_malmo_path(self):
        project_root = os.getcwd()
        os.environ['MALMO_XSD_PATH'] = os.path.join(project_root, 'Malmo', 'Schemas')

    def _load_mission_xml(self):
        if not os.path.exists(self.mission_xml_path):
            raise FileNotFoundError(f"Mission file not found: {self.mission_xml_path}")
        with open(self.mission_xml_path, 'r') as f:
            return f.read()

    def _start_agent_mission(self, role_index, mission_spec, experimentID):
        """
        Worker function to be run in a thread. Connects a specific agent.
        """
        host = self.agent_hosts[role_index]
        mission_record = MalmoPython.MissionRecordSpec()
        
        # Stagger: Agent 1 waits for Agent 0
        if role_index == 1:
            time.sleep(2)
            
        max_retries = 5
        for retry in range(max_retries):
            try:
                host.startMission(mission_spec, self.client_pool, mission_record, role_index, experimentID)
                logging.info(f"Role {role_index} connected.")
                return
            except RuntimeError as e:
                time.sleep(2)
        
        logging.error(f"Role {role_index} FAILED to connect after retries.")
        raise RuntimeError(f"Could not connect Role {role_index}")

    def reset(self):
        """
        Restarts the mission.
        Returns: Initial observations [obs_agent_0, obs_agent_1]
        """
        logging.info("Resetting Environment...")
        xml_content = self._load_mission_xml()
        my_mission = MalmoPython.MissionSpec(xml_content, True)
        experimentID = str(time.time())

        # 1. Launch Threads to Connect
        threads = []
        for i in range(2):
            t = threading.Thread(target=self._start_agent_mission, args=(i, my_mission, experimentID))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        # 2. Wait for Mission Start
        logging.info("Waiting for mission start...")
        has_begun = False
        while not has_begun:
            # Check both agents
            state0 = self.agent_hosts[0].getWorldState()
            state1 = self.agent_hosts[1].getWorldState()
            if state0.has_mission_begun and state1.has_mission_begun:
                has_begun = True
            time.sleep(0.1)
        
        logging.info("Mission Started! Waiting for initial observations...")
        # Small sleep to let first frames arrive
        time.sleep(1)
        
        return self._get_obs()

    def step(self, action_indices):
        """
        Executes actions, calculates rewards, checks done.
        """
        # 1. Send Commands
        for i, action_idx in enumerate(action_indices):
            cmd = self.actions.get(action_idx, "move 0")
            self.agent_hosts[i].sendCommand(cmd)

        # 2. Wait (The "Tick")
        time.sleep(0.5)

        # 3. Get New State
        obs = self._get_obs()
        
        # 4. Calculate Rewards & Done
        rewards = [-0.1, -0.1] # Small time penalty for every step
        done = False
        
        # Check if Collector (Agent 1) has the diamond
        if self._check_if_collector_has_diamond(obs[1]):
            print("\n>>> DIAMOND HANDOFF SUCCESS! REWARD +100 <<<")
            rewards = [100.0, 100.0] # Both agents share the victory
            done = True
        
        # Check if mission timed out naturally
        if not self.agent_hosts[0].getWorldState().is_mission_running:
            done = True

        return obs, rewards, done, {}

    def _check_if_collector_has_diamond(self, obs_json):
        """
        Parses the Collector's JSON observation to find a diamond.
        """
        if obs_json is None:
            return False
            
        # Scan up to 40 slots
        for i in range(40):
            key = f"InventorySlot_{i}_item"
            if key in obs_json:
                if obs_json[key] == 'diamond':
                    return True
        return False

    def _get_obs(self):
        """
        Fetches the latest JSON observation for both agents.
        Returns: [dict, dict] (or None if missing)
        """
        observations = [None, None]
        for i in range(2):
            world_state = self.agent_hosts[i].getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                # Get the last one
                msg = world_state.observations[-1].text
                try:
                    observations[i] = json.loads(msg)
                except json.JSONDecodeError:
                    logging.warning(f"Agent {i} received bad JSON.")
        return observations