import os
import sys
import time
import MalmoPython

# --- 1. SETUP PATHS ---
project_root = os.getcwd()
os.environ['MALMO_XSD_PATH'] = os.path.join(project_root, 'Malmo', 'Schemas')

# --- 2. DEFINE THE MISSION ---
mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Phase 1 Handoff</Summary>
    </About>

    <ServerSection>
        <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
        </ServerInitialConditions>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
            <ServerQuitWhenAnyAgentFinishes />
        </ServerHandlers>
    </ServerSection>

    <AgentSection mode="Survival">
        <Name>MinerAgent</Name>
        <AgentStart>
            <!-- This tag was missing in our previous attempts! -->
        </AgentStart>
        <AgentHandlers>
            <ContinuousMovementCommands />
            <ObservationFromFullStats />
            <ObservationFromGrid>
                <Grid name="floor3x3">
                    <min x="-1" y="-1" z="-1" />
                    <max x="1" y="-1" z="1" />
                </Grid>
            </ObservationFromGrid>
        </AgentHandlers>
    </AgentSection>
</Mission>'''

# --- 3. INIT AGENT ---
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    exit(1)

# Validate = True. If this fails, we want to know exactly why.
try:
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
except RuntimeError as e:
    print("XML ERROR:", e)
    exit(1)

my_mission_record = MalmoPython.MissionRecordSpec()

# --- 4. CONNECT ---
max_retries = 3
for retry in range(max_retries):
    try:
        print(f"Attempt {retry+1}...")
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:", e)
            print("TIP: Restart Minecraft (Close window -> Launch Offline) to clear the zombie session.")
            exit(1)
        else:
            time.sleep(2)

# --- 5. RUN ---
print("Waiting for mission start", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

print("\n>>> PHASE 1 MISSION STARTED <<<")
print("The agent should look down and spin.")

# Action Check
agent_host.sendCommand("pitch 90")
time.sleep(1)
for i in range(10):
    agent_host.sendCommand("turn 0.5")
    time.sleep(0.2)

print("Phase 1 Base is READY.")