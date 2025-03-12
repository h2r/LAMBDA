import argparse
import os
import json
import zipfile
import ai2thor
from ai2thor.controller import Controller
from time import sleep
import shutil


parser = argparse.ArgumentParser(description='Process a trajectory file path.')

parser.add_argument('--traj_path', type=str, required=True, help='Path to the trajectory file')
parser.add_argument('--scene', type=str, required=True, help='sim scene')
parser.add_argument('--slow', action='store_true', help='makes replay slower')

args = parser.parse_args()

TRAJ_PATH = args.traj_path
SCENE = args.scene
DIR_EXTRACT = "/mnt/ahmed/new_sim_data/files/1_3"


controller = None
last_event = None
i = 0

def init():
    global i, controller, last_event

    print("Starting ThorEnv...")
    if controller is not None:
        controller.stop()
        del controller
    controller = Controller(
        agentMode="arm",
        massThreshold=None,
        scene=SCENE,
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid= False,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        # width=300,
        # height=300,
        width= 1280,
        height= 720,
        fieldOfView=60
    )
    # self.controller.step(action="Initialize")
    last_event = controller.last_event

    fixedDeltaTime = 0.02
    incr = 0.025
    controller.step(action="SetHandSphereRadius", radius=0.1)
    controller.step(action="MoveArmBase", y=i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
    i += incr


# def get_first_event():
#     return last_event.metadata

def take_action(state_action):
    global i, last_event
    incr = 0.025
    x = 0
    y = 0
    z = 0
    fixedDeltaTime = 0.02
    move = 0.2
    a = None
    word_action = state_action['action']
    print(word_action)
    # print(f'word_action: {word_action}')
    if word_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
        # global_coord_agent = self.last_event.metadata['agent']['position']
        # prev_state_body = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z']]
        # diff = np.array(state_action['state_body']) - np.array(prev_state_body)
        # a = dict(action="Teleport", position=dict(x=state_action['state_body'][0], y=state_action['state_body'][1], z=state_action['state_body'][2]))

        # a = dict(action="Teleport", position=dict(x=diff[0], y=diff[1], z=diff[2]))

        if word_action == "MoveAhead":
            a = dict(action="MoveAgent", ahead=move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        elif word_action == "MoveBack":
            a = dict(action="MoveAgent", ahead=-move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        elif word_action == "MoveRight":
            a = dict(action="MoveAgent", ahead=0, right=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        elif word_action == "MoveLeft":
            a = dict(action="MoveAgent", ahead=0, right=-move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        # a = dict(action=word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

    elif word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
        a = dict(action = word_action)
    elif word_action in ['RotateAgent']:
        diff = state_action['body_yaw'] - last_event.metadata['agent']['rotation']['y']
        # print('rot diff: ', diff)
        # a = dict(action="Teleport", rotation=dict(x=0, y=diff, z=0))
        a = dict(action=word_action, degrees=diff, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
    elif word_action in ['MoveArmBase']:
        prev_ee_y = last_event.metadata["arm"]["joints"][3]['position']['y']
        curr_ee_y = state_action['global_state_ee'][1]
        diff = curr_ee_y - prev_ee_y
        # print(f'prev_ee_y: ', prev_ee_y)
        # print(f'curr_ee_y: ', curr_ee_y)
        # print(f'diff: {diff}')
        if diff > 0:
            i += incr
        elif diff < 0:
            i -= incr
        a = dict(action="MoveArmBase",y=i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
    elif word_action in ['MoveArm']:
        a = dict(action='MoveArm',position=dict(x=state_action['global_state_ee'][0], y=state_action['global_state_ee'][1], z=state_action['global_state_ee'][2]),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)

    # if a['action'] == 'Teleport' and word_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
        # breakpoint()
    
    try:
        if word_action == "LookDown":
            event = controller.step(a)
            event = controller.step(a)
        else:
            event = controller.step(a)
    except:
        breakpoint()
    if args.slow:
        sleep(0.33)
    success = event.metadata['lastActionSuccess']
    error = event.metadata['errorMessage']
    last_event = event
    return success, error, last_event.metadata


def process_json_files(main_dir):
    # Iterate through all subfolders in the main directory
    # Get a list of all subfolders that follow the 'folder_NUM' format
    folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f)) and f.startswith('folder_')]

    # Sort the folders based on the numerical part of the folder name
    sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[1]))

    j=0
    k=0
    for folder in sorted_folders:
        if j==0:
            j+=1
            continue
        folder_path = os.path.join(main_dir, folder)

        # Ensure the folder follows the "folder_NUM" format and is a directory
        if os.path.isdir(folder_path) and folder.startswith("folder_"):
            # Look for the json file inside the folder
            for file in os.listdir(folder_path):
                if file.endswith(".json") and file.startswith("data_chunk_"):
                    json_file_path = os.path.join(folder_path, file)

                    # Open and process the JSON file
                    with open(json_file_path, 'r') as json_file:
                        data = json.load(json_file)
                        if k == 0:
                            print(data['target_obj'])
                            print(data['nl_command'])
                            sleep(1)
                        k+=1
                        # breakpoint()

                        # Loop through the 'steps' and extract 'action', 'state_body', and 'state_ee'
                        for step in data.get("steps", []):
                            action = step.get("action")
                            state_body = step.get("global_state_body")
                            state_ee = step.get("global_state_ee")

                            state_action = {}
                            state_action['action'] = action
                            state_action['global_state_body'] = state_body[:3]
                            state_action['body_yaw'] = state_body[-1]
                            state_action['global_state_ee'] = state_ee[:3]

                            take_action(state_action)

    shutil.rmtree(os.path.splitext(TRAJ_PATH)[0])
    print('Deleted extracted directory!')



def unzip_file():
    # Check if the zip file exists
    if not os.path.exists(TRAJ_PATH):
        print(f"File {TRAJ_PATH} does not exist.")
        return

    # Get the name of the zip file without the .zip extension
    extract_to_dir = os.path.splitext(TRAJ_PATH)[0]

    # Create the directory if it doesn't exist
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    # Unzip the file
    with zipfile.ZipFile(TRAJ_PATH, 'r') as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(extract_to_dir)
        print(f"Extracted all files to {extract_to_dir}")

    # Delete the zip file after extraction
    # os.remove(TRAJ_PATH)
    # print(f"Deleted the zip file: {TRAJ_PATH}")


if __name__ == "__main__":
    unzip_file()
    init()
    process_json_files(DIR_EXTRACT+"/"+TRAJ_PATH.rsplit('/', 1)[-1][:-4])