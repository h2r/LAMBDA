import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Dict
import pdb
import gymnasium as gym
import numpy as np
import torch
import wandb
from torch.optim import Adam
from data import create_dataset
from tqdm import tqdm
from lanmp_dataloader.rt1_dataloader import DatasetManager, DataLoader
import gc
import json
import pandas as pd
from ai2thor_env import ThorEnv
import pickle
import time
from tqdm import tqdm
from ai2thor.controller import Controller
from motionglot.generate_lambda_actions import get_actions
from motionglot.tokenize_moma import detokenize_action
import cv2

np.random.seed(47)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path of the LAMBDA simulated dataset",
    )
    parser.add_argument(
        "--trajectory-save-path",
        type=str,
        default="dummy",
        help = "directory to save the generated trajectory predicted by the model"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )
    parser.add_argument(
        "--test-scene",
        default=None,
        help = "scene used as held-out test scene during k-fold cross validation",
    )
    parser.add_argument(
        "--split-type",
        default = 'task_split',
        choices = ['k_fold_scene', 'task_split', 'diversity_ablation'],
    )
    parser.add_argument(
        "--eval-set",
        default = 'test',
        choices = ['train', 'val', 'test'],
        help = "which of the 3 sets (train, val, held-out test) to use for inference rollouts"
    )
    parser.add_argument(
        "--subset-amt",
        default = None,
        choices =  ['25', '50', '75'],
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=3,
        help="eval batch size",
    )
    #MotionGlot specific args below
    parser.add_argument(
        "--image_token_model", 
        help="point to the file with the name of all files", 
        default= "motionglot/img_token_model.pth", 
        type=str
    )
    parser.add_argument("--checkpoint_path",
        help="point to the file with the name of all files",
        default= "dummy" ,
        type=str
    )
    parser.add_argument("--tokenizer_path", 
        help=" path to folder with tokenizer " , 
        default= "motionglot/lambda_tokenizer/lambda_task_gen_0", 
        type= str 
    ) 
    return parser.parse_args()


def main():
    with open("../../../collect_data/collect_sim/misc/cmd_id_dic.json", "r") as json_file:
        cmd_id_dic = json.load(json_file)

    args = parse_args()

    if args.wandb:
        wandb.init(project=f"mg-rollout-{args.split_type}-{args.test_scene}", config=vars(args))


    os.makedirs(args.trajectory_save_path, exist_ok=True)

    assert(os.path.isfile(args.checkpoint_path), "ERROR: checkpoint file does not exist")


    print("Loading dataset...")
    
    dataset_manager = DatasetManager(args.subset_amt, False, args.test_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = None, max_trajectories = None, dataset_path=args.dataset_path)
    
    train_dataloader = DataLoader(dataset_manager.train_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    test_dataloader = DataLoader(dataset_manager.test_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
        context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
    )

    action_space = gym.spaces.Dict(

        body_yaw_delta = gym.spaces.Box(
            low= 0, #train_dataloader.body_orientation_lim['min_yaw']
            high= 255, #train_dataloader.body_orientation_lim['max_yaw']
            shape=(1,), 
            dtype=int
        ),

        terminate_episode=gym.spaces.Discrete(2),

        arm_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        control_mode = gym.spaces.Discrete(12),
       
    )

    def start_reset(scene, controller):
        print("Starting ThorEnv...")
        if controller is not None:
            controller.stop()
            del controller
        controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=scene,
            visibilityDistance=1.5,
            gridSize=0.25,
            snapToGrid= False,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width= 1280,
            height= 720,
            fieldOfView=60
        )
        fixedDeltaTime = 0.02
        incr = 0.025
        i=0
        controller.step(action="SetHandSphereRadius", radius=0.1)
        controller.step(action="MoveArmBase", y=i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        last_event = controller.last_event
        i += incr
        return controller, last_event, i
    
    def take_action(state_action, last_event):
        incr = 0.025
        x = 0
        y = 0
        z = 0
        fixedDeltaTime = 0.02
        move = 0.2
        a = None
        word_action = state_action['word_action']
        i = state_action['i']
        # print(word_action)
        if word_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
            if word_action == "MoveAhead":
                a = dict(action="MoveAgent", ahead=move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            elif word_action == "MoveBack":
                a = dict(action="MoveAgent", ahead=-move, right=0, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            elif word_action == "MoveRight":
                a = dict(action="MoveAgent", ahead=0, right=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            elif word_action == "MoveLeft":
                a = dict(action="MoveAgent", ahead=0, right=-move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        elif word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
            a = dict(action = word_action)
        elif word_action in ['RotateAgent']:
            # diff = state_action['curr_body_yaw'] - last_event.metadata['agent']['rotation']['y']
            a = dict(action=word_action, degrees=state_action['body_yaw_delta'], returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        elif word_action in ['MoveArmBase']:
            prev_ee_y = last_event.metadata["arm"]["joints"][3]['position']['y']
            curr_ee_y = state_action['arm_position'][1]
            diff = curr_ee_y - prev_ee_y
            if diff > 0:
                i += incr
            elif diff < 0:
                i -= incr
            a = dict(action="MoveArmBase",y=i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        elif word_action in ['MoveArm']:
            a = dict(action='MoveArm',position=dict(x=state_action['arm_position'][0], y=state_action['arm_position'][1], z=state_action['arm_position'][2]),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        elif word_action in ['Done']:
            a = dict(action="Done")
        try:
            if word_action == "LookDown":
                event = controller.step(a)
                event = controller.step(a)
            else:
                event = controller.step(a)
        except Exception as e:
            print(e)         
            breakpoint()
        
        # time.sleep(0.1)
        success = event.metadata['lastActionSuccess']
        error = event.metadata['errorMessage']

        return success, error, event, i


    print('Creating pandas dataframe for trajectories...')
        
    controller = None

    if args.eval_set == "train":
        iterable_keys = train_dataloader.dataset.dataset_keys
    elif args.eval_set == "val":
        iterable_keys = val_dataloader.dataset.dataset_keys
    else:
        iterable_keys = test_dataloader.dataset.dataset_keys

    results_path = f'traj_rollouts/mg-rollout-{args.split_type}-{args.test_scene}/results.csv'
    if os.path.isfile(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['scene', 'nl_cmd', 'nav_to_target', 'grasped_target_obj', 'nav_to_target_with_obj', 'place_obj_at_goal', 'complete_traj'])
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if os.path.exists(f'traj_rollouts/mg-rollout-{args.split_type}-{args.test_scene}/trajs_done.pkl'):
        with open(f'traj_rollouts/mg-rollout-{args.split_type}-{args.test_scene}/trajs_done.pkl', 'rb') as f:
            completed_dict = pickle.load(f)
    else:
        completed_dict = {}

    if args.eval_set == 'train':
        iterable_keys = np.random.choice(np.array(iterable_keys), size=len(test_dataloader.dataset.dataset_keys), replace=False)

    for task in tqdm(iterable_keys):   

        traj_group = train_dataloader.dataset.hdf[task]
        
        traj_steps = list(traj_group.keys())

        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)

        nl_cmd = traj_json_dict['nl_command']

        #skip tasks that already rolled out
        if nl_cmd in completed_dict and completed_dict[nl_cmd] == 1:
            print("skipped")
            continue

        # start/reset THOR env for every trajectory/task
        controller, last_event, i = start_reset(traj_json_dict['scene'], controller)

        #extract the visual observation from initialzed environment
        curr_image = last_event.frame.copy()
        curr_image = cv2.resize(curr_image, (224, 224), interpolation=cv2.INTER_AREA)
        curr_image = torch.from_numpy(curr_image).permute(2,0,1).unsqueeze(0)

        #track the starting coordinates for body, yaw rotation and arm coordinate
        curr_arm_coordinate = np.array(list(last_event.metadata["arm"]["joints"][3]['position'].values()))
        agent_holding = np.array([])

        curr_base_coordinate = np.array(list(last_event.metadata["agent"]['position'].values()))

        def _get_target_obj_pos(all_objs, obj_id):
            pos = []
            if obj_id == 'Tennis_Racquet_5':
                obj_id = "Tennis_Racket_5"
            if obj_id == 'Tennis_Racquet_3':
                obj_id = "Tennis_Racket_3"
            for obj_dic in all_objs:
                if obj_dic['name'] == obj_id:
                    pos = list(obj_dic['position'].values())
            if not pos:
                breakpoint() #if this triggers during data collection, the assetId doesn't exist
            return pos
            
        def _extract_actions(action_string):
            """
            Parses an action string and returns a tuple with the action word and a list of indices.
            
            Parameters:
            - action_string (str): The action string to parse (e.g., "RotateAgent 3" or "MoveArm 1 , 2 , 3").
            
            Returns:
            - tuple: A tuple containing the action word (str) and a list of indices (list of ints).
            """
            action_string = action_string.strip()

            # Split the string into the action word and the rest
            parts = action_string.split(" ", 1)
            action_word = parts[0]  # The first part is the action word
            indices = None  # Default to None if no indices are found

            if len(parts) > 1:  # Check if there's more content after the action word
                indices_part = parts[1]  # The rest contains the indices
                # Parse indices, ensuring only valid integers are processed
                indices = [int(x.strip()) for x in indices_part.replace(",", " ").split() if x.strip().isdigit()]
                if not indices:  # If the list of indices is empty, set to None
                    indices = None
            return action_word, indices


        #track the total number of steps and the last control mode
        num_steps = 0; action_discrete = None; is_terminal = False

        #track data for all steps
        # trajectory_data = []
        
        print("\n")
        print("\n")
        print('TASK: ', traj_json_dict['nl_command'])
        print("\n")
        print("\n")
        time.sleep(1)
        pickedup = False
        while (action_discrete != 'Done' or is_terminal) and num_steps < 75:
            mg_tokens = get_actions(curr_image, nl_cmd, args.tokenizer_path, args.image_token_model, args.checkpoint_path)
            print(f'mg: {mg_tokens}')

            action_discrete, action_cont_idx = _extract_actions(mg_tokens)
            #skips nonsense motionglot predictions that are not real actions
            if action_discrete not in ["Done", "MoveArm", "LookUp", "LookDown", "RotateAgent", "MoveAhead", "MoveArmBase", "PickUpObject", "ReleaseObject", "MoveBack", "MoveLeft", "MoveRight"]:
                print(f"{action_discrete} not valid, skipped!")
                continue
            flag = False
            if action_cont_idx:
                for i in action_cont_idx:
                    if not isinstance(i, int):
                        flag = True
            if flag:
                continue
            if action_discrete in ['MoveArm', 'MoveArmBase']:
                if len(action_cont_idx) == 1:
                    continue
                if not action_cont_idx:
                    continue
                if action_discrete == "MoveArm" and len(action_cont_idx) != 3:
                    continue
                action_delta_arm = detokenize_action(action_discrete, action_cont_idx)
                action_delta_rot = 0
            elif action_discrete in ['RotateAgent']:
                if not action_cont_idx:
                    continue
                action_delta_rot = detokenize_action(action_discrete, action_cont_idx)
                action_delta_arm = [0,0,0]
            else:
                action_delta_arm = [0,0,0]
                action_delta_rot = 0
            
            #update the tracked coordinate data based on model output
            curr_arm_coordinate += action_delta_arm


            #execute the generated action in the AI2THOR simulator
            step_args = {
                'word_action': action_discrete,
                'body_yaw_delta': action_delta_rot,
                'arm_position': curr_arm_coordinate,
                'i': i
            }

            success, error, last_event, i = take_action(step_args, last_event)

            if last_event.metadata["arm"]['heldObjects'] and not pickedup:
                pickedup=True
                time.sleep(0.5)
                print("GRASPED SOMETHING!!!!")

            
            #fetch object holding from simulator; also maybe fetch coordinate of body/arm + yaw from simulator
            # agent_holding = np.array(last_event.metadata['arm']['heldObjects'])
            
            #fetch the new visual observation from the simulator, update the current mode and increment number of steps
            curr_image = last_event.frame.copy()
            curr_image = cv2.resize(curr_image, (224, 224), interpolation=cv2.INTER_AREA)
            curr_image = torch.from_numpy(curr_image).permute(2,0,1).unsqueeze(0)
            
            curr_arm_coordinate = np.array(list(last_event.metadata["arm"]["joints"][3]['position'].values()))
            
            num_steps +=1
            time.sleep(0.1)

            #add data to the dataframe CSV
            # step_data = {   
            #     'task': traj_json_dict['nl_command'],
            #     'scene': traj_json_dict['scene'],
            #     'img': curr_image,
            #     'yaw_body_delta': body_yaw_delta,
            #     'xyz_ee': curr_arm_coordinate,
            #     'xyz_ee_delta': arm_position_delta,
            #     'holding_obj': agent_holding,
            #     'control_mode': curr_mode,
            #     'action': curr_action,
            #     # 'terminate': terminate_episode,
            #     'step': num_steps,
            #     'timeout': num_steps >=1500,
            #     'error': error
            # }
            
            # trajectory_data.append(step_data)

        #save the final event with all metadata: save as a json file dict
        # save_path = os.path.join(args.trajectory_save_path, task)
        # with open(save_path, 'wb') as file:
        #     pickle.dump({'trajectory_data': trajectory_data, 'final_state': last_event.metadata}, file)

        #close the old GUI for AI2Thor after trajectory finishes
        # ai2thor_env.controller.stop()

        nav_to_target = input("Enter score for nav_to_target: ")
        grasped_target_obj = input("Enter score for grasped_target_obj: ")
        nav_to_target_with_obj = input("Enter score for nav_to_target_with_obj: ")
        place_obj_at_goal = input("Enter score for place_obj_at_goal: ")
        complete_traj = input("Enter score for complete_traj: ")

        traj_row = [traj_json_dict['scene'], traj_json_dict['nl_command'], nav_to_target, grasped_target_obj, nav_to_target_with_obj, place_obj_at_goal, complete_traj]
        results_df.loc[len(results_df)] = traj_row
        results_df.to_csv(results_path, index=False)
        
        completed_dict[traj_json_dict['nl_command']] = 1
        with open(f'traj_rollouts/mg-rollout-{args.split_type}-{args.test_scene}/trajs_done.pkl', 'wb') as f:
            pickle.dump(completed_dict, f)
        
if __name__ == "__main__":
    main()