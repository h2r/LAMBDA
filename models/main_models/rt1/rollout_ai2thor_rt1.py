import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Dict
import pdb
import gymnasium as gym
import numpy as np
import torch
import wandb
from sentence_transformers import SentenceTransformer
from torch.optim import Adam
import tensorflow_hub as hub 
from data import create_dataset
from rt1_pytorch.rt1.rt1_policy import RT1Policy
from rt1_pytorch.rm1.rm1_policy import RM1Policy
from rt1_pytorch.rl1.rl1_policy import RL1Policy
from tqdm import tqdm
from lanmp_dataloader.rt1_dataloader import DatasetManager, DataLoader
import gc
import json
import pandas as pd
from ai2thor_env import ThorEnv
import pickle
import time
from ai2thor.controller import Controller
import cv2

np.random.seed(47)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default=None,
        help="SentenceTransformer to use; default is None for original USE embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--arch-type",
        default = "transformer",
        choices = ["transformer", "lstm", "mamba"],
    )
    parser.add_argument(
        "--checkpoint-file-path",
        type=str,
        default="dummy", #NOTE: change according to checkpoint file that is to be loaded
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--trajectory-save-path",
        type=str,
        default="results/traj_rollouts/scene2",
        help = "directory to save the generated trajectory predicted by the model"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path of the LAMBDA simulated dataset",
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
        "--num-diversity-scenes",
        default = 3,
    )
    parser.add_argument(
        "--max-diversity-trajectories",
        default = 100,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=3,
        help="eval batch size",
    )
    parser.add_argument(
        "--use-dist",
        help='use distance input if true, not if false', 
        action='store_true'
    )
    return parser.parse_args()


def main():
    with open("../../../collect_data/collect_sim/misc/cmd_id_dic.json", "r") as json_file:
        cmd_id_dic = json.load(json_file)

    args = parse_args()

    if args.use_dist:
        dist = 'dist'
    else:
        dist='nodist'
    if args.wandb:
        wandb.init(project=f"rt1-rollout-{dist}-{args.split_type}-{args.test_scene}-subset25", config=vars(args))


    os.makedirs(args.trajectory_save_path, exist_ok=True)

    assert(os.path.isfile(args.checkpoint_file_path), "ERROR: checkpoint file does not exist")


    print("Loading dataset...")
    
    dataset_manager = DatasetManager(args.subset_amt, args.use_dist, args.test_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = args.num_diversity_scenes, max_trajectories = args.max_diversity_trajectories, dataset_path=args.dataset_path)
    
    train_dataloader = DataLoader(dataset_manager.train_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    test_dataloader = DataLoader(dataset_manager.test_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)

    if args.use_dist:
        observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
            context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
            ee_obj_dist=gym.spaces.Box(low=float(-1), high=np.inf, shape=(512,), dtype=np.float32), #added
            goal_dist=gym.spaces.Box(low=float(-1), high=np.inf, shape=(512,), dtype=np.float32) #added2
        )
    else:
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



    #NOTE: has to be Not None because of raw instruction input
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
    )
   

    def get_text_embedding(observation: Dict):
        
        if args.sentence_transformer is not None:
            return text_embedding_model.encode(observation)
        else:
            embedded_observation = []

            for i in range(0, observation.shape[1]):
                
                try:
                    embedded_observation.append( np.array(text_embedding_model(observation[:, i]) ) )
                except:
                    raise Exception('Error: task descriptions could not be embedded')

            embedded_observation = np.stack(embedded_observation, axis=1)
            return embedded_observation

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
        print(word_action)
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
        elif word_action in ['stop']:
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
        
        success = event.metadata['lastActionSuccess']
        error = event.metadata['errorMessage']

        return success, error, event, i


    print("Loading chosen checkpoint to model...")
    if args.arch_type == "transformer":
        policy = RT1Policy
    elif args.arch_type == "lstm":
        policy = RL1Policy
    else:
        policy = RM1Policy

    model_policy = policy(
        dist=args.use_dist,
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.checkpoint_file_path,
    ) 
    model_policy.model.eval()

    # Total number of params
    # total_params = sum(p.numel() for p in model_policy.model.parameters())
    # # Transformer params
    # transformer_params = sum(p.numel() for p in model_policy.model.transformer.parameters())
    # # FiLM-EfficientNet and TokenLearner params
    # tokenizer_params = sum(p.numel() for p in model_policy.model.image_tokenizer.parameters())
    # print(f"Total params: {total_params}")
    # print(f"Transformer params: {transformer_params}")
    # print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")


    print('Creating pandas dataframe for trajectories...')
        
    controller = None

    if args.eval_set == "train":
        iterable_keys = train_dataloader.dataset.dataset_keys
    elif args.eval_set == "val":
        iterable_keys = val_dataloader.dataset.dataset_keys
    else:
        iterable_keys = test_dataloader.dataset.dataset_keys

    results_path = f'traj_rollouts/mg-rollout-{dist}-{args.split_type}-{args.test_scene}-ft-seen/results.csv'
    if os.path.isfile(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['scene', 'nl_cmd', 'nav_to_target', 'grasped_target_obj', 'nav_to_target_with_obj', 'place_obj_at_goal', 'complete_traj'])
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if os.path.exists(f'traj_rollouts/mg-rollout-{dist}-{args.split_type}-{args.test_scene}-ft-seen/trajs_done.pkl'):
        with open(f'traj_rollouts/mg-rollout-{dist}-{args.split_type}-{args.test_scene}-ft-seen/trajs_done.pkl', 'rb') as f:
            completed_dict = pickle.load(f)
    else:
        completed_dict = {}

    if args.eval_set == 'train':
        iterable_keys = np.random.choice(np.array(iterable_keys), size=len(test_dataloader.dataset.dataset_keys), replace=False)

    for task in iterable_keys:
        # print(task)
        # continue
        traj_group = train_dataloader.dataset.hdf[task]
        
        traj_steps = list(traj_group.keys())

        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)

        nl_cmd = traj_json_dict['nl_command']
        # print(nl_cmd)
        # continue

        #skip tasks that already rolled out
        if nl_cmd in completed_dict and completed_dict[nl_cmd] == 1:
            print("skipped")
            continue

        #get NL cmd embedding
        language_command_embedding = get_text_embedding(np.array([[nl_cmd]]))
        language_command_embedding = np.repeat(language_command_embedding, 6, axis=1)

        # start/reset THOR env for every trajectory/task
        controller, last_event, i = start_reset(traj_json_dict['scene'], controller)

        #extract the visual observation from initialzed environment
        curr_image = last_event.frame.copy()
        visual_observation = np.expand_dims(np.expand_dims(curr_image, axis=0) , axis=0)
        visual_observation = np.repeat(visual_observation, 6, axis=1)

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

        def _get_distances(flag):
            goal_pos = traj_json_dict['goal_pos']
            dist_to_goal = np.linalg.norm(np.array(goal_pos) - curr_base_coordinate)
            dist_to_goal = np.array([[dist_to_goal]])
            if flag:
                dist_to_goal = np.repeat(dist_to_goal, 6, axis=1)


            all_obj = last_event.metadata["objects"]
            obj_id = traj_json_dict['target_obj']
            obj_pos = _get_target_obj_pos(all_obj, obj_id)
            ee_dist_to_obj = np.linalg.norm(np.array(obj_pos) - curr_arm_coordinate)
            ee_dist_to_obj = np.array([[ee_dist_to_obj]])
            if flag:
                ee_dist_to_obj = np.repeat(ee_dist_to_obj, 6, axis=1)
            
            return dist_to_goal, ee_dist_to_obj

        #extract distances from the environment
        if args.use_dist:
            dist_to_goal, ee_dist_to_obj = _get_distances(True)


        #track the total number of steps and the last control mode
        num_steps = 0; curr_mode = None; is_terminal = False

       
        #track data for all steps
        # trajectory_data = []
        
        print("\n")
        print("\n")
        print('TASK: ', traj_json_dict['nl_command'])
        print("\n")
        print("\n")
        time.sleep(1)
        pickedup = False
        while (curr_mode != 'stop' or is_terminal) and num_steps < 150:
            
            #provide the current observation to the model
            if args.use_dist:
                curr_observation = {
                    'image': visual_observation,
                    'context': language_command_embedding,
                    'goal_dist': dist_to_goal,
                    'ee_obj_dist': ee_dist_to_obj
                }
            else:
                curr_observation = {
                    'image': visual_observation,
                    'context': language_command_embedding
                }

            generated_action_tokens = model_policy.act(curr_observation)
        
           
            # terminate_episode = generated_action_tokens['terminate_episode'][0] #not needed for actual rolling out

            curr_mode = train_dataloader.dataset.detokenize_mode(generated_action_tokens['control_mode'][0])

            continuous_variables = {
                'body_yaw_delta': generated_action_tokens['body_yaw_delta'],
                'arm_position_delta': generated_action_tokens['arm_position_delta'],
                'curr_mode': curr_mode
            }
            continuous_variables = train_dataloader.dataset.detokenize_continuous_data(continuous_variables)
            body_yaw_delta = continuous_variables['body_yaw_delta'][0][0]
            arm_position_delta = np.squeeze(continuous_variables['arm_position_delta'])
            curr_action = train_dataloader.dataset.detokenize_action(curr_mode, body_yaw_delta, arm_position_delta)
        
            
            #update the tracked coordinate data based on model output
            curr_arm_coordinate += arm_position_delta


            #execute the generated action in the AI2THOR simulator
            step_args = {
                'word_action': curr_mode,
                'body_yaw_delta': body_yaw_delta,
                'arm_position': curr_arm_coordinate,
                'i': i
            }

            success, error, last_event, i = take_action(step_args, last_event)

            if last_event.metadata["arm"]['heldObjects'] and not pickedup:
                pickedup=True
                time.sleep(0.5)
                print("GRASPED SOMETHING!!!!")

            #fetch new distances from the environment
            if args.use_dist:
                new_dist_to_goal, new_ee_dist_to_obj = _get_distances(False)
                #removes the oldest distances in the window of 6 and adds the latest to replace it
                dist_to_goal = dist_to_goal[:,1:]
                dist_to_goal = np.concatenate((new_dist_to_goal, dist_to_goal), axis=1)
                ee_dist_to_obj = ee_dist_to_obj[:,1:]
                ee_dist_to_obj = np.concatenate((new_ee_dist_to_obj, ee_dist_to_obj), axis=1)

            
            #fetch object holding from simulator; also maybe fetch coordinate of body/arm + yaw from simulator
            # agent_holding = np.array(last_event.metadata['arm']['heldObjects'])
            
            #fetch the new visual observation from the simulator, update the current mode and increment number of steps
            curr_image = np.expand_dims(np.expand_dims(last_event.frame, axis=0) , axis=0)

            #removes the oldest observation in the window of 6 and adds the latest to replace it
            visual_observation = visual_observation[:,1:,:,:,:]
            visual_observation = np.concatenate((visual_observation, curr_image), axis=1)
            
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
        with open(f'traj_rollouts/mg-rollout-{dist}-{args.split_type}-{args.test_scene}-ft-seen/trajs_done.pkl', 'wb') as f:
            pickle.dump(completed_dict, f)
        
if __name__ == "__main__":
    main()