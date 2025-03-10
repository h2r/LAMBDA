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
from rt1_pytorch.rt1_policy import RT1Policy
from tqdm import tqdm
from lanmp_dataloader.rt1_dataloader import DatasetManager, DataLoader
import gc
import json
import pandas as pd
from ai2thor_env import ThorEnv
import pickle
import time
from tqdm import tqdm


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
        "--checkpoint-file-path",
        type=str,
        default="checkpoints/scene2/checkpoint_299183_loss_152.175.pt", #NOTE: change according to checkpoint file that is to be loaded
        help="directory to save checkpoints",
    )

    parser.add_argument(
        "--trajectory-save-path",
        type=str,
        default="traj_rollouts/scene2",
        help = "directory to save the generated trajectory predicted by the model"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )
    parser.add_argument(
        "--eval-scene",
        default=2,
        help = "scene used as validation during k-fold cross validation",
    )
    parser.add_argument(
        "--eval-subbatch",
        default=1,
    )
    parser.add_argument(
        "--split-type",
        default = 'k_fold_scene',
        choices = ['k_fold_scene', 'task_split', 'diversity_ablation'],
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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        wandb.init(project="rt1-rollout-data", config=vars(args))

    os.makedirs(args.trajectory_save_path, exist_ok=True)

    assert(os.path.isfile(args.checkpoint_file_path), "ERROR: checkpoint file does not exist")


    print("Loading dataset...")
    
    dataset_manager = DatasetManager(args.eval_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = args.num_diversity_scenes, max_trajectories = args.max_diversity_trajectories)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    

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

        body_pitch_delta = gym.spaces.Discrete(3),

        terminate_episode=gym.spaces.Discrete(2),

        pickup_release = gym.spaces.Discrete(3),

        body_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        arm_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        control_mode = gym.spaces.Discrete(7),
       
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

    


    print("Loading chosen checkpoint to model...")
    rt1_model_policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.checkpoint_file_path,
    ) 
    rt1_model_policy.model.eval()

    # Total number of params
    total_params = sum(p.numel() for p in rt1_model_policy.model.parameters())
    # Transformer params
    transformer_params = sum(p.numel() for p in rt1_model_policy.model.transformer.parameters())
    # FiLM-EfficientNet and TokenLearner params
    tokenizer_params = sum(p.numel() for p in rt1_model_policy.model.image_tokenizer.parameters())
    print(f"Total params: {total_params}")
    print(f"Transformer params: {transformer_params}")
    print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")


    print('Creating pandas dataframe for trajectories...')
    
    print_val = True
    

    for task in tqdm(val_dataloader.dataset.dataset_keys):
        

        #skip tasks that trajectory already generated for
        if os.path.isfile(os.path.join(args.trajectory_save_path, task)):
            continue
        elif print_val:
            print('START AT: ', val_dataloader.dataset.dataset_keys.index(task))
            print_val = False

        traj_group = val_dataloader.dataset.hdf[task]
        
        traj_steps = list(traj_group.keys())

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        language_command_embedding = get_text_embedding(np.array([[traj_json_dict['nl_command']]]))
        language_command_embedding = np.repeat(language_command_embedding, 6, axis=1)


        print('TASK: ', traj_json_dict['nl_command'])

        #initialize the AI2Thor environment
        ai2thor_env = ThorEnv(traj_json_dict['nl_command'])
        event = ai2thor_env.reset(traj_json_dict['scene'])

        

        #extract the visual observation from initialzed environment
        curr_image = event.frame
        visual_observation = np.expand_dims(np.expand_dims(curr_image, axis=0) , axis=0)
        visual_observation = np.repeat(visual_observation, 6, axis=1)
        
        '''
        OLD OBS FROM DATASET
        visual_observation = np.expand_dims(np.expand_dims(np.array(traj_group[traj_steps[0]]['rgb_0']), axis=0), axis=0)
        visual_observation = np.repeat(visual_observation, 6, axis=1)
        '''

        #track the starting coordinates for body, yaw rotation and arm coordinate
        curr_body_coordinate = np.array(list(event.metadata['agent']['position'].values()))
        curr_body_yaw = event.metadata['agent']['rotation']['y']
        curr_arm_coordinate = np.array(list(event.metadata['arm']['handSphereCenter'].values()))
        agent_holding = np.array([])
        

        #track the total number of steps and the last control mode
        num_steps = 0; curr_mode = None; is_terminal = False

       
        #track data for all steps
        trajectory_data = []
        
        while (curr_mode != 'stop' or is_terminal) and num_steps < ai2thor_env.max_episode_length:
            
            #provide the current observation to the model
            curr_observation = {
                'image': visual_observation,
                'context': language_command_embedding
            }
            
            generated_action_tokens = rt1_model_policy.act(curr_observation)

            #de-tokenize the generated actions from RT1
            pickup_release = val_dataloader.dataset.detokenize_pickup_release(generated_action_tokens['pickup_release'][0])
            body_pitch = val_dataloader.dataset.detokenize_head_pitch(generated_action_tokens['body_pitch_delta'][0])
            curr_mode = val_dataloader.dataset.detokenize_mode(generated_action_tokens['control_mode'][0])
            

            terminate_episode = generated_action_tokens['terminate_episode'][0]

            continuous_variables = {
                'body_position_delta': generated_action_tokens['body_position_delta'],
                'body_yaw_delta': generated_action_tokens['body_yaw_delta'],
                'arm_position_delta': generated_action_tokens['arm_position_delta'],
                'curr_mode': curr_mode
            }

            continuous_variables = val_dataloader.dataset.detokenize_continuous_data(continuous_variables)
            body_position_delta = np.squeeze(continuous_variables['body_position_delta'])
            body_yaw_delta = continuous_variables['body_yaw_delta'][0][0]
            arm_position_delta = np.squeeze(continuous_variables['arm_position_delta'])

            curr_action = val_dataloader.dataset.detokenize_action(curr_mode, body_position_delta, body_yaw_delta, arm_position_delta, pickup_release, body_pitch)



            #update the tracked coordinate data based on model output
            curr_body_coordinate += body_position_delta
            curr_body_yaw += body_yaw_delta
            curr_arm_coordinate += arm_position_delta


            #execute the generated action in the AI2THOR simulator
            step_args = {
                'xyz_body': curr_body_coordinate,
                'xyz_body_delta': body_position_delta,
                'curr_body_yaw': curr_body_yaw,
                'body_yaw_delta': body_yaw_delta,
                'arm_position_delta': arm_position_delta,
                'arm_position': curr_arm_coordinate
            }
            success, error, event = ai2thor_env.step(curr_action, step_args)

            time.sleep(0.25)
           
            #fetch object holding from simulator; also maybe fetch coordinate of body/arm + yaw from simulator
            agent_holding = np.array(event.metadata['arm']['heldObjects'])
            
            #fetch the new visual observation from the simulator, update the current mode and increment number of steps
            curr_image = np.expand_dims(np.expand_dims(event.frame, axis=0) , axis=0)

            visual_observation = visual_observation[:,1:,:,:,:]
            visual_observation = np.concatenate((visual_observation, curr_image), axis=1)
            num_steps +=1

            curr_body_coordinate = np.array(list(event.metadata['agent']['position'].values()))
            curr_body_yaw = event.metadata['agent']['rotation']['y']
            curr_arm_coordinate = np.array(list(event.metadata['arm']['handSphereCenter'].values()))
            

            #add data to the dataframe CSV
            step_data = {   
                'task': traj_json_dict['nl_command'],
                'scene': traj_json_dict['scene'],
                'img': curr_image,
                'xyz_body': curr_body_coordinate,
                'xyz_body_delta': body_position_delta,
                'yaw_body': curr_body_yaw,
                'yaw_body_delta': body_yaw_delta,
                'pitch_body': body_pitch,
                'xyz_ee': curr_arm_coordinate,
                'xyz_ee_delta': arm_position_delta,
                'pickup_dropoff': pickup_release,
                'holding_obj': agent_holding,
                'control_mode': curr_mode,
                'action': curr_action,
                'terminate': terminate_episode,
                'step': num_steps,
                'timeout': num_steps >= ai2thor_env.max_episode_length,
                'error': error
            }
            
            trajectory_data.append(step_data)

        #save the final event with all metadata: save as a json file dict
        save_path = os.path.join(args.trajectory_save_path, task)
        with open(save_path, 'wb') as file:
            pickle.dump({'trajectory_data': trajectory_data, 'final_state': event.metadata}, file)

        #close the old GUI for AI2Thor after trajectory finishes
        ai2thor_env.controller.stop()
        time.sleep(0.5)
   
if __name__ == "__main__":
    main()