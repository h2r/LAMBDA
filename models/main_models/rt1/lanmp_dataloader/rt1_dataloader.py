import os
import sys
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from PIL import Image
from tqdm import tqdm
# from models.utils.data_utils import split_data
import pdb
#mainly for debugging
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import sys
from copy import copy
import random
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import pickle as pk

np.random.seed(47)

sys.path.append('..')

# DATASET_PATH = '/mnt/ahmed/lanmp_dataset_newest.hdf5'
DATASET_PATH = '/oscar/data/stellex/shared/lanmp/lanmp_dataset_newest.hdf5'

'''
train_keys, val_keys, test_keys = split_data(self.args.data, splits['train'], splits['val'], splits['test'])
'''

def cluster(hdf5_path, low_div):
    random.seed(3)
    model = SentenceTransformer("all-mpnet-base-v2") # all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-distilroberta-v1, sentence-t5-base, all-mpnet-base-v2

    if not os.path.exists("sim_commands.npy") or not os.path.exists("command_key_dict.json"):
        commands=[]
        command_key_dict = {}
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf_file:
            # Iterate through each trajectory group
            for key,(trajectory_name, trajectory_group) in zip(hdf_file.keys() ,hdf_file.items()):
                # Iterate through each timestep group within the trajectory
                for timestep_name, timestep_group in trajectory_group.items():
                    # Read and decode the JSON metadata
                    metadata = json.loads(timestep_group.attrs['metadata'])
                    commands.append(metadata['nl_command'])
                    command_key_dict[metadata['nl_command']] = key
                    break

        np.save("sim_commands.npy", commands)
        with open('command_key_dict.json', 'w') as f:
            json.dump(command_key_dict, f)
    else:
        commands = np.load('sim_commands.npy', allow_pickle=True).tolist()
        with open('command_key_dict.json', 'r') as file:
            command_key_dict = json.load(file)

    # Convert the commands to embeddings
    embeddings = model.encode(commands)

    # Compute the cosine similarity matrix
    # sim_mat = cosine_similarity(embeddings)
    # print(sim_mat)

    if not os.path.exists("cluster_dict.json"):
        # Apply Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3)
        clusters = clustering.fit_predict(embeddings)

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
        print(f'Silhouette Score: {silhouette_avg}')

        # Calculate the Davies-Bouldin index
        db_index = davies_bouldin_score(embeddings, clusters)
        print(f'Davies-Bouldin Index: {db_index}')

        # Create a defaultdict with lists as default values
        cluster_dict = defaultdict(list)

        # Populate the dictionary
        for string, cluster_id in zip(commands, clusters):
            cluster_dict[int(cluster_id)].append(string)

        cluster_dict = dict(cluster_dict)

        #save dict
        with open('cluster_dict.json', 'w') as f:
            json.dump(cluster_dict, f)
    else:
        with open('cluster_dict.json', 'r') as file:
            cluster_dict = json.load(file)
    # Find the cluster with the longest list
    sorted_clusters = sorted(cluster_dict, key=lambda k: len(cluster_dict[k]), reverse=True)

    if low_div:
        start = 0
        end = 10
    else: # high div
        start = 14
        end = 106

    # tot = 0
    train_keys = []
    if low_div:
        for i in range(start,end):
            # print(f"{len(cluster_dict[sorted_clusters[i]])} elements.")
            command_lst = cluster_dict[sorted_clusters[i]]
            for cmd in command_lst:
                key = command_key_dict[cmd]
                train_keys.append(key)
            # tot += len(cluster_dict[sorted_clusters[i]])
        #print(f'Total: {tot}')
        num_elements_to_pick = len(train_keys) - 2
        train_keys = random.sample(train_keys, num_elements_to_pick)
    else:
        for i in range(start,end):
            # print(f"{len(cluster_dict[sorted_clusters[i]])} elements.")
            command_lst = cluster_dict[sorted_clusters[i]]
            for cmd in command_lst:
                key = command_key_dict[cmd]
                train_keys.append(key)
            # tot += len(cluster_dict[sorted_clusters[i]])
        #print(f'Total: {tot}')
        # num_elements_to_pick = len(train_keys) - 
        # train_keys = random.sample(train_keys, num_elements_to_pick)
    test_keys = []
    for i in range(10,14): #test
        command_lst = cluster_dict[sorted_clusters[i]]
        for cmd in command_lst:
            key = command_key_dict[cmd]
            test_keys.append(key)
    num_elements_to_pick = len(test_keys) - 5
    test_keys = random.sample(test_keys, num_elements_to_pick)

    return train_keys, test_keys

def split_data(hdf5_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Assuming trajectories or data units are top-level groups in the HDF5 file
        keys = list(hdf_file.keys())
        total_items = len(keys)
        
        # Generate a shuffled array of indices
        indices = np.arange(total_items)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_end = int(train_ratio * total_items)
        val_end = train_end + int(val_ratio * total_items)
        
        # Split the indices
        train_indices = indices[:train_end+1]
        val_indices = indices[train_end+1:val_end+1]
        test_indices = indices[val_end+1:]
        
        # Convert indices back to keys (assuming order in keys list is stable and matches original order)
        train_keys = [keys[i] for i in train_indices]
        val_keys = [keys[i] for i in val_indices]
        test_keys = [keys[i] for i in test_indices]

        return train_keys, val_keys, test_keys

def split_by_scene(hdf5_path):
    #mapping which keys are relevant to specific scenes
    scene_to_keys = {}

    with h5py.File(hdf5_path, 'r') as hdf_file:

        keys = list(hdf_file.keys())

        for k in keys:
            traj_json_dict = json.loads(hdf_file[k]['folder_0'].attrs['metadata'])

            if traj_json_dict['scene'] not in scene_to_keys:
                scene_to_keys[traj_json_dict['scene']] = []
            
            scene_to_keys[traj_json_dict['scene']].append(k)
    
    for k in scene_to_keys.keys():
        scene_to_keys[k] = list(sorted(scene_to_keys[k]))
    
    with open('./lanmp_dataloader/scene_to_keys.json', 'w') as f:
        json.dump(scene_to_keys, f)

    return scene_to_keys

def low_high_scene(hdf5_path, train_runs, test_run):
    np.random.seed(420)
    test_trajs = []
    scene1 = []
    scene2 = []
    scene3 = []
    scene4 = []
    div_by_two = 52
    div_by_three = 26 #31
    div_by_four = 26
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Iterate over each group in the HDF5 file, each group represents a trajectory
        for trajectory_name in hdf_file:
            trajectory_group = hdf_file[trajectory_name]

            # Get the first group name from sorted keys
            first_timestep_name = sorted(trajectory_group.keys())[0]
            first_timestep_group = trajectory_group[first_timestep_name]

            # Read JSON data from the 'metadata' attribute of the first timestep
            if 'metadata' in first_timestep_group.attrs:
                json_data = json.loads(first_timestep_group.attrs['metadata'])
                if json_data['scene'] == test_run:
                    test_trajs.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train8_1':
                    scene1.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train12_3':
                    scene2.append(trajectory_name)
                elif json_data['scene'] == 'FloorPlan_Train5_1':
                    scene3.append(trajectory_name)
                else:
                    scene4.append(trajectory_name)
    
    if len(train_runs) == 1:
        train_trajs = np.array(scene4)
    elif len(train_runs) == 2:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_two, replace=False),
            np.random.choice(scene2, size=div_by_two, replace=False),
        ])
    elif len(train_runs) == 3:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_three, replace=False),
            np.random.choice(scene2, size=div_by_three, replace=False),
            np.random.choice(scene3, size=div_by_three-1, replace=False)
        ])
    elif len(train_runs) == 4:
        train_trajs = np.concatenate([
            np.random.choice(scene1, size=div_by_four, replace=False),
            np.random.choice(scene2, size=div_by_four, replace=False),
            np.random.choice(scene3, size=div_by_four, replace=False),
            np.random.choice(scene4, size=div_by_four, replace=False)

        ])
    
    test_trajs = np.random.choice(test_trajs, size=20, replace=False)
    return train_trajs.tolist(), test_trajs.tolist()

def sort_folders(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

class DatasetManager(object):

    '''
    NOTE: kwargs should contain a dictionary with keys {'train_split' : x, 'val_split': y, 'test_split':z} where x+y+z = 1
    '''
    def __init__(self, subset_amt, use_dist, test_scene=1, train_split=0.8, val_split=0.1, test_split=0.1, split_style='task_split', diversity_scenes=1, max_trajectories=100, low_div=True):
        self.use_dist = use_dist
        
        assert( train_split + val_split + test_split == 1.0, 'Error: train, val and test split do not sum to 1.0')

        
        # train_keys, val_keys, test_keys = split_data(DATASET_PATH, train_split, val_split, test_split)
        if 'scene_to_keys.json' not in os.listdir('./lanmp_dataloader'):
            self.scene_to_keys = split_by_scene(DATASET_PATH)
        else:
            with open('./lanmp_dataloader/scene_to_keys.json') as f:
                self.scene_to_keys = json.load(f)
             

        self.scenes = list(sorted(list(self.scene_to_keys.keys())))

        assert(split_style in ['k_fold_scene', 'task_split', 'diversity_ablation'], "Error: input split_style is invalid")
        if split_style == 'k_fold_scene':
            assert(int(test_scene) < len(self.scenes), "Error: input test scene is out of index space")

            train_keys = []
            val_keys = []

            # Iterate through all scenes except the test scene
            for x in range(len(self.scenes)):
                if x != int(test_scene):
                    scene_keys = self.scene_to_keys[self.scenes[x]]
                    np.random.shuffle(scene_keys)
                    # Stratified split: use 80% for training and 20% for validation
                    split_idx = int(0.8 * len(scene_keys))
                    train_keys += scene_keys[:split_idx]
                    val_keys += scene_keys[split_idx:]

            # The test set is assigned manually based on the test_scene input
            test_keys = self.scene_to_keys[self.scenes[int(test_scene)]]
            print(len(train_keys))
            print(len(val_keys))
            print(len(test_keys))

            # Ensure no overlap between train, val, and test sets
            assert(len(set(train_keys) & set(val_keys)) == 0), "Error: Train and Val sets overlap"
            assert(len(set(train_keys) & set(test_keys)) == 0), "Error: Train and Test sets overlap"
            assert(len(set(val_keys) & set(test_keys)) == 0), "Error: Val and Test sets overlap"

            if not os.path.isfile('train_cmds_scene_gen_{test_scene}.pkl') and not os.path.isfile(f'val_cmds_scene_gen_{test_scene}.pkl'):
                hdf = h5py.File(DATASET_PATH, 'r')
                train_cmds = []
                for key in train_keys:
                    traj_group = hdf[key]
                
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    #extract the NL command
                    json_str = traj_group[traj_steps[0]].attrs['metadata']
                    traj_json_dict = json.loads(json_str)
                    nl_command = traj_json_dict['nl_command']
                    train_cmds.append(nl_command)
                
                val_cmds = []
                for key in val_keys:
                    traj_group = hdf[key]
                
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    #extract the NL command
                    json_str = traj_group[traj_steps[0]].attrs['metadata']
                    traj_json_dict = json.loads(json_str)
                    nl_command = traj_json_dict['nl_command']
                    val_cmds.append(nl_command)
                
                
                with open(f'train_cmds_scene_gen_{test_scene}.pkl', 'wb') as file:
                    pk.dump(train_cmds, file)
                with open(f'val_cmds_scene_gen_{test_scene}.pkl', 'wb') as file:
                    pk.dump(val_cmds, file)
        elif split_style == 'task_split':
            train_keys = []
            val_keys = []
            test_keys = []

            for scene in self.scenes:
                
                scene_keys = copy(self.scene_to_keys[scene])
                np.random.shuffle(scene_keys)
                    
                split_idx = int(len(scene_keys)*(train_split))
                split_idx2 = int(len(scene_keys)*(train_split+val_split))
                
                if subset_amt is not None:
                    train_keys_temp = scene_keys[:split_idx]
                    num_samples = int(np.ceil(len(train_keys_temp) * (int(subset_amt) / 100)))
                    train_keys += np.random.choice(train_keys_temp, size=num_samples, replace=False).tolist()
                else:
                    train_keys += scene_keys[:split_idx]
                val_keys += scene_keys[split_idx:split_idx2]
                test_keys += scene_keys[split_idx2:]

            # Ensure no overlap between train, val, and test sets
            assert(len(set(train_keys) & set(val_keys)) == 0), "Error: Train and Val sets overlap"
            assert(len(set(train_keys) & set(test_keys)) == 0), "Error: Train and Test sets overlap"
            assert(len(set(val_keys) & set(test_keys)) == 0), "Error: Val and Test sets overlap"


            if not os.path.isfile('train_cmds_task_gen.pkl') and not os.path.isfile('val_cmds_task_gen.pkl'):
                hdf = h5py.File(DATASET_PATH, 'r')
                train_cmds = []
                for key in train_keys:
                    traj_group = hdf[key]
                
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    #extract the NL command
                    json_str = traj_group[traj_steps[0]].attrs['metadata']
                    traj_json_dict = json.loads(json_str)
                    nl_command = traj_json_dict['nl_command']
                    train_cmds.append(nl_command)
                
                val_cmds = []
                for key in val_keys:
                    traj_group = hdf[key]
                
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    #extract the NL command
                    json_str = traj_group[traj_steps[0]].attrs['metadata']
                    traj_json_dict = json.loads(json_str)
                    nl_command = traj_json_dict['nl_command']
                    val_cmds.append(nl_command)
                
                
                with open('train_cmds_task_gen.pkl', 'wb') as file:
                    pk.dump(train_cmds, file)
                with open('val_cmds_task_gen.pkl', 'wb') as file:
                    pk.dump(val_cmds, file)

####################################################################################
            # import pickle
            # import cv2
            # from skimage.transform import resize
            # import matplotlib.pyplot as plt

            # hdf = h5py.File(DATASET_PATH, 'r')
            # keys = train_keys + val_keys + test_keys

            # final_arr = []
            # for key in tqdm(keys):
            #     traj_group = hdf[key]
            #     traj_steps = list(traj_group.keys())
            #     traj_steps.sort(key=sort_folders) 

            #     next_discrete_actions = []
            #     for i in range(len(traj_steps)):
            #         json_str = traj_group[traj_steps[i]].attrs['metadata']
            #         traj_json_dict = json.loads(json_str)
            #         discrete_action = traj_json_dict['steps'][0]['action']
            #         next_discrete_actions.append(discrete_action)
            #     next_discrete_actions = next_discrete_actions[1:]                        

            #     traj_arr = []
            #     for i in range(len(traj_steps)):
            #         step_arr = []

            #         json_str = traj_group[traj_steps[i]].attrs['metadata']
            #         traj_json_dict = json.loads(json_str)
            #         nl_command = traj_json_dict['nl_command']
            #         scene = traj_json_dict['scene']
            #         rgb = np.array(traj_group[traj_steps[i]]['rgb_{}'.format(i)])
            #         rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
             
            #         if i == len(traj_steps) -1 :
            #             discrete_action = "Done"
            #             continuous_actions = (
            #                 [0.0,0.0,0.0] # [x,y,z] deltas of each for base (recommend: don't use)
            #                 + [0.0] # [yaw] delta of the rotation in degrees
            #                 + [0.0,0.0,0.0] # [x,y,z] deltas of each for end-effector
            #             )
            #         else:
            #             discrete_action = next_discrete_actions[i]
            #             continuous_actions = (
            #                 traj_json_dict['steps'][0]['delta_global_state_body'][:3] # [x,y,z] deltas of each for base (recommend: don't use)
            #                 + [traj_json_dict['steps'][0]['delta_global_state_body'][-1]] # [yaw] delta of the rotation in degrees
            #                 + traj_json_dict['steps'][0]['delta_global_state_ee'][:3] # [x,y,z] deltas of each for end-effector
            #             )
                    
            #         step_arr.append(scene)
            #         step_arr.append(nl_command)
            #         step_arr.append(rgb)
            #         step_arr.append(discrete_action)
            #         step_arr.append(continuous_actions)
                
            #         traj_arr.append(step_arr)

            #     final_arr.append(traj_arr)
            
            # chunk_size = 100
            # num_chunks = len(final_arr) // chunk_size + (1 if len(final_arr) % chunk_size != 0 else 0)
            # breakpoint()
            # # Loop through and save each chunk
            # for i in range(num_chunks):
            #     start_index = i * chunk_size
            #     end_index = start_index + chunk_size
            #     chunk = final_arr[start_index:end_index]
            #     with open(f'/mnt/ahmed/lambda_dataset_chunk_{i}.pkl', 'wb') as file:
            #         pickle.dump(chunk, file)
            # breakpoint()

######################################################################################

        elif split_style == 'diversity_ablation':
            assert(diversity_scenes < len(self.scene_to_keys.keys()), "Error: number of train scenes for diversity ablations cannot be {}".format(len(self.scene_to_keys.keys())))

            ordered_scenes = []; ordered_trajs = []
            
            for scene, traj in self.scene_to_keys.items():

                ordered_scenes.append(scene)
                ordered_trajs.append(len(traj))
            
            ordered_index = sorted(range(0, len(ordered_trajs)), key = lambda x: ordered_trajs[x])

            ordered_trajs = list(sorted(ordered_trajs))
            ordered_scenes = [ordered_scenes[i] for i in ordered_index]

            print('EVAL SCENE: {} has {} trajectories'.format(ordered_scenes[-1], ordered_trajs[-1]))
            val_keys = self.scene_to_keys[ordered_scenes[-1]]
            other_scenes = list(reversed(ordered_scenes[:-1]))
            other_trajs = list(reversed(ordered_trajs[:-1]))


            num_per_scene = int(max_trajectories/diversity_scenes)
            train_keys = []

            for i in range(diversity_scenes):
                train_keys += random.sample(self.scene_to_keys[other_scenes[i]], num_per_scene)
            
            if len(train_keys) < max_trajectories:

                random_scene = random.sample(other_scenes[:diversity_scenes], 1)[0]
                train_keys += random.sample(self.scene_to_keys[random_scene], max_trajectories-len(train_keys))

        elif split_style == 'low_high_scene':
            with open('lanmp_dataloader/div_runs.json', 'r') as file:
                div_runs = json.load(file)
            train_keys, val_keys = low_high_scene(DATASET_PATH, div_runs['train_envs'], div_runs['test_env'])

        elif split_style == 'cluster':
            train_keys, val_keys = cluster(DATASET_PATH, low_div=low_div)


        if 'attribute_limits.json' not in os.listdir('./lanmp_dataloader'):
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = self.determine_min_max_range([train_keys, val_keys, test_keys])
        else:
            with open('./lanmp_dataloader/attribute_limits.json') as f:
                attribute_limits = json.load(f)
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = attribute_limits[0], attribute_limits[1], attribute_limits[2]

        self.train_dataset = RT1Dataset(self.use_dist, train_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        self.val_dataset = RT1Dataset(self.use_dist, val_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        self.test_dataset = RT1Dataset(self.use_dist, test_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)

    def determine_min_max_range(self, data_subset_keys):

        body_pose = {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf'), 'min_z': float('inf'), 'max_z':float('-inf')}
        body_orientation = {'min_yaw': float('inf'), 'max_yaw': float('-inf')}
        end_effector_pose = {'min_x': float('inf'), 'max_x': float('-inf'), 'min_y': float('inf'), 'max_y': float('-inf'), 'min_z': float('inf'), 'max_z': float('-inf')}
        
        


        with h5py.File(DATASET_PATH, 'r') as hdf:
            for dataset_keys in data_subset_keys:

                if dataset_keys is None:
                    continue


                for i in range(len(dataset_keys)):
                    prev_body_x = None
                    prev_body_y = None
                    prev_body_z = None
                    prev_body_yaw = None
                    prev_ee_x = None
                    prev_ee_y = None
                    prev_ee_z = None

                    print('Index: {} of {}'.format(i, len(dataset_keys)))
                    traj_group = hdf[dataset_keys[i]]
                    traj_steps = list(traj_group.keys())
                    traj_steps.sort(key=sort_folders) 

                    for j in range(len(traj_steps)):

                        step_metadata = json.loads(traj_group[traj_steps[j]].attrs['metadata'])

                        body_x = step_metadata['steps'][0]['global_state_body'][0]
                        body_y = step_metadata['steps'][0]['global_state_body'][1]
                        body_z = step_metadata['steps'][0]['global_state_body'][2]
                        
                        body_yaw = step_metadata['steps'][0]['global_state_body'][3]
                        

                        ee_x = step_metadata['steps'][0]['global_state_ee'][0]
                        ee_y = step_metadata['steps'][0]['global_state_ee'][1]
                        ee_z = step_metadata['steps'][0]['global_state_ee'][2]



                        body_pose['min_x'] = min(body_pose['min_x'], body_x - prev_body_x if prev_body_x is not None else 0)
                        body_pose['max_x'] = max(body_pose['max_x'], body_x - prev_body_x if prev_body_x is not None else 0)

                        body_pose['min_y'] = min(body_pose['min_y'], body_y - prev_body_y if prev_body_y is not None else 0)
                        body_pose['max_y'] = max(body_pose['max_y'], body_y - prev_body_y if prev_body_y is not None else 0)
                        
                        body_pose['min_z'] = min(body_pose['min_z'], body_z - prev_body_z if prev_body_z is not None else 0)
                        body_pose['max_z'] = max(body_pose['max_z'], body_z - prev_body_z if prev_body_z is not None else 0)

                        body_orientation['min_yaw'] = min(body_orientation['min_yaw'], body_yaw - prev_body_yaw if prev_body_yaw is not None else 0)
                        body_orientation['max_yaw'] = max(body_orientation['max_yaw'], body_yaw - prev_body_yaw if prev_body_yaw is not None else 0)

                        end_effector_pose['min_x'] = min(end_effector_pose['min_x'], ee_x - prev_ee_x if prev_ee_x is not None else 0)
                        end_effector_pose['max_x'] = max(end_effector_pose['max_x'], ee_x - prev_ee_x if prev_ee_x is not None else 0)

                        end_effector_pose['min_y'] = min(end_effector_pose['min_y'], ee_y - prev_ee_y if prev_ee_y is not None else 0)
                        end_effector_pose['max_y'] = max(end_effector_pose['max_y'], ee_y - prev_ee_y if prev_ee_y is not None else 0)

                        end_effector_pose['min_z'] = min(end_effector_pose['min_z'], ee_z - prev_ee_z if prev_ee_z is not None else 0)
                        end_effector_pose['max_z'] = max(end_effector_pose['max_z'], ee_z - prev_ee_z if prev_ee_z is not None else 0)


                        prev_body_x = body_x
                        prev_body_y = body_y
                        prev_body_z = body_z
                        prev_body_yaw = body_yaw
                        prev_ee_x = ee_x
                        prev_ee_y = ee_y
                        prev_ee_z = ee_z 

        

        #cache the saved max and min values if already computed to save time
        attribute_limits = [body_pose, body_orientation, end_effector_pose]
        with open('./lanmp_dataloader/attribute_limits.json', 'w') as f:
            json.dump(attribute_limits, f)
        
        
        return body_pose, body_orientation, end_effector_pose

    def collate_batches(self, batch, shuffle_batch = False):
        
        
        collated_batch = []

        # merging batch elements with variable length
        for out in range(len(batch[0])):
            collated_output = []
            for idx in range(len(batch)):
                if batch[idx][out].dtype.type == np.str_:
                    collated_output.append(batch[idx][out])
                else:
                    collated_output.append(torch.from_numpy(batch[idx][out]))
            
            if batch[idx][out].dtype.type!=np.str_:
                collated_output = torch.cat(collated_output, dim=0)
            else:
                
                collated_output = np.concatenate(collated_output, axis=0)
            
            collated_batch.append(collated_output)

        #shuffling all the batched samples across the trajectories to get random order
        if shuffle_batch:
            permutation = torch.randperm(collated_batch[0].size(0))
            
            for i in range(len(collated_batch)):
                collated_batch[i] = collated_batch[i][permutation]
        
        return collated_batch
                    

class RT1Dataset(Dataset):

    def __init__(self, use_dist, data_split_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim, tokenize_action=True):

        self.use_dist = use_dist
        self.dataset_keys = data_split_keys 
        self.body_pose_lim = body_pose_lim
        self.body_orientation_lim = body_orientation_lim
        self.end_effector_pose_lim = end_effector_pose_lim
        self.num_bins = 254

        self.tokenize_action = tokenize_action

        self.hdf =  h5py.File(DATASET_PATH, 'r')
    
    def __len__(self):
        return len(self.dataset_keys)


    def make_data_discrete(self, dictionary):
        #body x, y, z coordinate
        # dictionary['body_position_deltas'][:,0] = 1 + (dictionary['body_position_deltas'][:,0] - self.body_pose_lim['min_x'])/ (self.body_pose_lim['max_x'] - self.body_pose_lim['min_x'] ) * self.num_bins
        # dictionary['body_position_deltas'][:,0] = dictionary['body_position_deltas'][:,0].astype(int)
        
        # if self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] > 0:
        #     dictionary['body_position_deltas'][:,1] = 1 + (dictionary['body_position_deltas'][:,1] - self.body_pose_lim['min_y'])/(self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] ) * self.num_bins  
        # else:
        #     dictionary['body_position_deltas'][:,1].fill(0)
        # dictionary['body_position_deltas'][:,1] = dictionary['body_position_deltas'][:,1].astype(int)
        
        # dictionary['body_position_deltas'][:,2] = 1 + (dictionary['body_position_deltas'][:,2] - self.body_pose_lim['min_z'])/ (self.body_pose_lim['max_z'] - self.body_pose_lim['min_z'] ) * self.num_bins
        # dictionary['body_position_deltas'][:,2] = dictionary['body_position_deltas'][:,2].astype(int)

        #body yaw and pitch
        dictionary['body_yaw_deltas'] = 1 + (dictionary['body_yaw_deltas'] - self.body_orientation_lim['min_yaw']) / (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) * self.num_bins
        dictionary['body_yaw_deltas'] = dictionary['body_yaw_deltas'].astype(int)

        #end effector x, y, z coordinate
        dictionary['arm_position_deltas'][:,0] = 1 + (dictionary['arm_position_deltas'][:,0] - self.end_effector_pose_lim['min_x'])/ (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,0] = dictionary['arm_position_deltas'][:,0].astype(int)

        dictionary['arm_position_deltas'][:,1] = 1 + (dictionary['arm_position_deltas'][:,1] - self.end_effector_pose_lim['min_y'])/ (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,1] = dictionary['arm_position_deltas'][:,1].astype(int)

        dictionary['arm_position_deltas'][:,2] = 1 + (dictionary['arm_position_deltas'][:,2] - self.end_effector_pose_lim['min_z'])/ (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z'] ) * self.num_bins
        dictionary['arm_position_deltas'][:,2] =  dictionary['arm_position_deltas'][:,2].astype(int)

        #find if and where episode terminates so you can fill those entries with 0s
        if 1.0 in dictionary['terminate_episode']:
            terminate_idx = np.where(np.array(dictionary['terminate_episode'])>0)[0][0]

            # dictionary['body_position_deltas'][terminate_idx:,:].fill(0)
            dictionary['body_yaw_deltas'][terminate_idx:].fill(0)
            dictionary['arm_position_deltas'][terminate_idx:,:].fill(0)

        
        return dictionary
           
    
    def detokenize_continuous_data(self, dictionary):

        if dictionary['curr_mode'] == 'stop':
            dictionary['body_yaw_delta'] = [[0.0]]
            dictionary['arm_position_deltas'] = [[0.0, 0.0, 0.0]]

        else:
            dictionary['body_yaw_delta'][0][0] = (dictionary['body_yaw_delta'][0][0] - 1) * (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) / self.num_bins + self.body_orientation_lim['min_yaw']
            dictionary['arm_position_delta'][0][0] = (dictionary['arm_position_delta'][0][0] - 1) * (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x']) / self.num_bins + self.end_effector_pose_lim['min_x']
            dictionary['arm_position_delta'][0][1] = (dictionary['arm_position_delta'][0][1] - 1) * (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y']) / self.num_bins + self.end_effector_pose_lim['min_y']
            dictionary['arm_position_delta'][0][2] = (dictionary['arm_position_delta'][0][2] - 1) * (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z']) / self.num_bins + self.end_effector_pose_lim['min_z']
        return dictionary


    def make_data_discrete_old(self, dictionary):

        if not bool(dictionary['is_terminal']):
        
            #body x, y, z coordinate
            dictionary['body_position'][0] = 1 + int( (dictionary['body_position'][0] - self.body_pose_lim['min_x'])/ (self.body_pose_lim['max_x'] - self.body_pose_lim['min_x'] ) * self.num_bins)
            
            dictionary['body_position'][1] = 1 + int( (dictionary['body_position'][1] - self.body_pose_lim['min_y'])/ (self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] ) * self.num_bins) if self.body_pose_lim['max_y'] - self.body_pose_lim['min_y'] > 0 else 0
            
            dictionary['body_position'][2] = 1 + int( (dictionary['body_position'][2] - self.body_pose_lim['min_z'])/ (self.body_pose_lim['max_z'] - self.body_pose_lim['min_z'] ) * self.num_bins)

            #body yaw and pitch
            dictionary['body_yaw'] = 1 + int( (dictionary['body_yaw'] - self.body_orientation_lim['min_yaw']) / (self.body_orientation_lim['max_yaw'] - self.body_orientation_lim['min_yaw']) * self.num_bins)

            #end effector x, y, z coordinate
            dictionary['arm_position'][0] = 1 + int( (dictionary['arm_position'][0] - self.end_effector_pose_lim['min_x'])/ (self.end_effector_pose_lim['max_x'] - self.end_effector_pose_lim['min_x'] ) * self.num_bins)
            dictionary['arm_position'][1] = 1 + int( (dictionary['arm_position'][1] - self.end_effector_pose_lim['min_y'])/ (self.end_effector_pose_lim['max_y'] - self.end_effector_pose_lim['min_y'] ) * self.num_bins)
            dictionary['arm_position'][2] = 1 + int( (dictionary['arm_position'][2] - self.end_effector_pose_lim['min_z'])/ (self.end_effector_pose_lim['max_z'] - self.end_effector_pose_lim['min_z'] ) * self.num_bins)
       
        #if terminal action is chosen, then produce 'no action' discrete value for each of the state variables
        else:
            dictionary['body_position'][0] = 0
            dictionary['body_position'][1] = 0
            dictionary['body_position'][2] = 0

            dictionary['body_yaw'] = 0

            dictionary['arm_position'][0] = 0
            dictionary['arm_position'][1] = 0
            dictionary['arm_position'][2] = 0
        

    def get_mode(self, action):        
        value = None

        # if action == 'stop':
        #     value = 0
        # elif action in set( ['LookDown', 'LookUp']):
        #     value = 5
        # elif action in set(['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']):
        #     value = 1
        # elif action in set(['PickupObject', 'ReleaseObject']):
        #     value = 4
        # elif action in set(['MoveArm', 'MoveArmBase']):
        #     value = 3
        # elif action  == 'RotateAgent':
        #     value = 2

        if action == 'stop':
            value = 0
        elif action == 'MoveAhead':
            value = 1
        elif action == 'MoveRight':
            value = 2
        elif action == 'MoveLeft':
            value = 3
        elif action == 'MoveBack':
            value = 4
        elif action == 'LookDown':
            value = 5
        elif action == 'LookUp':
            value = 6
        elif action == 'PickupObject':
            value = 7
        elif action == 'ReleaseObject':
            value = 8
        elif action == 'MoveArm':
            value = 9
        elif action == 'MoveArmBase':
            value = 10
        elif action == 'RotateAgent':
            value = 11
        
        assert(type(value)==int, 'Get Mode didn\'t return an int')
        return value
    
    def detokenize_mode(self, token):

        tokenization_dict = {0: 'stop', 1:'MoveAhead', 2:'MoveRight', 3:'MoveLeft', 4:'MoveBack', 5:'LookDown', 6:'LookUp', 7:'PickupObject', 8:'ReleaseObject', 9:'MoveArm', 10:'MoveArmBase', 11:'RotateAgent'}

        return tokenization_dict[token]
    
    def detokenize_action(self, detokenized_mode, body_yaw_delta, arm_position_delta):
        return detokenized_mode

    def __getitem__(self, idx):
        
        traj_group = self.hdf[self.dataset_keys[idx]]
        
        traj_steps = list(traj_group.keys())
        traj_steps.sort(key=sort_folders) 

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        nl_command = traj_json_dict['nl_command']


        #compute remainder in case padding of action tokens and observations needed
        padding_length = 6 - (len(traj_steps)%6) if len(traj_steps)%6 > 0 else 0
        terminate = False

        start = 0; end = min(len(traj_steps), 6)

        #return list of dictionaries with attributes required for RT1
        all_image_obs = []
        all_nl_commands = []
        all_is_terminal = []
        all_base = []
        all_body_yaw_deltas = []
        all_arm_position_deltas = []
        all_control_mode = []
        all_pad_lengths = []

        if self.use_dist:
            all_ee_obj_dists = [] #added
            all_goal_dists = [] #added2

        #build the dictionary for each trajectory. the while loop is for 1 trajectory
        while end <= len(traj_steps) and not terminate:

            '''
                mode: stop, body, yaw, manipulation, grasping, head pitch
                gripper: (x, y, z, grasp)
                body: (x, y, yaw, look up/down)
            '''
            image_obs = []
            nl_commands = []
            base = []
            body_yaw_deltas = []
            arm_position_deltas = []
            terminate_episodes = []
            control_modes = []

            if self.use_dist:
                ee_obj_dists = [] #added
                goal_dists = [] #added2

            #6 step window
            for i in range(start, end):

                #visual observation
                ith_obs = np.array(traj_group[traj_steps[i]]['rgb_{}'.format(i)])
                image_obs.append(ith_obs)

                #natural language command
                nl_commands.append(nl_command)

                current_metadata = json.loads(traj_group[traj_steps[i]].attrs['metadata'])

                if i < len(traj_steps)-1:
                    next_metadata = json.loads(traj_group[traj_steps[i+1]].attrs['metadata'])
                
                    body_yaw_delta = next_metadata['steps'][0]['global_state_body'][3] - current_metadata['steps'][0]['global_state_body'][3]
                    arm_position_delta = np.array(next_metadata['steps'][0]['global_state_ee'][:3]) - np.array(current_metadata['steps'][0]['global_state_ee'][:3])

                    #terminate episode / mode
                    terminate_episode = int(i == len(traj_steps)-1)
                    control_mode = self.get_mode(next_metadata['steps'][0]['action'])
                    if self.use_dist:
                        ee_obj_dist = next_metadata['steps'][0]['curr_ee_to_target_obj_dist'] #added
                        goal_dist = next_metadata['steps'][0]['curr_base_to_goal_dist'] #added2
                else:
                    body_yaw_delta = 0.0
                    arm_position_delta = np.array([0.0, 0.0, 0.0])

                    #is terminal / mode -- for last step
                    terminate_episode = int(i == len(traj_steps)-1)
                    control_mode = self.get_mode('stop')
                    if self.use_dist:
                        ee_obj_dist = float(-1) #added
                        goal_dist = float(-1) #added2

                body_yaw_deltas.append(body_yaw_delta)
                arm_position_deltas.append(arm_position_delta)
                terminate_episodes.append(terminate_episode)
                control_modes.append(control_mode)
                if self.use_dist:
                    ee_obj_dists.append(ee_obj_dist) #added
                    goal_dists.append(goal_dist) #added2

            #check for remainder and pad data with extra
            if end >= len(traj_steps) and padding_length > 0:
                
                for pad in range(0, padding_length):
                    
                    image_obs.append(ith_obs)
                    nl_commands.append(nl_command)

                    body_yaw_deltas.append(0.0)
                    arm_position_deltas.append(np.array([0.0, 0.0, 0.0]))
                    terminate_episodes.append(0)
                    control_modes.append(0.0)
                    if self.use_dist:
                        ee_obj_dists.append(float(-1)) #added
                        goal_dists.append(float(-1)) #added2
                
                terminate = True
            elif end >= len(traj_steps):
                terminate = True
                


            #pre-process and discretize numerical data 
            body_yaw_deltas = np.stack(body_yaw_deltas)
            arm_position_deltas = np.stack(arm_position_deltas)
            
            if self.tokenize_action:
                
                tokenized_actions = {
                    'body_yaw_deltas': body_yaw_deltas,
                    'arm_position_deltas': arm_position_deltas,
                    'terminate_episode': terminate_episodes
                }
                
                tokenized_actions = self.make_data_discrete(tokenized_actions)
                                
                body_yaw_deltas = np.expand_dims(tokenized_actions['body_yaw_deltas'], axis=1)
                
                arm_position_deltas = tokenized_actions['arm_position_deltas']
                

            all_image_obs.append(np.stack(image_obs))
            all_nl_commands.append(np.stack(nl_commands))
            all_is_terminal.append(np.stack(terminate_episodes))
            all_body_yaw_deltas.append(body_yaw_deltas)
            all_arm_position_deltas.append(arm_position_deltas)
            all_control_mode.append(np.stack(control_modes))

            if self.use_dist:
                all_ee_obj_dists.append(np.stack(ee_obj_dists)) #added
                all_goal_dists.append(np.stack(goal_dists)) #added2

            all_pad_lengths.append(0 if not end >= len(traj_steps) else padding_length)
            
            #move the window by 6
            start += 6
            end = min(end + 6, len(traj_steps))

        if self.use_dist:
            return np.stack(all_image_obs), np.stack(all_nl_commands), np.stack(all_is_terminal), np.stack(all_body_yaw_deltas), np.stack(all_arm_position_deltas), np.stack(all_control_mode), np.stack(all_ee_obj_dists), np.stack(all_goal_dists), np.stack(all_pad_lengths), #added, added2
        else:
            return np.stack(all_image_obs), np.stack(all_nl_commands), np.stack(all_is_terminal), np.stack(all_body_yaw_deltas), np.stack(all_arm_position_deltas), np.stack(all_control_mode), np.stack(all_pad_lengths)

            

# if __name__ == '__main__':

   
#     dataset_manager = DatasetManager(0, 0.8, 0.1, 0.1)

#     dataloader = DataLoader(dataset_manager.train_dataset, batch_size=3,
#                         shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)

#     val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size=2,
#                         shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)

    
    
#     for batch, sample_batch in enumerate(dataloader):
        
#         # print('BATCH {}:'.format(batch))
#         # print('Num Steps: {}'.format(sample_batch[0].shape[0]))
#         print('Batch {}: '.format(batch), sample_batch[0].shape[0])