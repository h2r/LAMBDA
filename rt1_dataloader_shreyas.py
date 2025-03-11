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

DATASET_PATH = '/oscar/data/stellex/shared/lanmp/lanmp_dataset.hdf5'

'''
train_keys, val_keys, test_keys = split_data(self.args.data, splits['train'], splits['val'], splits['test'])
'''

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
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
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
    
    with open('./scene_to_keys.json', 'w') as f:
        json.dump(scene_to_keys, f)

    return scene_to_keys



def sort_folders(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

class DatasetManager(object):

    '''
    NOTE: kwargs should contain a dictionary with keys {'train_split' : x, 'val_split': y, 'test_split':z} where x+y+z = 1
    '''
    def __init__(self, val_scene=1, train_split=0.8, val_split=0.1, test_split=0.1):

        assert( train_split + val_split + test_split == 1.0, 'Error: train, val and test split do not sum to 1.0')

        
        #train_keys, val_keys, test_keys = split_data(DATASET_PATH, train_split, val_split, test_split)
        if 'scene_to_keys.json' not in os.listdir('.'):
            self.scene_to_keys = split_by_scene(DATASET_PATH)
        else:
            with open('./scene_to_keys.json') as f:
                self.scene_to_keys = json.load(f)
             

        self.scenes = list(sorted(list(self.scene_to_keys.keys())))

        assert( val_scene < len(self.scenes), "Error: input scene is out of index space")

        train_keys = []
        for x in range(0, len(self.scenes)):
            if x!=val_scene:
                train_keys += self.scene_to_keys[self.scenes[x]]  

        val_keys = self.scene_to_keys[self.scenes[val_scene]]
        test_keys = None



        if 'attribute_limits.json' not in os.listdir('.'):
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = self.determine_min_max_range([train_keys, val_keys, test_keys])
        else:

            with open('./attribute_limits.json') as f:
                attribute_limits = json.load(f)
            body_pose_lim, body_orientation_lim, end_effector_pose_lim = attribute_limits[0], attribute_limits[1], attribute_limits[2]

        self.train_dataset = RT1Dataset(train_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        self.val_dataset = RT1Dataset(val_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)
        # self.test_dataset = RT1Dataset(test_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim)

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

                        body_x = step_metadata['steps'][0]['state_body'][0]
                        body_y = step_metadata['steps'][0]['state_body'][1]
                        body_z = step_metadata['steps'][0]['state_body'][2]
                        
                        body_yaw = step_metadata['steps'][0]['state_body'][3]
                        

                        ee_x = step_metadata['steps'][0]['state_ee'][0]
                        ee_y = step_metadata['steps'][0]['state_ee'][1]
                        ee_z = step_metadata['steps'][0]['state_ee'][2]



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
        with open('./attribute_limits.json', 'w') as f:
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

    

    def __init__(self, data_split_keys, body_pose_lim, body_orientation_lim, end_effector_pose_lim):

        #stores the keys in the dataset for the appropriate split (train, validation or test)
        self.dataset_keys = data_split_keys
        self.body_pose_lim = body_pose_lim
        self.body_orientation_lim = body_orientation_lim
        self.end_effector_pose_lim = end_effector_pose_lim
        self.num_bins = 254

        self.hdf =  h5py.File(DATASET_PATH, 'r')
    
    def __len__(self):
        return len(self.dataset_keys)

    
    def make_data_discrete(self, dictionary):

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
        
    def get_head_pitch(self, action):

        value = 0

        if action == 'LookDown':
            value = 1
        elif action == 'LookUp':
            value = 2

        return value
    
    def get_mode(self, action):

        #mode: (0) stop, (1) body, (2) yaw,  (3) manipulation, (4) grasping, (5) head pitch
        
        value = None

        if action == 'stop':
            value = 0
        elif action in set( ['LookDown', 'LookUp']):
            value = 5
        elif action in set(['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']):
            value = 1
        elif action in set(['PickupObject', 'ReleaseObject']):
            value = 4
        elif action in set(['MoveArm', 'MoveArmBase']):
            value = 3
        elif action  == 'RotateAgent':
            value = 2
        
        assert(type(value)==int, 'Get Mode didn\'t return an int')
        return value
    
    def get_pickup_release(self, action):

        if action == 'PickupObject':
            value = 1
        elif action == 'ReleaseObject':
            value = 2
        else: 
            value = 0
        
        return value

    def __getitem__(self, idx):
        
        
        traj_group = self.hdf[self.dataset_keys[idx]]
        
        traj_steps = list(traj_group.keys())
        traj_steps.sort(key=sort_folders) 

        #extract the NL command
        json_str = traj_group[traj_steps[0]].attrs['metadata']
        traj_json_dict = json.loads(json_str)
        nl_command = traj_json_dict['nl_command']

        start = 0; end = min(len(traj_steps), 6)

        #return list of dictionaries with attributes required for RT1
        all_image_obs = []
        all_nl_commands = []
        all_is_terminal = []
        all_pickup_release = []
        all_body_position = []
        all_body_yaw = []
        all_body_pitch = []
        all_arm_position = []
        all_mode = []



        #build the dictionary for each sequence
        while end < len(traj_steps):

            '''
                mode: stop, body, yaw, manipulation, grasping, head pitch
                gripper: (x, y, z, grasp)
                body: (x, y, yaw, look up/down)
            '''
            image_obs = []

            for i in range(start, end):
                ith_obs = np.array(traj_group[traj_steps[i]]['rgb_{}'.format(i)])
                
                image_obs.append(ith_obs)
            
            image_obs = np.stack(image_obs)
            
            
            
            before_end_step_metadata = json.loads(traj_group[traj_steps[end-1]].attrs['metadata'])
            end_step_metadata = json.loads(traj_group[traj_steps[end]].attrs['metadata'])                



            dictionary = {
                'observation': image_obs,
                'nl_command': nl_command, #DONE
                'is_terminal': int(end_step_metadata['steps'][0]['action']=='stop'), #DONE
                'pickup_release': self.get_pickup_release(end_step_metadata['steps'][0]['action']), #DONE
                'body_position': np.array(end_step_metadata['steps'][0]['state_body'][:3])-np.array(before_end_step_metadata['steps'][0]['state_body'][:3]), #DONE 
                'body_yaw': end_step_metadata['steps'][0]['state_body'][3] - before_end_step_metadata['steps'][0]['state_body'][3], #DONE
                'body_pitch': self.get_head_pitch(end_step_metadata['steps'][0]['action']), #DONE
                'arm_position': np.array(end_step_metadata['steps'][0]['state_ee'][:3]) - np.array(before_end_step_metadata['steps'][0]['state_ee'][:3]), #DONE
                'mode': self.get_mode(end_step_metadata['steps'][0]['action']) #DONE
            }

            #pre-process the data dictonary
            self.make_data_discrete(dictionary)


            all_image_obs.append(dictionary['observation'])
            all_nl_commands.append(dictionary['nl_command'])
            all_is_terminal.append(dictionary['is_terminal'])
            all_pickup_release.append(dictionary['pickup_release'])
            all_body_position.append(dictionary['body_position'])
            all_body_yaw.append(dictionary['body_yaw'])
            all_body_pitch.append(dictionary['body_pitch'])
            all_arm_position.append(dictionary['arm_position'])
            all_mode.append(dictionary['mode'])
            

            start += 1
            end += 1

        #add the terminal 'stop' step
        all_image_obs.append(dictionary['observation'])
        all_nl_commands.append(dictionary['nl_command'])
        all_is_terminal.append(1)
        all_pickup_release.append(0)
        all_body_position.append([0,0,0])
        all_body_yaw.append(0)
        all_body_pitch.append(0)
        all_arm_position.append([0,0,0])
        all_mode.append(0)



        
       
        return np.stack(all_image_obs), np.stack(all_nl_commands), np.stack(all_is_terminal), np.stack(all_pickup_release), np.stack(all_body_position), np.stack(all_body_yaw), np.stack(all_body_pitch), np.stack(all_arm_position), np.stack(all_mode)


if __name__ == '__main__':

   

    # with h5py.File(DATASET_PATH, 'r') as hdf:

    #     pdb.set_trace()
    #     dataset_keys,_, _ = split_data(DATASET_PATH, 0.8, 0.1, 0.1)
    #     traj_group = hdf[dataset_keys[0]]
        
    #     traj_steps = list(traj_group.keys())
    #     traj_steps.sort(key=sort_folders)

    #     #extract the NL command
    #     json_str = traj_group[traj_steps[0]].attrs['metadata']
    #     traj_json_dict = json.loads(json_str)
    #     nl_command = traj_json_dict['nl_command']

    #     start = 0; end = min(len(traj_steps), 6)

    #     #return list of dictionaries with attributes required for RT1
    #     data_sequence = []

    '''
    for i in range(0,5):
        dataset_manager = DatasetManager(i, 0.7, 0.2, 0.1)

        dataloader = DataLoader(dataset_manager.train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)
        
        for batch, sample_batch in enumerate(dataloader):
            print('BATCH {}:'.format(batch))
            print('Num Steps: {}'.format(sample_batch[0].shape[0]))

    '''
    
    dataset_manager = DatasetManager(1, 0.7, 0.2, 0.1)

    dataloader = DataLoader(dataset_manager.train_dataset, batch_size=3,
                        shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches)

    pdb.set_trace()
    for batch, sample_batch in enumerate(dataloader):
        
        print('BATCH {}:'.format(batch))
        print('Num Steps: {}'.format(sample_batch[0].shape[0]))

    

    '''
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

dataloader = DataLoader(face_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    '''


