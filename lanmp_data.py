import os
import json
import torch
import copy
import progressbar
import h5py
import numpy as np
import pdb

def create_dataset(
    datasets=["fractal20220817_data"],
    split="train",
    trajectory_length=6,
    batch_size=32,
    num_epochs=1
    ):
    trajectory_datasets = []
    pdb.set_trace()
    for dataset in datasets:
        #with open("/users/sjulian2/data/ajaafar/NPM-Dataset/models/main_models/alfred/data/splits/split_keys_discrete_relative_extra.json", 'r') as f:
        #with open("/users/sjulian2/data/ajaafar/NPM-Dataset/models/main_models/alfred/", 'w') as f: 
            #json.dump(split_keys_dict, f)

        #for task in progressbar.progressbar(split_keys): #task is a trajectory
        with h5py.File('/users/sjulian2/data/shared/lanmp/lanmp_dataset.hdf5', 'r') as hdf:
            pdb.set_trace()
            traj_group = hdf[task] # used to be task
            traj_steps = list(traj_group.keys())
            first_step_key = traj_steps[0]  # Get the first step key
            json_str = traj_group[first_step_key].attrs['metadata']
            traj_json_dict = json.loads(json_str)
            scene = traj_json_dict['scene']
            nl_command = traj_json_dict['nl_command']




if __name__ == "__main__":
    create_dataset()
