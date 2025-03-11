import json
import h5py
from ai2thor.controller import Controller
import os
import openai
import numpy as np


PATH = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5" #HDF5 dataset path
with open('dic_release.json', 'r') as file:
    dic = json.load(file)

with h5py.File(PATH, 'r') as hdf_file:
    # Iterate through each trajectory group
    for trajectory_name, trajectory_group in hdf_file.items():
        if trajectory_name in dic.keys():
            if dic[trajectory_name] == 1:
                continue
        else:
            continue
        print(dic[trajectory_name])
        # print(f"Trajectory: {trajectory_name}")
        i = 0
        obj_pos = None
        # Iterate through each timestep group within the trajectory
        for timestep_name, timestep_group in trajectory_group.items():
            # print(f"  Step: {timestep_name}")

            metadata = json.loads(timestep_group.attrs['metadata'])

            cmd = metadata['nl_command']
            # action = metadata['steps'][0]['action']
            # print(action)
            # print(metadata['steps'][0]['held_objs'])
            # if action == "PickupObject":
            #     breakpoint()
            print(cmd)
            break