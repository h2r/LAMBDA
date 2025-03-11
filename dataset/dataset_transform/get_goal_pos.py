import json
import h5py
from ai2thor.controller import Controller
import os
import openai
import numpy as np


PATH = "/users/ajaafar/data/shared/lanmp/sim_dataset.hdf5" #HDF5 dataset path
dic = {}
with h5py.File(PATH, 'r') as hdf_file:
    # Iterate through each trajectory group
    for trajectory_name, trajectory_group in hdf_file.items():
        print(f"Trajectory: {trajectory_name}")
        continue
        # obj_pos = None
        # Iterate through each timestep group within the trajectory
        # for timestep_name, timestep_group in trajectory_group.items():
        for timestep_name, timestep_group in reversed(list(trajectory_group.items())):
            # print(f"  Step: {timestep_name}")

            metadata = json.loads(timestep_group.attrs['metadata'])

            cmd = metadata['nl_command']
            action = metadata['steps'][0]['action']
            if trajectory_name == "data_20:07:12":
                print(cmd)
                breakpoint()
            # if action == "ReleaseObject":
                # print(metadata['steps'][0]['held_objs'])
                # print(metadata['steps'][0]['state_body'])
                # print(action)
                # print(cmd)
                # breakpoint()

                # dic[trajectory_name] = metadata['steps'][0]['state_body']
                # break

# with open("traj_goal_pos.json", "w") as json_file:
#     json.dump(dic, json_file, indent=4)