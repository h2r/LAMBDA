import h5py

# Path to your HDF5 file
hdf5_file_path = '/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5'

# Open the HDF5 file in read mode
with h5py.File(hdf5_file_path, 'r') as hdf_file:
    # List all groups (trajectories) in the file
    trajectories = list(hdf_file.keys())
    # Print the total number of trajectories
    print(f"Total trajectories: {len(trajectories)}")
    
    if trajectories:
        # Iterate over each trajectory to print its timesteps
        for trajectory in trajectories:
            trajectory_group = hdf_file[trajectory]
            timesteps = list(trajectory_group.keys())
            # Print the trajectory name and its timesteps count
            print(f"{trajectory}: {len(timesteps)} timesteps")
            
            # Optionally, list all timesteps for the trajectory
            # print(f"Timesteps in {trajectory}: {', '.join(timesteps)}")
            
            # Check and print the last timestep in the current trajectory
            if timesteps:
                last_timestep = sorted(timesteps)[-1]
                print(f"Last timestep in {trajectory}: {last_timestep}")
        
        # Identify and print the last trajectory processed
        last_trajectory = sorted(trajectories)[-1]
        print(f"Last trajectory processed: {last_trajectory}")
