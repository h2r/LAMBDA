import os
import json
import numpy as np
import h5py
import zipfile
from tqdm import tqdm

# Directory containing the zip files
zip_files_directory = '/users/ajaafar/data/shared/lanmp/Trajectories'
# HDF5 file to create or update
hdf5_file_path = '/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5'
# Specific trajectories to reprocess
reprocess_trajectories = ['data_16:03:52:00', 'data_16:03:52']

with h5py.File(hdf5_file_path, 'a') as hdf_file:  # Open HDF5 file with append mode
    zip_files = sorted([f for f in os.listdir(zip_files_directory) if f.endswith('.zip')])
    for zip_file_name in tqdm(zip_files, desc='Processing Zip Files'):
        trajectory_group_name = os.path.splitext(zip_file_name)[0]

        # Determine if the current trajectory needs reprocessing
        needs_reprocessing = trajectory_group_name in reprocess_trajectories
        
        # Reprocess or skip logic
        if trajectory_group_name in hdf_file and not needs_reprocessing:
            print(f"Skipping completed trajectory: {trajectory_group_name}")
            continue  # Skip this loop iteration if the trajectory is complete and not marked for reprocessing
        
        print(f"Processing trajectory: {trajectory_group_name}")
        if needs_reprocessing and trajectory_group_name in hdf_file:
            print(f"Reprocessing trajectory: {trajectory_group_name}")
            del hdf_file[trajectory_group_name]  # Remove the existing group to start fresh

        # Process trajectory
        trajectory_group = hdf_file.create_group(trajectory_group_name)
        zip_file_path = os.path.join(zip_files_directory, zip_file_name)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('temp_extract_dir')
            folders = sorted(os.listdir('temp_extract_dir'))
            for folder_name in tqdm(folders, desc=f'Folders in {zip_file_name}'):
                folder_path = os.path.join('temp_extract_dir', folder_name)
                timestep_group = trajectory_group.create_group(folder_name)

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.json'):
                        with open(file_path, 'r') as json_file:
                            json_data = json.load(json_file)
                            timestep_group.attrs['metadata'] = json.dumps(json_data)
                    elif file_name.endswith('.npy'):
                        data = np.load(file_path)
                        dataset_name = os.path.splitext(file_name)[0]
                        timestep_group.create_dataset(dataset_name, data=data, compression="gzip")

        # Cleanup the temporary directory after processing each zip file
        if os.path.exists('temp_extract_dir'):
            for root, dirs, files in os.walk('temp_extract_dir', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('temp_extract_dir')
            
        # Flush to disk after processing each trajectory
        hdf_file.flush()