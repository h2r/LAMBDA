import os
import zipfile
import json
from collections import defaultdict

zip_files_directory = '/users/ajaafar/scratch/lanmp_dataset/Trajectories/' 

# Dictionary to store the nl_command strings and their associated zip files
nl_commands_files = defaultdict(list)

# Temporary dictionary to hold counts of directories for comparison
dir_counts = defaultdict(int)

# Iterate over every zip file in the directory
for file_name in os.listdir(zip_files_directory):
    zip_file_path = os.path.join(zip_files_directory, file_name)
    if os.path.isfile(zip_file_path):  # Ensure it's a file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            json_file_path = 'folder_0/data_chunk_0.json'  # Adjust as needed
            try:
                with zip_ref.open(json_file_path) as json_file_ref:
                    json_data = json.load(json_file_ref)
                    nl_command = json_data.get("nl_command", "")
                    if nl_command:
                        nl_commands_files[nl_command].append(file_name)
                        # Count the number of directories for this zip file
                        dir_counts[file_name] = sum([1 for x in zip_ref.namelist() if x.endswith('/')])
            except KeyError:
                print(f"No JSON file found in {zip_file_path} at {json_file_path}")

# Identify and delete the zip file with more directories among duplicates
for nl_command, files in nl_commands_files.items():
    if len(files) > 1:  # Indicates a duplicate
        print(f'Duplicate nl_command: "{nl_command}" found in zip files: {files}')
        # Determine which file to delete based on the higher number of directories
        to_delete = max(files, key=lambda x: dir_counts[x])
        delete_path = os.path.join(zip_files_directory, to_delete)
        os.remove(delete_path)
        print(f"Deleted '{to_delete}' because it had more directories.")
    else:
        print("No duplicates found!")