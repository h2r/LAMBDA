import pandas as pd
import zipfile
import os
import json

# Step 1: Read and filter the CSV file
csv_file_path = 'cmd_scene_dic.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

scene_num = "8_1"

# Filter the DataFrame based on the 'scene' column
filtered_df = df[df['scene'] == f'FloorPlan_Train{scene_num}'].copy()

# Initialize the 'exists' column
filtered_df['exists'] = ''

# Step 2: Collect 'nl_command's from the JSON files inside the ZIP files
zip_folder_path = f'/mnt/ahmed/new_sim_data/files/{scene_num}'  # Replace with the path to your ZIP files
zip_files = [f for f in os.listdir(zip_folder_path) if f.endswith('.zip')]

commands_in_zip = set()
cmds = []
# repeats = []
for zip_filename in zip_files:
    zip_path = os.path.join(zip_folder_path, zip_filename)
    # if zip_path == f"/mnt/ahmed/new_sim_data/files/{scene_num}/data_17:46:02.zip":
    #     # print('###########################')
    #     # breakpoint()
    #     break
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Assume each ZIP contains one JSON file
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.json'):
                with zip_ref.open(file_info) as json_file:
                    data = json.load(json_file)
                    nl_command = data.get('nl_command')
                    # if nl_command in cmds:
                    #     repeats.append(zip_path)
                    # if nl_command == "Pick up the basketball next to the baseball bat and put it in the bin.":
                    #     print(zip_path)
                    cmds.append(nl_command)
                    if nl_command == "Pick up the apple by the laptop and put it on the TV table.":
                        breakpoint()
                        print('hi')
                    break
                    if nl_command:
                        commands_in_zip.add(nl_command)
    # No extraction to disk; processed in-memory
# print(repeats)
# diff = list(set(filtered_df['cmd'].tolist()) - set(cmds))
diff = list(set(cmds) ^ set(filtered_df['cmd'].tolist()))
breakpoint()

# Step 3: Update the 'exists' column based on matching commands
filtered_df.loc[filtered_df['cmd'].isin(commands_in_zip), 'exists'] = 'yes'

# Step 4: Save the updated DataFrame to a new CSV file
output_csv_path = 'cmd_scene_dic_new.csv'
filtered_df.to_csv(output_csv_path, index=False)
