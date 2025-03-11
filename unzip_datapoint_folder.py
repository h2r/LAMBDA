import os
import zipfile
import pdb
# Source directory containing the zip files
source_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_23/data_23'

# Destination directory to save the extracted contents
destination_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_23_unzipped'

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.zip'):
        # Construct full paths for the zip file and destination folder
#        pdb.set_trace()
        zip_file_path = os.path.join(source_directory, filename)
        dest_folder_path = os.path.join(destination_directory, filename[:-4])  # Remove .zip extension
        
        # Create the destination folder if it doesn't exist
        os.makedirs(dest_folder_path, exist_ok=True)
        
        # Unzip the contents of the zip file into the destination folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder_path)
