import os
import zipfile
from tqdm import tqdm
import tempfile
import json
from PIL import Image
import numpy as np
import pdb
dest_folder = '/users/sjulian2/data/sjulian2/figures_paper/jpg_images/'
folder_path = '/users/sjulian2/data/sjulian2/figures_paper' 

for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if 'folder' in item:
=        folder_num = item[7:]
        for file_name in os.listdir(item_path):
            file_path = os.path.join(item_path, file_name)
            if file_name.startswith('rgb'):
                image_array = np.load(file_path)
                image = Image.fromarray(image_array, 'RGB')
                save_name = 'rgb_' + str(folder_num) + '.jpg'
                image.save(dest_folder + save_name)
