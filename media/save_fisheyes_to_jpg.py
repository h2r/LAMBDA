import cv2
import os
import zipfile
from tqdm import tqdm
import tempfile
import json
from PIL import Image
import numpy as np
import pdb

#directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_33_unzipped'
directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_23_unzipped'
#unstitched_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_33_unstitched'
unstitched_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_23_unstitched'
#stitched_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_33_stitched'
stitched_directory = '/users/sjulian2/data/sjulian2/figures_paper/real_imgs/data_23_stitched'

# Iterate over all files in the source directory
for folder in os.listdir(directory):
    folder_num = folder[7:]
    folder_path = os.path.join(directory, folder)
    found_right = 0
    found_left = 0
    path_right = ''
    path_left = ''
    
    for file_name in os.listdir(folder_path):
        if file_name.startswith('left_fisheye_image_') and found_left == 0:
            path_left = os.path.join(folder_path, file_name)
            found_left = 1
        elif file_name.startswith('right_fisheye_image_') and found_right == 0:
            path_right = os.path.join(folder_path, file_name)
            found_right = 1
    
    # convert each to a jpg
    image_array = np.load(path_right)
    image = Image.fromarray(image_array, 'RGB')
    save_name = '/right_fisheye_image_' + str(folder_num) + '.jpg'
    save_directory_right = unstitched_directory + save_name
    image.save(unstitched_directory + save_name)

    image_array = np.load(path_left)
    image = Image.fromarray(image_array, 'RGB')
    save_name = '/left_fisheye_image_' + str(folder_num) + '.jpg'
    save_directory_left = unstitched_directory + save_name
    image.save(save_directory_left)
    #pdb.set_trace()
    # stitch them and save to stitched directory
    # update: this stitching code almost always fails with the spot fisheye images 
    '''image_paths = [save_directory_right, save_directory_left]
    imgs = [] 
  
    for i in range(len(image_paths)): 
        imgs.append(cv2.imread(image_paths[i])) 
        
    # showing the original pictures 
    #cv2.imshow('1',imgs[0]) 
    #cv2.imshow('2',imgs[1]) 
    
    stitchy=cv2.Stitcher.create() 
    (status,output)=stitchy.stitch(imgs) 
    
    if status != cv2.STITCHER_OK: 
        print('ERROR: could not stitch images')
        print('folder num '+ folder_num)
    else:  
        print('SUCCESS: stitch worked!') 
        stitched_save_directory = stitched_directory + '/stitched_image_{}.jpg'.format(folder_num)
        cv2.imwrite(stitched_save_directory, output)
        cv2.waitKey(0)'''
    #pdb.set_trace()
    # final output 
    #cv2.imshow('final result',output)
    
