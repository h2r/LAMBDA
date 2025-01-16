import os
import json
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
import torch 
import numpy as np  
import argparse
import os 
from tqdm import tqdm 
import random
import pickle 
from transformers import GPT2Tokenizer
import configparser 
from transformers import AutoTokenizer
import random 
from copy import copy
# from . import tokenize_lanmp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motionglot import tokenize_lanmp
from itertools import chain


np.random.seed(47)

def load_img_tokenizer():

    img_model = tokenize_lanmp.Net(codebook_dim=64, codebook_size=2048).cuda()
    ckpt = torch.load(args.image_token_model, map_location='cuda')
    img_model.load_state_dict(ckpt  )
    img_model.eval()

    return img_model.to("cuda")

def get_max_min_ranges():
    all_files = os.listdir(args.dataset_path )
    
    all_yaw = []
    all_x = []
    all_y = []
    all_z = []

    all_id0 , all_id1 , all_id2 , all_id3, all_id4, all_id5, all_id6 = [],[], [], [], [], [], []

    all_id0 , all_id1 , all_id2 , all_id3, all_id4, all_id5, all_id6 = [],[], [], [], [], [], []

    for file in tqdm( all_files):
        if file.split(".")[1] == "pkl":
            path =os.path.join( args.dataset_path , file )
            data =  tokenize_lanmp.get_data(path )

            num_traj = len(data)
            for i in range(num_traj):
                num_steps = len( data[i] )
                traj_command = []
                for j in range(num_steps):
                    val = data[i][j][3]
                    if val == "MoveArm":
                        x,y,z = data[i][j][4][ 4] ,  data[i][j][4][ 5] ,  data[i][j][4][ 6]
                        all_x.append(x)
                        all_y.append(y)
                        all_z.append(z)

                    if val == "MoveArmBase":
                        id0, id1, id2, id3,id4,id5,id6 =  data[i][j][4][0] , data[i][j][4][1] ,data[i][j][4][2] , data[i][j][4][3] ,data[i][j][4][4],data[i][j][4][5] , data[i][j][4][6]
                        all_id0.append( id0)
                        all_id1.append( id1)
                        all_id2.append( id2)
                        all_id3.append( id3)
                        all_id4.append( id4)
                        all_id5.append( id5)
                        all_id6.append( id6)

                    if  val == "RotateAgent":
                        yaw = data[i][j][4][ 3]
                        all_yaw.append(yaw)
        
    x_min, x_max = np.min(all_x) , np.max(all_x)
    y_min, y_max = np.min(all_y) , np.max(all_y)
    z_min, z_max = np.min(all_z) , np.max(all_z)
    yaw_min, yaw_max = np.min(all_yaw) , np.max(all_yaw)
    
    id0_min, id0_max = np.min(all_id0) , np.max(all_id0)
    id1_min, id1_max = np.min(all_id1) , np.max(all_id1)
    id2_min, id2_max = np.min(all_id2) , np.max(all_id2)
    id3_min, id3_max = np.min(all_id3) , np.max(all_id3)
    id4_min, id4_max = np.min(all_id4) , np.max(all_id4)
    id5_min, id5_max = np.min(all_id5) , np.max(all_id5)
    id6_min, id6_max = np.min(all_id6) , np.max(all_id6)
    return [yaw_min , yaw_max] , [ x_min, x_max ] , [ y_min, y_max  ] ,[z_min, z_max] , [ id0_min, id0_max ] , [id1_min, id1_max] , [id2_min, id2_max] , [id3_min, id3_max] , [id4_min, id4_max] , [id5_min, id5_max] ,[id6_min, id6_max] 

def get_bin_index(x, min_val, max_val, num_bins):
    """
    Returns the bin index (0-based) for a continuous value x that lies
    between min_val and max_val (inclusive) given num_bins bins.
    """
    # Avoid division by zero or invalid input
    if max_val <= min_val or num_bins <= 0:
        raise ValueError("Invalid range or number of bins.")

    # Calculate the bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Compute the raw bin index (floating point)
    raw_index = (x - min_val) / bin_width
    
    # Convert to integer index by flooring
    index = int(raw_index)
    
    # Clamp index within valid range [0, num_bins-1]
    if index < 0:
        index = 0
    elif index >= num_bins:
        index = num_bins - 1
        
    return index

def load_traj():
    all_files = os.listdir(args.dataset_path )
    yaw, eef_x, eef_y, eef_z , id0_range, id1_range, id2_range,id3_range, id4_range,id5_range,id6_range = get_max_min_ranges()
    yaw, eef_x, eef_y, eef_z , id0_range, id1_range, id2_range,id3_range, id4_range,id5_range,id6_range = get_max_min_ranges()

    all_text = []
    all_imgs = []
    all_lang_instruct = []
    all_scenes = []

    for file in tqdm( all_files):
        if file.split(".")[1] == "pkl":
            path =os.path.join( args.dataset_path , file )
            data =  tokenize_lanmp.get_data(path )
            num_traj = len(data)
            for i in range(num_traj):
                num_steps = len( data[i] )
                traj_text = []
                traj_img = []
                lang_instruct = []
                scene = data[i][0][0]
                for j in range(num_steps):
                    
                    val = data[i][j][3]
                    if val == "MoveArm":
                        x,y,z = data[i][j][4][ 4] ,  data[i][j][4][ 5] ,  data[i][j][4][ 6]
                        x_idx = get_bin_index( x , eef_x[0] , eef_x[1] , args.num_bins )
                        y_idx = get_bin_index( y , eef_y[0] , eef_y[1] , args.num_bins )
                        z_idx = get_bin_index( z , eef_z[0] , eef_z[1] , args.num_bins )

                        string_val = "MoveArm " + str(x_idx) +" , " + str(y_idx) + " , " + str(z_idx)
                    elif val == "MoveArmBase":
                        id0, id1, id2, id3,id4,id5,id6 =  data[i][j][4][0] , data[i][j][4][1] ,data[i][j][4][2] , data[i][j][4][3] ,data[i][j][4][4],data[i][j][4][5] , data[i][j][4][6]
                        id0_idx = 0#get_bin_index( id0 , id0_range[0] , id0_range[1] , args.num_bins)
                        id1_idx = 0#get_bin_index(id1  , id1_range[0] , id1_range[1], args.num_bins )
                        id2_idx = 0#get_bin_index(id2 , id2_range[0] , id2_range[1] , args.num_bins)
                        id3_idx = 0#get_bin_index(id3 , id3_range[0] , id3_range[1] , args.num_bins)
                        id4_idx = get_bin_index(id4 , id4_range[0] , id4_range[1] , args.num_bins)
                        id5_idx = get_bin_index(id5  , id5_range[0] , id5_range[1] , args.num_bins )
                        id6_idx = get_bin_index(id6  , id6_range[0] , id6_range[1] , args.num_bins )
                        string_val = "MoveArmBase " + str( id5_idx )
                    elif val == "RotateAgent":
                        yaw1 = data[i][j][4][ 3]
                        yaw_idx = get_bin_index( yaw1 , yaw[0] , yaw[1], args.num_bins )

                        string_val = "RotateAgent " + str( yaw_idx )
                    elif val == "MoveArmBase":
                        id0, id1, id2, id3,id4,id5,id6 =  data[i][j][4][0] , data[i][j][4][1] ,data[i][j][4][2] , data[i][j][4][3] ,data[i][j][4][4],data[i][j][4][5] , data[i][j][4][6]

                        id0_idx = 0#get_bin_index( id0 , id0_range[0] , id0_range[1] , args.num_bins)
                        id1_idx = 0#get_bin_index(id1  , id1_range[0] , id1_range[1], args.num_bins )
                        id2_idx = 0#get_bin_index(id2 , id2_range[0] , id2_range[1] , args.num_bins)
                        id3_idx = 0#get_bin_index(id3 , id3_range[0] , id3_range[1] , args.num_bins)
                        id4_idx = get_bin_index(id4 , id4_range[0] , id4_range[1] , args.num_bins)
                        id5_idx = get_bin_index(id5  , id5_range[0] , id5_range[1] , args.num_bins )
                        id6_idx = get_bin_index(id6  , id6_range[0] , id6_range[1] , args.num_bins )

                        string_val = "MoveArmBase" + str( id0_idx ) + " , " + str( id1_idx ) + " , " +  str( id2_idx ) + " , " +  str( id3_idx ) + " , " + str( id4_idx ) + " , " + str( id5_idx ) + " , " + str( id6_idx ) 
                    else:
                        string_val = str(val)
                    
                    traj_text.append( string_val)
                    traj_img.append( torch.tensor(data[i][j][2] ).unsqueeze(0)  )
                    lang_instruct.append(data[i][j][1]  )
                
                all_text.append(traj_text)
                all_imgs.append( traj_img )
                all_lang_instruct.append(lang_instruct)
                all_scenes.append(scene)
                
    with open('all_imgs.pkl', 'wb') as file:
        pickle.dump(all_imgs, file)
    with open('all_text.pkl', 'wb') as file:
        pickle.dump(all_text, file)
    with open('all_lang_instruct.pkl', 'wb') as file:
        pickle.dump(all_lang_instruct, file)
    with open('all_scenes.pkl', 'wb') as file:
        pickle.dump(all_scenes, file)


    return all_imgs , all_text , all_lang_instruct, all_scenes

def get_image_strings( image_idx ):
    num_seqs = len(image_idx )
    image_string = []

    for i in range(num_seqs):
        
        data = image_idx[i]
        if data == []:
            string_out = []
        else:

            data2 = ''.join( [f'<img_id_{int(j)}>' for j in image_idx[i] ] )
            # ids, mask = tokenize_string(data2)
            string_out = data2

        image_string.append( string_out )

    return image_string  

def tokenize_string( data ):
    tokenization_output=  tokenizer(data, return_tensors="pt", 
    padding="max_length", max_length= VOCAB['block_size']  )
    
    return tokenization_output['input_ids']  , tokenization_output['attention_mask']

def get_unified_strings( img_tokens, lang_instruct,  text_output ):


    img_string = get_image_strings( [ img_tokens] )

    unified_string = "give action command: " + lang_instruct + img_string[0] + VOCAB['eop_char']  +  VOCAB['toa_char']  + text_output + VOCAB['eog_char'] 
    ids, mask = tokenize_string( unified_string )

    return ids , mask 

def tokenize_images():

    num_traj = len(all_img)
    all_tokens_ids = []
    all_masks = []

    for i in tqdm( range(num_traj)):
        num_steps = len( all_img[i] )
        tokens_ids_per_traj = []
        mask_per_traj = []
        for j in range( num_steps ):
            img = (all_img[i][j].permute(0,3,1,2)/255.0).to("cuda")
            _, idx, _= img_token_model(img  )  
            a,b,c = idx.shape 
            token_idx = idx.squeeze(0).view(  b*c  )
            ids, mask = get_unified_strings( token_idx , all_lang[i][j] , text_strings[i][j]  )
            tokens_ids_per_traj.append( ids )
            mask_per_traj.append( mask )
        all_tokens_ids.append( torch.cat( tokens_ids_per_traj , dim =0 )  )
        all_masks.append( torch.cat( mask_per_traj , dim =0 )  )
    
    with open('all_tokens_ids.pkl', 'wb') as file:
        pickle.dump(all_tokens_ids, file)
    with open('all_masks.pkl', 'wb') as file:
        pickle.dump(all_masks, file)

    return all_tokens_ids , all_masks

def define_vocabulary():
    
    num_image_tokens = 2048

    tokenizer = AutoTokenizer.from_pretrained( args.model_type )

    ### add motion tokens 
    tokenizer.add_tokens(
            [f"<img_id_{i}>" for i in range(num_image_tokens  ) ] )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = True
    
    ## add meta tokens

    tokenizer.add_tokens( "<eop>" , special_tokens=True ) ## end of prompt token
    tokenizer.add_tokens( "<toa>" , special_tokens=True ) ## translate to human token
    tokenizer.add_tokens( "<eog>" , special_tokens=True ) ## end of generation


    VOCAB = {}

    VOCAB['language_id_range'] = [ 0, tokenizer.vocab_size ]
    VOCAB['eos_id'] = tokenizer.eos_token # - 1 

    VOCAB['total_vocab_size'] = len(tokenizer) #VOCAB['motion_id_range'][1] +1 
    VOCAB['block_size'] = 1024
    VOCAB['eop_id'] = tokenizer.convert_tokens_to_ids("<eop>" )
    VOCAB['eop_char'] = "<eop>"
    VOCAB['eog_char'] = "<eog>"
    VOCAB['toa_char'] = "<toa>"

    return tokenizer , VOCAB 


def split_by_scene(all_token_ids, all_masks, all_scenes, all_lang):
    #mapping which ids are relevant to specific scenes
    scene_to_ids = {}
    scene_to_masks = {}
    scene_to_langs = {}

    for token_id, mask, scene, lang in zip(all_token_ids, all_masks, all_scenes, all_lang):
        if scene not in scene_to_ids:
            scene_to_ids[scene] = []
        scene_to_ids[scene].append(token_id.tolist())
        
        if scene not in scene_to_masks:
            scene_to_masks[scene] = []
        scene_to_masks[scene].append(mask.tolist())

        if scene not in scene_to_langs:
            scene_to_langs[scene] = []
        scene_to_langs[scene].append(lang)
    
    with open('scene_to_ids.json', 'w') as f:
        json.dump(scene_to_ids, f)
    with open('scene_to_masks.json', 'w') as f:
        json.dump(scene_to_masks, f)
    with open('scene_to_langs.json', 'w') as f:
        json.dump(scene_to_langs, f)

    return scene_to_ids, scene_to_masks, scene_to_langs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--image_token_model", help="point to the file with the name of all files", default= "img_token_model.pth" , type=str)
    parser.add_argument("--processed_img_data_path" , help="path to the pre processed dataset" , default= "lambda_dataset_imgs.pt" )
    parser.add_argument("--dataset_path", help="point to the file with the name of all files", default= "pickle_data" , type=str)
    parser.add_argument("--split_type", default = None, choices =  ['scene_gen', 'task_gen'],)
    parser.add_argument("--num_bins", help="point to the file with the name of all files", default= 256 , type=int )
    parser.add_argument("--test_scene", help="held out test scene for scene generalization", default= 0 , type=int )
    parser.add_argument("--model_type", help="path to folder with prompts ", default= "openai-community/gpt2"  ,type =str ) # "openai-community/gpt2"

    args = parser.parse_args()
    train_split = 0.8
    val_split = 0.1

    tokenizer , VOCAB  = define_vocabulary()
    
    img_token_model = load_img_tokenizer()


    with open('all_imgs.pkl', 'rb') as file:
        all_img = pickle.load(file)
    with open('all_text.pkl', 'rb') as file:
        all_text = pickle.load(file)
    with open('all_lang_instruct.pkl', 'rb') as file:
        all_lang = pickle.load(file)
    with open('all_scenes.pkl', 'rb') as file:
        all_scenes = pickle.load(file)
    with open('all_masks.pkl', 'rb') as file:
        all_masks = pickle.load(file)
    with open('all_tokens_ids.pkl', 'rb') as file:
        all_token_ids = pickle.load(file)

    
    # all_img, text_strings , all_lang, all_scenes = load_traj()

    # all_token_ids, all_masks = tokenize_images()

    print(len(all_token_ids) , len(all_masks))

    num_traj = len(all_token_ids)
    assert len(all_token_ids) == len(all_masks)

    # all_train, all_valid= [] , [] 
    # all_train_mask, all_valid_mask = [], []


    # if os.path.exists("scene_to_ids.json") and os.path.exists("scene_to_masks.json") and os.path.exists("scene_to_langs.json"):
    #     with open("scene_to_ids.json", "r") as file:
    #         scene_to_ids = json.load(file)
    #     with open("scene_to_masks.json", "r") as file:
    #         scene_to_masks = json.load(file)
    #     with open("scene_to_langs.json", "r") as file:
    #         scene_to_langs = json.load(file)
    # else:
    #     scene_to_ids, scene_to_masks, scene_to_langs = split_by_scene(all_token_ids, all_masks, all_scenes, all_lang)

    # scenes = list(sorted(list(scene_to_ids.keys())))
    
    train_langs = []
    val_langs = []

    train_data = []
    val_data = []
    # test_data = []

    train_mask = []
    val_mask = []
    # test_mask = []

    if args.split_type == 'task_gen':
        # for scene in scenes:   
        #     scene_ids = copy(scene_to_ids[scene])
        #     mask_ids = copy(scene_to_masks[scene])
            
        #     # indices = np.arange(len(scene_ids))
        #     # np.random.shuffle(indices)
        #     # scene_ids = scene_ids[indices]
        #     # mask_ids = mask_ids[indices]

        #     split_idx = int(len(scene_ids)*(train_split))
        #     split_idx2 = int(len(scene_ids)*(train_split+val_split))
            
        #     train_data += scene_ids[:split_idx]
        #     val_data += scene_ids[split_idx:split_idx2]
        #     # test_data += scene_ids[split_idx2:]

        #     train_mask += mask_ids[:split_idx]
        #     val_mask += mask_ids[split_idx:split_idx2]
        #     # test_mask += mask_ids[split_idx2:]
        

        with open('train_cmds_task_gen.pkl', 'rb') as file:
            train_cmds_task_gen = pickle.load(file)
        with open('val_cmds_task_gen.pkl', 'rb') as file:
            val_cmds_task_gen = pickle.load(file)

        # Iterate through train_cmds_task_gen to find matching elements and their indices
        for cmd in train_cmds_task_gen:
            for i, lang_group in enumerate(all_lang):
                if cmd == lang_group[0]:  # Match only the first element of each inner list
                    train_langs.append(lang_group[0])
                    train_data.append(all_token_ids[i].tolist())
                    train_mask.append(all_masks[i].tolist())
        # Iterate through val_cmds_task_gen to find matching elements and their indices
        for cmd in val_cmds_task_gen:
            for i, lang_group in enumerate(all_lang):
                if cmd == lang_group[0]:  # Match only the first element of each inner list
                    val_langs.append(lang_group[0])
                    val_data.append(all_token_ids[i].tolist())
                    val_mask.append(all_masks[i].tolist())

        assert set(train_langs).isdisjoint(set(val_langs)), "The lists have overlapping elements!"

        # Flatten the first two dimensions (3D to 2D)
        train_data = torch.tensor(list(chain.from_iterable(train_data)))
        val_data = torch.tensor(list(chain.from_iterable(val_data)))
        train_mask = torch.tensor(list(chain.from_iterable(train_mask)))
        val_mask = torch.tensor(list(chain.from_iterable(val_mask)))
    else:
        assert(args.test_scene < len(scenes), "Error: input test scene is out of index space")

        # Iterate through all scenes except the test scene
        for x in range(len(scenes)):
            if x != args.test_scene:
                scene_ids = scene_to_ids[scenes[x]]
                scene_masks = scene_to_masks[scenes[x]] 

                np.random.shuffle(scene_ids)
                np.random.shuffle(scene_masks)


                # Stratified split: use 80% for training and 20% for validation
                split_idx = int(train_split * len(scene_ids))
                
                train_data += scene_ids[:split_idx]
                val_data += scene_ids[split_idx:]
                
                train_mask += scene_masks[:split_idx]
                val_mask += scene_masks[split_idx:]

        # The test set is assigned manually based on the test_scene input
        # test_data = scene_to_ids[scenes[args.test_scene]]
        # test_mask = scene_to_masks[scenes[args.test_scene]]


        # Flatten the first two dimensions (3D to 2D)
        train_data = torch.tensor(list(chain.from_iterable(train_data)))
        val_data = torch.tensor(list(chain.from_iterable(val_data)))
        train_mask = torch.tensor(list(chain.from_iterable(train_mask)))
        val_mask = torch.tensor(list(chain.from_iterable(val_mask)))

    DATA= {} 

    DATA['train_data'] = train_data
    DATA['valid_data'] = valid_data
    DATA['train_mask'] = train_mask
    DATA['valid_mask'] = valid_mask

    print(train_data.shape, train_mask.shape ,valid_mask.shape ,valid_data.shape)

    tokenizer.save_pretrained("lambda_tokenizer/lambda_task_gen")

    with open("train_data_lambda/lambda_task_gen.pkl", "wb") as f:
        pickle.dump(DATA , f)