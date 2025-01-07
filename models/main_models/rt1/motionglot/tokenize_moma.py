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
from . import tokenize_lanmp


def load_img_tokenizer():

    img_model = tokenize_lanmp.Net(codebook_dim=64, codebook_size=2048).cuda()
    ckpt = torch.load(args.image_token_model, map_location='cuda')
    img_model.load_state_dict(ckpt  )
    img_model.eval()

    return img_model.to("cuda")

def get_max_min_ranges():
    dataset_path = "/users/ajaafar/data/ajaafar/LaNMP-Dataset/models/main_models/rt1/motionglot/pickle_data"
    all_files = os.listdir(dataset_path )
    
    all_yaw = []
    all_x = []
    all_y = []
    all_z = []

    for file in tqdm( all_files):
        if file.split(".")[1] == "pkl":
            path =os.path.join( dataset_path , file )
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

                    if  val == "RotateAgent":
                        yaw = data[i][j][4][ 3]
                        all_yaw.append(yaw)
        
    x_min, x_max = np.min(all_x) , np.max(all_x)
    y_min, y_max = np.min(all_y) , np.max(all_y)
    z_min, z_max = np.min(all_z) , np.max(all_z)
    yaw_min, yaw_max = np.min(all_yaw) , np.max(all_yaw)
    return [yaw_min , yaw_max] , [ x_min, x_max ] , [ y_min, y_max  ] ,[z_min, z_max] 

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


def detokenize_action(discrete_action, index):
    
    def get_val_from_bin_index(index, min_val, max_val, num_bins):
        # Avoid division by zero or invalid input
        if max_val <= min_val or num_bins <= 0:
            raise ValueError("Invalid range or number of bins.")

        # Calculate the bin width
        bin_width = (max_val - min_val) / num_bins
        
        # Compute the continuous value corresponding to the center of the bin
        x = min_val + (index + 0.5) * bin_width
        
        # Clamp the value to ensure it's within range (optional, based on assumptions)
        x = max(min(x, max_val), min_val)
        
        return x


    # yaw, eef_x, eef_y, eef_z = get_max_min_ranges()
    yaw = [-346.2767677307129, 353.14554023742676]
    eef_x = [-0.22499990463256836, 0.3750004768371582]
    eef_y = [-0.2999999523162842, 0.15000009536743164]
    eef_z = [-0.23398804664611816, 0.26000285148620605]
    
    num_bins = 128

    if discrete_action in ["MoveArm", "MoveArmBase"]:
        # if discrete_action == "MoveArmBase":
        #     breakpoint()
        x_delta_val = get_val_from_bin_index(index[0], eef_x[0] , eef_x[1] , num_bins)
        y_delta_val = get_val_from_bin_index(index[1], eef_y[0] , eef_y[1] , num_bins)
        z_delta_val = get_val_from_bin_index(index[2], eef_z[0] , eef_z[1] , num_bins)
        return [x_delta_val, y_delta_val, z_delta_val]
    elif discrete_action == "RotateAgent":
        yaw_delta_val = get_val_from_bin_index(index[0], yaw[0] , yaw[1] , num_bins)
        return yaw_delta_val
    

def load_traj():
    print('AHHHHHHHHHHH!!!!')
    breakpoint()
    all_files = os.listdir(args.dataset_path )
    yaw, eef_x, eef_y, eef_z = get_max_min_ranges()

    all_text = []
    all_imgs = []
    all_lang_instruct = []

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
                for j in range(num_steps):
                    
                    val = data[i][j][3]
                    if val == "MoveArm":
                        x,y,z = data[i][j][4][ 4] ,  data[i][j][4][ 5] ,  data[i][j][4][ 6]
                        x_idx = get_bin_index( x , eef_x[0] , eef_x[1] , args.num_bins )
                        y_idx = get_bin_index( y , eef_y[0] , eef_y[1] , args.num_bins )
                        z_idx = get_bin_index( z , eef_z[0] , eef_z[1] , args.num_bins )

                        string_val = "MoveArm " + str(x_idx) +" , " + str(y_idx) + " , " + str(z_idx)
                    elif val == "MoveArmBase":
                        x,y,z = data[i][j][4][ 4] ,  data[i][j][4][ 5] ,  data[i][j][4][ 6]
                        x_idx = get_bin_index( x , eef_x[0] , eef_x[1] , args.num_bins )
                        y_idx = get_bin_index( y , eef_y[0] , eef_y[1] , args.num_bins )
                        z_idx = get_bin_index( z , eef_z[0] , eef_z[1] , args.num_bins )

                        string_val = "MoveArmBase " + str(x_idx) +" , " + str(y_idx) + " , " + str(z_idx)
                    elif val == "RotateAgent":
                        yaw1 = data[i][j][4][ 3]
                        yaw_idx = get_bin_index( yaw1 , yaw[0] , yaw[1], args.num_bins )

                        string_val = "RotateAgent " + str( yaw_idx )
                    else:
                        string_val = str(val)
                    
                    traj_text.append( string_val)
                    traj_img.append( torch.tensor(data[i][j][2] ).unsqueeze(0)  )
                    lang_instruct.append(data[i][j][1]  )
                
                all_text.append(traj_text)
                all_imgs.append( traj_img )
                all_lang_instruct.append( lang_instruct )
                
    
    return all_imgs , all_text , all_lang_instruct

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



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--image_token_model", help="point to the file with the name of all files", default= "img_token_model.pth" , type=str)
    parser.add_argument("--processed_img_data_path" , help="path to the pre processed dataset" , default= "/oscar/home/sharitha/data/datasets/lambda/lanmp_dataset_imgs.pt" )
    parser.add_argument("--dataset_path", help="point to the file with the name of all files", default= "/oscar/home/sharitha/data/datasets/lambda/" , type=str)
    parser.add_argument("--num_bins", help="point to the file with the name of all files", default= 128 , type=int )
    parser.add_argument("--model_type", help="path to folder with prompts ", default= "openai-community/gpt2"  ,type =str ) # "openai-community/gpt2"

    args = parser.parse_args()
    train_val_split = 0.8 

    tokenizer , VOCAB  = define_vocabulary()
    
    img_token_model = load_img_tokenizer()
    all_img, text_strings , all_lang = load_traj()

    all_token_ids, all_masks = tokenize_images()

    print(len(all_token_ids) , len(all_masks))

    num_traj = len(all_token_ids)
    assert len(all_token_ids) == len(all_masks)

    all_train, all_valid= [] , [] 
    all_train_mask, all_valid_mask =[], []

    for i in tqdm( range(num_traj)):
        split_idx = int( train_val_split*len( all_token_ids[i] ) )
        
        train_data = all_token_ids[i][0:split_idx, :]
        valid_data = all_token_ids[i][split_idx:, : ]
        
        train_mask = all_masks[i][0:split_idx, :]
        valid_mask = all_masks[i][split_idx:, : ]

        all_train.append( train_data)
        all_valid.append( valid_data )
        all_train_mask.append(train_mask )
        all_valid_mask.append( valid_mask )
    
    train_data , valid_data = torch.cat( all_train , dim =0  ) , torch.cat( all_valid , dim =0  )
    train_mask , valid_mask = torch.cat( all_train_mask , dim =0  ) , torch.cat( all_valid_mask , dim =0  )

    DATA= {} 

    DATA['train_data']  = train_data
    DATA['valid_data']  = valid_data
    DATA['train_mask']  = train_mask
    DATA['valid_mask']  = valid_mask

    print(train_data.shape, train_mask.shape ,valid_mask.shape ,valid_data.shape   )

    tokenizer.save_pretrained("lambda_tokenizer/moma")

    with open("train_data_lambda/moma.pkl", "wb") as f:
        pickle.dump(DATA , f)


