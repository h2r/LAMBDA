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
from . import tokenize_moma as TM 
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel

def load_img_tokenizer(image_token_model):
    
    img_model = tokenize_lanmp.Net(codebook_dim=64, codebook_size=2048).to("cuda")
    ckpt = torch.load(image_token_model, map_location="cuda" )
    img_model.load_state_dict(ckpt  )
    img_model.eval()

    return img_model.to("cuda" )

def convert_img_to_obs_string( img_idx , lang_instruction ):
    a,b,c = img_idx.shape
    token_idx = img_idx.squeeze(0).view(  b*c  )
    image_string = TM.get_image_strings( [token_idx] )
    
    unified_string = "give action command: " + lang_instruction +  image_string[0] + "<eop>"  +  "<toa>" 

    return unified_string

def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def GPT_model( model_path ):
    model = GPT2LMHeadModel.from_pretrained(model_path ).to("cuda")
    model.eval()
    return model

def filter_output( output_string ):

    start = output_string.split("<toa>")[1]
    output = start.split("<eog>")

    return output[0]

def generate( obs_string , tokenizer, checkpoint_path ):
    prompt_token =  tokenizer(obs_string, return_tensors="pt"  )
    GPT = GPT_model(checkpoint_path)
    output_text  = GPT.generate(prompt_token.input_ids.to("cuda") , max_new_tokens=7 , min_new_tokens = 5 ,
    do_sample = True,   num_return_sequences=1, temperature= 1, attention_mask =prompt_token.attention_mask.to("cuda") )
    completed_text = tokenizer.decode(output_text[0], skip_special_tokens=False)

    action = filter_output(completed_text)

    return action

def get_actions( img_tensor , lang_instruction, tokenizer_path, image_token_model, checkpoint_path ):
    tokenizer = get_tokenizer(tokenizer_path)
    img_token_model = load_img_tokenizer(image_token_model)
    _, idx, _= img_token_model((img_tensor/255.0).to("cuda") )
    obs_string = convert_img_to_obs_string(idx , lang_instruction)
    action = generate( obs_string , tokenizer, checkpoint_path )
    return action


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--image_token_model", help="point to the file with the name of all files", default= "img_token_model.pth" , type=str)
    parser.add_argument("--processed_img_data_path" , help="path to the pre processed dataset" , default= "/oscar/home/sharitha/data/datasets/lambda/lanmp_dataset_imgs.pt" )
    parser.add_argument("--checkpoint_path", help="point to the file with the name of all files", default= "/oscar/scratch/sharitha/research/motionglot/tokenizer/ImageTokenizer/pick_place_manip/checkpoint-1200" , type=str)
    parser.add_argument("--tokenizer_path", help=" path to folder with tokenizer " , default= "/oscar/scratch/sharitha/research/motionglot/tokenizer/ImageTokenizer/lambda_tokenizer/moma", type= str ) 

    args = parser.parse_args()

    all_images  = torch.load( args.processed_img_data_path )
    
    ## replace this line with an image as a tensor of shape [1, 3, 224,224 ] 
    ## no need to normalize this the get_actions function would internall handle it
    sample_image= all_images[0, : ].unsqueeze(0).permute( 0,3,1,2 )
    lang_instruction = "move towards the cabinet"
    get_actions(sample_image , lang_instruction )


