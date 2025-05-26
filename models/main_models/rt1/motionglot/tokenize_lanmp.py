import os
import json
import torch
import torch.optim as optim
import warnings
import torch 
import numpy as np  
import argparse
import os 
from tqdm import tqdm 
import random
from transformers import GPT2Tokenizer
import configparser 
from os.path import join as pjoin
from transformers import AutoTokenizer
import random 
import h5py
import re 
from skimage.io import imsave, imread
from .imagetokenizer.imagetokenizer.model import OmniTokenizer
from .imagetokenizer.imagetokenizer.model import Magvit2Tokenizer
from .imagetokenizer.imagetokenizer.model import TiTok
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
import torchvision.transforms as T
from PIL import Image
from vector_quantize_pytorch import VectorQuantize, Sequential
import torch.nn as nn
from tqdm.auto import trange
import pickle
from accelerate import Accelerator
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.nn.parallel import DistributedDataParallel as DDP


def sort_folders(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def prepare_dataset():

    with torch.no_grad():
        resize_transform = T.Resize((224, 224 ))
        all_imgs = []

        with h5py.File(args.dataset_path , 'r') as hdf_file:
            # Iterate through each trajectory group
            for trajectory_name, trajectory_group in tqdm( hdf_file.items()):
                traj_steps = list(trajectory_group.keys())
                traj_steps.sort(key=sort_folders) 

                start = 0; end =  len(traj_steps) #min(len(traj_steps), 6)
                terminate = False

                for i in range(start, end):
                    ith_obs = np.array(trajectory_group[traj_steps[i]]['rgb_{}'.format(i)])
                    img_data = resize_transform(torch.tensor( ith_obs , dtype=torch.float32 ).permute(2,0,1).unsqueeze(0) )
                    all_imgs.append(img_data)

            data = torch.cat( all_imgs , dim=0 )
            torch.save( data, args.processed_data_path ) 

class Lnmap_img_dataloader( Dataset ):
    def __init__(self):

        self.img_datapath = args.processed_data_path
        self.all_imgs = torch.load( self.img_datapath )
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index ):
        return self.all_imgs[index].to("cuda")/255.0

class Net(nn.Module):
    def __init__(self, codebook_dim=128, codebook_size=512, vq_kwargs=None):
        """
        codebook_dim  : dimension of the embeddings in vector quantization
        codebook_size : number of codes in the codebook
        vq_kwargs     : any additional keyword args to pass into VectorQuantize
        """
        super().__init__()
        if vq_kwargs is None:
            vq_kwargs = {}

        # ---- Encoder (downsample by 8x) ----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down 2×

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down 4×

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down 8×
        )

        # ---- Vector Quantization bottleneck ----
        self.vq = VectorQuantize(
            dim=codebook_dim,          # 64 by default here
            codebook_size=codebook_size,  # 512 by default
            accept_image_fmap=True,    # important if input to VQ is a feature map
            **vq_kwargs
        )

        # ---- Decoder (upsample from 8× back to original size) ----
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # Up from 8× to 4×
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # Up from 4× to 2×
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # Up from 2× to 1× (original resolution)
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Vector-quantize (returns tuple of quantized tensor, indices, commit_loss)
        # You can either do: x, indices, commit_loss = self.vq(x)
        x , indices, commit_loss = self.vq(x)  # we only need the quantized output for forward pass

        # Decode
        x = self.decoder(x)
        return x ,  indices, commit_loss

def train(model, train_traj, valid_traj ):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_batches = len(train_traj)
    criterion = torch.nn.L1Loss()
    model , optimizer = accelerator.prepare(model , optimizer  )

    if(accelerator.is_main_process):
        num_batches = num_batches 

    for eph in tqdm(range(args.num_epochs)):
        TotalLoss = 0

        for i in range(num_batches ):
            optimizer.zero_grad()

            input_data = (train_traj[i].permute(0,3,1,2)/255.0).to(args.device)
            out, indices, cmt_loss = model(input_data)
            out = torch.clip(out  , 0, 1 )

            rec_loss = criterion(out , input_data)
            Loss = rec_loss + 10.0*cmt_loss

            accelerator.backward(Loss)
            accelerator.wait_for_everyone()
            optimizer.step()  
            optimizer.zero_grad()
            accelerator.wait_for_everyone()
            TotalLoss += Loss 
        accelerator.log({"loss":TotalLoss })
        if(accelerator.is_main_process and eph %10 ==0 ):
            
            input_data = valid_traj.to("cuda")/255.0
            valid_data = input_data.to(args.device).permute(0,3,1,2)

            model.eval()
            with torch.no_grad():
                
                reconstruction , _, cmt_loss = model.forward(valid_data )
                valid_loss = criterion(reconstruction , valid_data) +10* cmt_loss 
                img = reconstruction.permute(0,2,3,1).to("cpu").numpy()[0, : ]
                im1 = Image.fromarray( (img*255.0).astype(np.uint8) ).save("output.png")

                model_save_path = "img_token_model" + ".pth"
                model1= accelerator.get_state_dict(model)
                torch.save(model1 ,model_save_path)
                accelerator.log({"valid_loss":valid_loss })  
            model.train()

def get_data(data_path):

    print(data_path)

    with open(data_path , "rb") as f:
        data = pickle.load(f)
    
    return data

def process_data():

    all_files = os.listdir(args.dataset_path )

    all_imgs = []

    for file in tqdm( all_files):
        path =os.path.join( args.dataset_path , file )
        data =  get_data(path )

        num_traj = len(data)

        for i in range(num_traj):
            num_steps = len( data[i] )
            for j in range(num_steps):
                all_imgs.append( torch.tensor(data[i][j][2] ).unsqueeze(0)  )
    
    data = torch.cat(all_imgs , dim =0 )
    torch.save( data , args.processed_data_path )

def load_images( data_path , batch_size, device_id ):

    trajectories  = torch.load( data_path )
    train_traj , valid_traj = trajectories[:-100, : ] , trajectories[-100:, : ]

    start_idx = np.arange( 0, len(train_traj) , batch_size )
    end_idx = start_idx + batch_size

    batch_split_traj = [] 

    for i in range( len(start_idx)):
        st = start_idx[i]
        ed = end_idx[i]

        batch_data = train_traj[st:ed , : ]
        batch_split_traj.append( batch_data )
    
    device_split_data = split_list( batch_split_traj , 6 )

    return device_split_data[device_id] , valid_traj

def split_list(input_list, n):
    # Determine the length of each sublist
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--image_token_model", help="point to the file with the name of all files", default= "fun-research/TiTok" , type=str)
    parser.add_argument("--dataset_path", help="point to the file with the name of all files", default= None , type=str)
    parser.add_argument("--processed_data_path" , help="path to the pre processed dataset" , default= "/oscar/home/sharitha/data/datasets/lambda/lanmp_dataset_imgs.pt" )
    parser.add_argument("--prepare_dataset", help="set to 1 to prepare dataset", default= 0 , type=int )
    parser.add_argument("--batch_size", help="set to 1 to prepare dataset", default= 256 , type=int )
    parser.add_argument("--num_epochs", help="set the number of epochs" , default= 100 , type =int )
    parser.add_argument("--lr", help="learning rate" , default= 1e-4 , type=float)
    parser.add_argument("--start_wandb", help="start wandb " , default=1 , type =int )

    args = parser.parse_args()
    assert(args.dataset_path != None, "Error, LAMBDA simulated dataset path not provided")

    # img_token_model = get_image_tokenizer()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False ,  gradient_as_bucket_view=False) 
    if( args.start_wandb == 1):
        print("Initilized Wandb ")
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs ] , log_with="wandb" )
        accelerator.init_trackers(
            "QUANT_LAM",
            config={
            "learning_rate": args.lr,
            "architecture": "Robot VQVAE TRAINING",
            }
        )
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs ])
    
    if args.prepare_dataset == 1:
        # prepare_dataset()
        process_data()

    
    dataset = Lnmap_img_dataloader()

    lr = 3e-4
    train_iter = 10000
    num_codes = 512
    seed = 1234
    rotation_trick = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = accelerator.device
    device_id = int (str(args.device).split(":")[1])

    train_traj, valid_traj = load_images(args.processed_data_path , args.batch_size , device_id )

    model = Net(codebook_dim=64, codebook_size=2048).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train(model, train_traj, valid_traj )



