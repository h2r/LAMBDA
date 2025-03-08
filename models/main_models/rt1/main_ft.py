import argparse
import os
from typing import Dict
import pdb
import gymnasium as gym
import numpy as np
import torch
import wandb
from sentence_transformers import SentenceTransformer
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import tensorflow_hub as hub 
from data import create_dataset
from rt1_pytorch.rt1_policy import RT1Policy
from tqdm import tqdm
from lanmp_dataloader.rt1_dataloader import DatasetManager, DataLoader
import gc
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_sched",
        default = None,
        choices = ['exponential', 'plateau'],
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.95,
        help="plateau scheduler reduction factor",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default= 25,
        help="plateau scheduler batch patience",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="exponential scheduler step size",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=3,
        help="train batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=3,
        help="eval batch size",
    )
    parser.add_argument(
        "--train-subbatch",
        default=8,
    )
    parser.add_argument(
        "--eval-subbatch",
        default=8,
    )
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default=None,
        help="SentenceTransformer to use; default is None for original USE embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0, #200
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="checkpoint frequency in number of batches; defaults to None. If 0, will save at every best validation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/temp", #"checkpoints/diversity_v1_4"
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--val_loss_dir",
        type=str,
        default="val_losses/kfold",
        help="directory to save validation losses",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="checkpoint to load from; defaults to None",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )
    parser.add_argument(
        "--test-scene",
        default=1,
        help = "scene used as held-out test during k-fold cross",
    )
    parser.add_argument(
        "--split-type",
        default = 'k_fold_scene',
        choices =  ['k_fold_scene', 'task_split','diversity_ablation', 'low_high_scene_ablation', 'cluster_ablation'],
    )
    parser.add_argument(
        "--subset-amt",
        default = None,
        choices =  ['25', '50', '75'],
    )
    parser.add_argument(
        '--low_div', 
        help='low diversity if true, else high', 
        action='store_true'
    )
    parser.add_argument(
        "--num-diversity-scenes",
        default = 4,
    )
    parser.add_argument(
        "--max-diversity-trajectories",
        default = 100,
    )
    parser.add_argument(
        "--use-dist",
        help='use distance input if true, not if false', 
        action='store_true'
    )
    parser.add_argument(
        "--freeze",
        help='if true, freezes layers to finetune the model, if false trains model from scratch', 
        action='store_true'
    )
    return parser.parse_args()


def main():

    args = parse_args()
    args.eval_subbatch = int(args.eval_subbatch)
    args.train_subbatch = int(args.train_subbatch)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.use_dist:
        dist = 'dist'
    else:
        dist='nodist'

    if not args.load_checkpoint:
        train_type = "train"
    else:
        train_type = "ft"

    if args.freeze:
        freeze = "frozen"
    else:
        freeze = "unfrozen" 
    if args.wandb:
        wandb.init(project=f"mamba-{train_type}-{freeze}-{dist}-{args.split_type}-{args.subset_amt}-{args.test_scene}", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading dataset...")

    dataset_manager = DatasetManager(args.subset_amt, args.use_dist, args.test_scene, 0.8, 0.1, 0.1, split_style = args.split_type, diversity_scenes = args.num_diversity_scenes, max_trajectories = args.max_diversity_trajectories, low_div=args.low_div)
    
    if args.wandb and args.split_type == 'diversity_ablation':
        wandb.log({"task_keys": dataset_manager.train_dataset.dataset_keys})
    
    train_dataloader = DataLoader(dataset_manager.train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn= dataset_manager.collate_batches, drop_last = False)
    val_dataloader = DataLoader(dataset_manager.val_dataset, batch_size = args.eval_batch_size, shuffle=True, num_workers=2, collate_fn= dataset_manager.collate_batches, drop_last = False)
    
    if args.use_dist:
        observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
            context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
            ee_obj_dist=gym.spaces.Box(low=float(-1), high=np.inf, shape=(512,), dtype=np.float32), #added
            goal_dist=gym.spaces.Box(low=float(-1), high=np.inf, shape=(512,), dtype=np.float32) #added2
        )
    else:
        observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
            context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32)
        )

    action_space = gym.spaces.Dict(

        body_yaw_delta = gym.spaces.Box(
            low= 0, #train_dataloader.body_orientation_lim['min_yaw']
            high= 255, #train_dataloader.body_orientation_lim['max_yaw']
            shape=(1,), 
            dtype=int
        ),

        terminate_episode=gym.spaces.Discrete(2),

        arm_position_delta = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (3,),
            dtype = np.int32
        ),

        control_mode = gym.spaces.Discrete(12),
       
    )

    print("Building policy...")
    policy = RT1Policy(
        dist=args.use_dist,
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    

    if args.freeze:
        # Freeze all layers except the last one
        unfrozen_keywords = [
        "action_encoder", "transformer.encoder.layers.0", "transformer.encoder.layers.1", 
        "transformer.encoder.layers.2", "transformer.encoder.layers.3", 
        "transformer.decoder.layers.0", "transformer.decoder.layers.1", 
        "transformer.decoder.layers.2", "transformer.decoder.layers.3", 
        "transformer.encoder.norm", "transformer.decoder.norm", "to_logits"
        ]

        # Freeze all parameters except those containing the keywords
        for name, param in policy.model.named_parameters():
            if any(keyword in name for keyword in unfrozen_keywords):
                param.requires_grad = True
                # print(f"Unfrozen: {name}")
            else:
                param.requires_grad = False
                print(f"Frozen: {name}")   
    
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    if args.lr_sched:
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)

    #NOTE: has to be Not None because of raw instruction input
    
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
    )
    
    
    # # Total number of params
    # total_params = sum(p.numel() for p in policy.model.parameters())
    # # Transformer params
    # transformer_params = sum(p.numel() for p in policy.model.transformer.parameters())
    # # FiLM-EfficientNet and TokenLearner params
    # tokenizer_params = sum(p.numel() for p in policy.model.image_tokenizer.parameters())
    # print(f"Total params: {total_params}")
    # print(f"Transformer params: {transformer_params}")
    # print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")

    def get_text_embedding(observation: Dict):
        
        if args.sentence_transformer is not None:
            return text_embedding_model.encode(observation)
        else:
            embedded_observation = []

            for i in range(0, observation.shape[1]):
                
                try:
                    embedded_observation.append( np.array(text_embedding_model(observation[:, i]) ) )
                except:
                    print('EMBEDDING FAILED!')
                    # breakpoint()
                    
            # try:
            embedded_observation = np.stack(embedded_observation, axis=1)
            # except:
                # breakpoint()
            return embedded_observation

    print("Training...")
    num_batches = 0
    total_train_steps = 0
    total_val_steps = 0

    
    best_val_loss  = np.inf
    for epoch in range(args.epochs):
        print("STARTING EPOCH {}".format(epoch+1))
        
        for batch, train_batch in enumerate(train_dataloader):
            batch_steps = train_batch[0].shape[0]

            for idx in range(0, batch_steps, int(args.train_subbatch)):
                
                policy.model.train() #put into training mode
                
                num_batches += 1
                
                try:
                    res = get_text_embedding(train_batch[1][idx : min(idx + int(args.train_subbatch), batch_steps)])
                except:
                    breakpoint()
                
                if args.use_dist:
                    observations = {
                        "image": train_batch[0][idx : min(idx + int(args.train_subbatch), batch_steps)],
                        "context": res,
                        "ee_obj_dist": train_batch[6][idx : min(idx + int(args.train_subbatch), batch_steps)], #added
                        "goal_dist": train_batch[7][idx : min(idx + int(args.train_subbatch), batch_steps)] #added2
                    }
                else:
                    observations = {
                        "image": train_batch[0][idx : min(idx + int(args.train_subbatch), batch_steps)],
                        "context": res,
                    }
                actions = {
                    'terminate_episode': train_batch[2][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'body_yaw_delta': train_batch[3][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'arm_position_delta': train_batch[4][idx : min(idx + int(args.train_subbatch), batch_steps)],
                    'control_mode': train_batch[5][idx : min(idx + int(args.train_subbatch), batch_steps)]
                }

                padding = train_batch[-1][idx : min(idx + int(args.train_subbatch), batch_steps)]
                total_train_steps += batch_steps

                loss, loss_std = policy.loss(observations, actions)

                if args.wandb:
                    print(f"Train loss Batch {num_batches}: {loss.item()} ± {loss_std.item()}")
                    wandb.log({"train_loss": loss.item(), "train_loss_std": loss_std.item()}, step= total_train_steps)
                else:
                    print(f"Train loss Batch {num_batches}: {loss.item()}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.lr_sched == "exponential":
                    scheduler.step()
                observations = {}; actions = {}

                
                if args.eval_freq and num_batches % args.eval_freq == 0:
                    
                    # Clear cache and collected garbage
                    gc.collect()
                    torch.cuda.empty_cache()

                    total_eval_loss = 0
                    total_eval_loss_std = 0
                    total_eval_count = 0

                    print("Evaluating...")
                    # batches
                    for batch, val_batch in enumerate(val_dataloader):
                        
                        batch_steps_val = val_batch[0].shape[0]

                        print(f'Section {batch+1} of {len(val_dataloader)}')
                        # subbatches
                        for idx in tqdm(range(0, batch_steps_val, args.eval_subbatch)):

                            
                            policy.model.eval()
                            

                            total_val_steps += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)].shape[0]
                            total_eval_count += val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)].shape[0]
                            
                            with torch.no_grad():
                                res = get_text_embedding(val_batch[1][idx : min(idx + args.eval_subbatch, batch_steps_val)])
                                
                                if args.use_dist:
                                    observations = {
                                        "image": val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                        "context": res,
                                        "ee_obj_dist": val_batch[6][idx : min(idx + int(args.eval_subbatch), batch_steps_val)], #added
                                        "goal_dist": val_batch[7][idx : min(idx + int(args.eval_subbatch), batch_steps_val)] #added2
                                    }
                                else:
                                    observations = {
                                        "image": val_batch[0][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                        "context": res,
                                    }

                                actions = {
                                    'terminate_episode': val_batch[2][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                    'body_yaw_delta': val_batch[3][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                    'arm_position_delta': val_batch[4][idx : min(idx + args.eval_subbatch, batch_steps_val)],
                                    'control_mode': val_batch[5][idx : min(idx + args.eval_subbatch, batch_steps_val)]
                                }
                                
                                padding = val_batch[-1][idx : min(idx + args.eval_subbatch, batch_steps_val)]

                                eval_loss, eval_loss_std = policy.loss(observations, actions) #subbatch eval loss
                            
                            
                            total_eval_loss += eval_loss.item()*observations['image'].shape[0]
                            total_eval_loss_std += np.power(eval_loss_std.item(), 2)*observations['image'].shape[0]
                    
                    avg_eval_loss = total_eval_loss / total_eval_count
                    avg_eval_loss_std = np.sqrt(total_eval_loss_std / total_eval_count)

                    if args.lr_sched == "plateau":
                        scheduler.step(avg_eval_loss)

                    if args.wandb:
                        wandb.log(
                            {"eval_loss": avg_eval_loss, "eval_loss_std": avg_eval_loss_std},
                            step=total_train_steps,
                        )
                        val_dic = {}
                        print(f"Eval loss Batch {num_batches}: {avg_eval_loss}")
                        if avg_eval_loss < best_val_loss:
                            best_val_loss = avg_eval_loss
                            val_dic['best_val_loss'] = avg_eval_loss

                            if args.checkpoint_freq == 0:
                                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_best.pt"
                                torch.save(policy.model.state_dict(), checkpoint_path)
                                print(f"Saved checkpoint to {checkpoint_path}")
                        else:
                            val_dic['best_val_loss'] = best_val_loss
                        val_dic['curr_val_loss'] = avg_eval_loss
                    else:
                        val_dic = {}
                        print(f"Eval loss Batch {num_batches}: {avg_eval_loss}")
                        if avg_eval_loss < best_val_loss:
                            best_val_loss = avg_eval_loss
                            val_dic['best_val_loss'] = avg_eval_loss

                            if args.checkpoint_freq == 0:
                                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_best.pt"
                                torch.save(policy.model.state_dict(), checkpoint_path)
                                print(f"Saved checkpoint to {checkpoint_path}")
                        else:
                            val_dic['best_val_loss'] = best_val_loss
                        val_dic['curr_val_loss'] = avg_eval_loss


                    os.makedirs(args.val_loss_dir, exist_ok=True)
                    with open(f'{args.val_loss_dir}/val_loss_batch_{num_batches}.json', 'w') as json_file:
                        json.dump(val_dic, json_file, indent=4)

                    
                if args.checkpoint_freq and num_batches % args.checkpoint_freq == 0:
                    checkpoint_path = (
                        f"{args.checkpoint_dir}/checkpoint_"
                        + f"{num_batches}"
                        + f".pt"
                    )
                    torch.save(policy.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
        
        print("FINISHED EPOCH {}".format(epoch+1))
    
    checkpoint_path = f"{args.checkpoint_dir}/checkpoint_last.pt"
    torch.save(policy.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print("Finished Training!")

if __name__ == "__main__":
    main()