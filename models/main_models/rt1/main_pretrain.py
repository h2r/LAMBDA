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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
from data import create_dataset
from rt1_pytorch.rt1_policy import RT1Policy
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=list,
        default=["fractal20220817_data"],
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train[:-1000]",
        help="use e.g. train[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="train[-1000:]",
        help="use e.g. eval[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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
        default=8,
        help="train batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="eval batch size",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=6,
        help="number of frames per trajectory",
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
        "--val_loss_dir",
        type=str,
        default="val_losses/kfold",
        help="directory to save validation losses",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="checkpoint frequency in number of batches; defaults to None. If 0, then saves at best val scores",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/rt1_pretraining",
        help="directory to save checkpoints",
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
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project="rt1-pretraining-v1", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading dataset...")
    
    train_dataset = create_dataset(
        datasets=args.datasets,
        split=args.train_split,
        trajectory_length=args.trajectory_length,
        batch_size=args.train_batch_size,
        num_epochs=args.epochs,
    )
    eval_dataset = create_dataset(
        datasets=args.datasets,
        split=args.eval_split,
        trajectory_length=args.trajectory_length,
        batch_size=args.eval_batch_size,
        num_epochs=args.epochs,
    )

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(low=0, high=255, shape=(256, 320, 3)),
        context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
    )
    action_space = gym.spaces.Dict(
        gripper_closedness_action=gym.spaces.Box(
           low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        ),
        terminate_episode=gym.spaces.Discrete(3),
        base_displacement_vector=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        ),
        rotation_delta=gym.spaces.Box(
            low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
        ),
        base_displacement_vertical_rotation=gym.spaces.Box(
            low=-np.pi / 2.0, high=np.pi / 2.0, shape=(1,), dtype=np.float32
        ),
        world_vector=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    )

    print("Building policy...")
    policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    if args.lr_sched == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.lr_sched == "plateau":
	    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=args.factor, verbose=False)
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else None
    )
    # Total number of params
    total_params = sum(p.numel() for p in policy.model.parameters())
    # Transformer params
    transformer_params = sum(p.numel() for p in policy.model.transformer.parameters())
    # FiLM-EfficientNet and TokenLearner params
    tokenizer_params = sum(p.numel() for p in policy.model.image_tokenizer.parameters())
    print(f"Total params: {total_params}")
    print(f"Transformer params: {transformer_params}")
    print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")

    def get_text_embedding(observation: Dict):
        if text_embedding_model is not None:
            return text_embedding_model.encode(observation["instruction"])
        else:
            return observation["embedding"]

    print("Training...")
    num_batches = 0
    
    best_val_loss  = np.inf
    for batch in tqdm(train_dataset):
        
        policy.model.train()
 
        num_batches += 1

        if num_batches <= 0:
            continue 

        #Image Shape: 8, 6, 480, 640, 3 => (batch, length, height, width, channel)
        #Context Shape: 8, 6, 512 => (batch, length, embedding)
        observations = {
            "image": batch["observation"]["image"],
            "context": get_text_embedding(batch["observation"]),
        }

       
        actions = batch["action"]
        
        try:
            loss, loss_std = policy.loss(observations, actions)
        except Exception as e:
            print('-------------LOSS COMPUTATION FAILED!!!--------')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {e}')
            continue

        if args.wandb:
            wandb.log({"loss": loss.item(), "loss_std": loss_std.item()}, step=num_batches * args.train_batch_size)
        print(f"Train loss Batch {num_batches}: {loss.item()}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.lr_sched == "exponential":
            scheduler.step()

        if args.eval_freq and num_batches % args.eval_freq == 0:
            print("Evaluating...")
            policy.model.eval()
            with torch.no_grad():
                batch = next(eval_dataset)
                observations = {
                    "image": batch["observation"]["image"],
                    "context": get_text_embedding(batch["observation"]),
                }
                actions = batch["action"]
                eval_loss, eval_loss_std = policy.loss(observations, actions)
            eval_loss = eval_loss.item()

            if args.lr_sched == "plateau":
                scheduler.step(eval_loss)

            wandb.log(
                {"eval_loss": eval_loss, "eval_loss_std": eval_loss_std.item()},
                step=num_batches * args.train_batch_size,
            )
            val_dic = {}
            print(f"Eval loss Batch {num_batches}: {eval_loss}")
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                val_dic['best_val_loss'] = eval_loss
                if args.checkpoint_freq == 0:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    checkpoint_path = (f"{args.checkpoint_dir}/checkpoint_best.pt")
                    torch.save(policy.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
            else:
                val_dic['best_val_loss'] = best_val_loss
            val_dic['curr_val_loss'] = eval_loss
        
        
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

    
    checkpoint_path = f"{args.checkpoint_dir}/checkpoint_last.pt"
    torch.save(policy.model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print("Finished Training!")

if __name__ == "__main__":
    main()